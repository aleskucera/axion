from math import sqrt
from typing import Any
from typing import Optional

import warp as wp
from warp.optim.linear import _Matrix
from warp.optim.linear import aslinearoperator
from warp.utils import array_inner

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})


@wp.kernel
def _cg_kernel_1_fixed(
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    """Graph-compatible CG kernel part 1: Updates x and r."""
    i = wp.tid()

    # Unconditionally compute alpha, with a safe division
    alpha = rz_old.dtype(0.0)
    if p_Ap[0] != 0.0:
        alpha = rz_old[0] / p_Ap[0]

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]


@wp.kernel
def _cr_kernel_1_fixed(
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
):
    """Graph-compatible CR kernel part 1: Updates x, r, and z."""
    i = wp.tid()

    # Unconditionally compute alpha, with a safe division
    alpha = zAz_old.dtype(0.0)
    if y_Ap[0] > 0.0:
        alpha = zAz_old[0] / y_Ap[0]

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]
    z[i] = z[i] - alpha * y[i]


@wp.kernel
def _cr_kernel_2_fixed(
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Az: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    """Graph-compatible CR kernel part 2: Updates p and Ap."""
    i = wp.tid()

    # Unconditionally compute beta, with a safe division
    beta = zAz_old.dtype(0.0)
    if zAz_old[0] > 0.0:
        beta = zAz_new[0] / zAz_old[0]

    p[i] = z[i] + beta * p[i]
    Ap[i] = Az[i] + beta * Ap[i]


def cr_solver_graph_compatible(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    iters: int,
    preconditioner: Optional[_Matrix] = None,
    compute_final_residual: bool = True,
) -> Optional[float]:
    """
    Computes an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Residual algorithm for a **fixed number of iterations**.
    This version is designed to be captured by a CUDA graph.

    Args:
        A: The linear system's left-hand-side.
        b: The linear system's right-hand-side.
        x: Initial guess and final solution vector (updated in-place).
        iters: The fixed number of iterations to perform.
        M: Optional left-preconditioner.
        compute_final_residual: If True, computes and returns the L2 norm of the
                                final residual (r = b - Ax). This requires an
                                extra mat-vec multiply and synchronization after the loop.

    Returns:
        The final residual norm if `compute_final_residual` is True, otherwise None.
        The solution `x` is always updated in-place.
    """

    if iters <= 0:
        return 0.0

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)
    vec_dim = x.shape[0]

    # ============== 1. Initialization ==========================================
    # This part runs once before the main loop.

    # Residual r = b - Ax
    r = wp.empty_like(b)
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    # Notations below follow the Conjugate Residual method pseudo-code.
    # z := M^-1 r
    # y := M^-1 Ap
    if preconditioner is None:
        z = wp.clone(r)
    else:
        z = wp.zeros_like(r)
        preconditioner.matvec(r, z, z, alpha=1.0, beta=0.0)

    Az = wp.zeros_like(b)
    A.matvec(z, Az, Az, alpha=1.0, beta=0.0)

    p = wp.clone(z)
    Ap = wp.clone(Az)

    if preconditioner is None:
        # y is not used in the non-preconditioned case, but we allocate it
        # to keep the kernel signatures consistent if needed in a mixed setup.
        # Here we just make it an alias to avoid allocation.
        y = Ap
    else:
        y = wp.zeros_like(Ap)

    # Scalar values stored in single-element arrays
    zAz_old = wp.empty(n=1, dtype=scalar_dtype, device=device)
    zAz_new = wp.empty(n=1, dtype=scalar_dtype, device=device)
    y_Ap = wp.empty(n=1, dtype=scalar_dtype, device=device)

    # Initial dot product: zAz = dot(z, Az)
    array_inner(z, Az, out=zAz_new)

    # ============== 2. Fixed Iteration Loop ====================================
    # This loop is designed to be captured by a CUDA graph.
    # It contains no CPU-side logic or synchronization.

    for _ in range(iters):
        # The value from the previous iteration becomes the "old" value for this one.
        wp.copy(dest=zAz_old, src=zAz_new)

        # Preconditioned case
        if preconditioner is not None:
            # y = M^-1 * Ap
            preconditioner.matvec(Ap, y, y, alpha=1.0, beta=0.0)
            array_inner(Ap, y, out=y_Ap)

            # Update x, r, z
            wp.launch(
                kernel=_cr_kernel_1_fixed,
                dim=vec_dim,
                device=device,
                inputs=[zAz_old, y_Ap, x, r, z, p, Ap, y],
            )
        # Non-preconditioned case (updates are simpler, like Conjugate Gradient)
        else:
            # dot(Ap, p) since y=Ap and z=r
            array_inner(Ap, p, out=y_Ap)  # Using y_Ap as a temp buffer for dot(Ap, p)

            # Update x, r
            wp.launch(
                kernel=_cg_kernel_1_fixed,
                dim=vec_dim,
                device=device,
                # In this case zAz_old is rz_old from the CG algorithm
                inputs=[zAz_old, y_Ap, x, r, p, Ap],
            )
            # Update z (which is an alias for r)
            wp.copy(dest=z, src=r)

        # Az = A * z
        A.matvec(z, Az, Az, alpha=1.0, beta=0.0)
        # zAz_new = dot(z, Az)
        array_inner(z, Az, out=zAz_new)

        # Update p, Ap
        wp.launch(
            kernel=_cr_kernel_2_fixed,
            dim=vec_dim,
            device=device,
            inputs=[zAz_old, zAz_new, z, p, Az, Ap],
        )

    # # ============== 3. Finalization (Optional) ===============================
    # if compute_final_residual:
    #     # To get the true final residual, we must recompute r = b - Ax
    #     A.matvec(x, b, r, alpha=-1.0, beta=1.0)
    #     r_norm_sq = wp.empty(n=1, dtype=scalar_dtype, device=device)
    #     array_inner(r, r, out=r_norm_sq)
    #
    #     # Synchronize to get the final result back to the CPU
    #     wp.synchronize()
    #     return sqrt(r_norm_sq.numpy()[0])

    return None


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    wp.init()

    device = wp.get_preferred_device()
    n = 256
    dtype = wp.float32

    # Create a simple SPD matrix A (diagonal) and a right-hand side b
    a_diag = wp.zeros(n, dtype=dtype, device=device)
    wp.launch(
        kernel=lambda x: x.assign(wp.tid() + 1.0), dim=n, inputs=[a_diag], device=device
    )
    A = wp.mat_from_diag(a_diag)

    b = wp.full(n, 1.0, dtype=dtype, device=device)
    x_initial = wp.zeros(n, dtype=dtype, device=device)  # Initial guess

    # --- Run the solver directly ---
    print("--- Running solver directly ---")
    x_direct = wp.clone(x_initial)
    final_res_direct = cr_solver_graph_compatible(
        A, b, x_direct, iters=50, preconditioner=None
    )
    wp.synchronize()
    print(f"Final residual after 50 iterations (direct run): {final_res_direct:.6e}")

    # --- Run the solver using a CUDA Graph ---
    if device.is_cuda:
        print("\n--- Capturing and running solver with CUDA Graph ---")
        x_graph = wp.clone(x_initial)

        # Capture the solver execution into a graph
        wp.capture_begin(device)
        try:
            cr_solver_graph_compatible(
                A,
                b,
                x_graph,
                iters=50,
                preconditioner=None,
                compute_final_residual=False,
            )
        finally:
            graph = wp.capture_end(device)

        # Warm-up (optional, but good practice)
        wp.capture_launch(graph)
        wp.synchronize()

        # Time the execution
        # Re-initialize x to zero before the timed run
        x_graph.zero_()

        start_time = wp.ScopedTimer("CUDA Graph Execution")
        wp.capture_launch(graph)
        wp.synchronize()
        # The end of the ScopedTimer block will print the elapsed time

        # Manually compute final residual to verify the result
        final_r = wp.empty_like(b)
        A.matvec(x_graph, b, final_r, alpha=-1.0, beta=1.0)
        final_r_norm_sq = wp.zeros(1, dtype=dtype, device=device)
        array_inner(final_r, final_r_norm_sq)
        wp.synchronize()
        final_res_graph = sqrt(final_r_norm_sq.numpy()[0])
        print(f"Final residual after 50 iterations (graph run): {final_res_graph:.6e}")

        # Check that results are consistent
        assert (
            abs(final_res_direct - final_res_graph) < 1e-5
        ), "Direct and graph results should match"
        print("Direct and graph-based results are consistent.")
