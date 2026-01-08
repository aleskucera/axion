from typing import Any
from typing import Optional

import warp as wp
from axion.tiled import TiledDot
from warp.optim.linear import LinearOperator


@wp.kernel
def _cr_kernel_1(
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any, ndim=2),
    r: wp.array(dtype=Any, ndim=2),
    z: wp.array(dtype=Any, ndim=2),
    p: wp.array(dtype=Any, ndim=2),
    Ap: wp.array(dtype=Any, ndim=2),
    y: wp.array(dtype=Any, ndim=2),
):
    world_idx, i = wp.tid()

    # Use separate load to ensure previous value is read
    zAz_val = zAz_old[world_idx]
    yAp_val = y_Ap[world_idx]

    alpha = zAz_val.dtype(0.0)
    if yAp_val > 0.0:
        alpha = zAz_val / yAp_val

    x[world_idx, i] = x[world_idx, i] + alpha * p[world_idx, i]
    r[world_idx, i] = r[world_idx, i] - alpha * Ap[world_idx, i]
    z[world_idx, i] = z[world_idx, i] - alpha * y[world_idx, i]


@wp.kernel
def _cr_kernel_2(
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any, ndim=2),
    p: wp.array(dtype=Any, ndim=2),
    Az: wp.array(dtype=Any, ndim=2),
    Ap: wp.array(dtype=Any, ndim=2),
):
    world_idx, i = wp.tid()

    zAz_old_val = zAz_old[world_idx]
    zAz_new_val = zAz_new[world_idx]

    beta = zAz_old_val.dtype(0.0)
    if zAz_old_val > 0.0:
        beta = zAz_new_val / zAz_old_val

    p[world_idx, i] = z[world_idx, i] + beta * p[world_idx, i]
    Ap[world_idx, i] = Az[world_idx, i] + beta * Ap[world_idx, i]


@wp.kernel
def _check_convergence_kernel(
    r_sq: wp.array(dtype=Any, ndim=2),  # Shape (N, 1)
    b_sq: wp.array(dtype=Any, ndim=2),  # Shape (N, 1)
    tol_sq: Any,  # Scalar float (rel_tol^2)
    atol_sq: Any,  # Scalar float (atol^2)
    iter_count: wp.array(dtype=int),  # Shape (1,)
    max_iters: int,
    keep_running: wp.array(dtype=int),  # Shape (1,)
):
    """
    Checks if simulation should continue.
    Logic:
      - Increment iteration count.
      - If iter_count >= max_iters, stop.
      - Else, check residuals. If ANY world is not converged, set keep_running = 1.
    """
    # 1. Update Iteration Count (only one thread does this to avoid race)
    tid = wp.tid()
    if tid == 0:
        # Increment
        current_iter = iter_count[0] + 1
        iter_count[0] = current_iter

        # Check max iters
        if current_iter >= max_iters:
            keep_running[0] = 0
            return

    # 2. Check Convergence (All threads)
    # If we are already forced to stop by max_iters (checked above),
    # this part effectively doesn't matter for control flow, but we execute to be safe.
    # However, to properly sync, we rely on the atomic write to keep_running.

    # We need to ensure we don't overwrite the '0' from max_iters with a '1' here
    # if max_iters was reached.
    current_iter = iter_count[0]
    if current_iter >= max_iters:
        return

    # Effective tolerance squared: max(atol^2, tol^2 * |b|^2)
    # Note: b_sq is |b|^2

    row, col = wp.tid()

    # Check bounds (since this might launch with more threads than num_worlds if using block dims)
    if row >= r_sq.shape[0]:
        return

    val_r_sq = r_sq[row, 0]
    val_b_sq = b_sq[row, 0]

    # target = max(atol^2, tol^2 * |b|^2)
    rel_term = val_b_sq * tol_sq
    target = atol_sq
    if rel_term > target:
        target = rel_term

    # If residual is too high, we must continue
    if val_r_sq > target:
        keep_running[0] = 1


@wp.kernel
def _print_stats(iter_count: wp.array(dtype=int)):
    wp.printf("CR iterations: %d\n", iter_count[0])


class CRSolver:
    """
    A persistent Conjugate Residual solver that pre-allocates memory
    to avoid overhead during simulation steps.
    Includes GPU-side early exit using CUDA Graphs.
    """

    def __init__(self, num_worlds, vec_dim, dtype, device):
        self.num_worlds = num_worlds
        self.vec_dim = vec_dim
        self.dtype = dtype
        self.device = device
        self.scalar_dtype = wp.types.type_scalar_type(dtype)

        # Pre-allocate vectors (N x D)
        shape = (num_worlds, vec_dim)
        self.r = wp.zeros(shape=shape, dtype=dtype, device=device)
        self.z = wp.zeros(shape=shape, dtype=dtype, device=device)
        self.Az = wp.zeros(shape=shape, dtype=dtype, device=device)
        self.p = wp.zeros(shape=shape, dtype=dtype, device=device)
        self.Ap = wp.zeros(shape=shape, dtype=dtype, device=device)
        self.y = wp.zeros(shape=shape, dtype=dtype, device=device)

        # Pre-allocate scalars (N)
        self.zAz_old = wp.zeros(shape=(num_worlds), dtype=self.scalar_dtype, device=device)
        self.zAz_new = wp.zeros(shape=(num_worlds), dtype=self.scalar_dtype, device=device)
        self.y_Ap = wp.zeros(shape=(num_worlds), dtype=self.scalar_dtype, device=device)

        # Tolerance buffers (N x 1)
        self.b_sq = wp.zeros(shape=(num_worlds, 1), dtype=self.scalar_dtype, device=device)
        self.r_sq = wp.zeros(shape=(num_worlds, 1), dtype=self.scalar_dtype, device=device)

        # Loop control buffers
        self.keep_running = wp.zeros(shape=(1,), dtype=int, device=device)
        self.iter_count = wp.zeros(shape=(1,), dtype=int, device=device)

        # Pre-allocate Dot Product helper
        self.tiled_dot = TiledDot(
            shape=(num_worlds, 1, vec_dim),
            dtype=self.scalar_dtype,
            tile_size=256,
            device=device,
        )

    def solve(
        self,
        A: LinearOperator,
        b: wp.array,
        x: wp.array,
        iters: int,
        tol: float = 1e-5,
        atol: float = 1e-7,
        M: Optional[LinearOperator] = None,
    ):
        """
        Solves Ax = b using Conjugate Residual method.
        Exits early if converged (checked on GPU).
        """
        # Ensure dimension match
        if x.shape[0] != self.num_worlds or x.shape[1] != self.vec_dim:
            raise ValueError(
                f"CRSolver dimension mismatch. Expected ({self.num_worlds}, {self.vec_dim}), got {x.shape}"
            )

        # Handle "fixed" strategy (negative tolerances)
        tol_sq = float(tol**2) if tol >= 0 else -1.0
        atol_sq = float(atol**2) if atol >= 0 else -1.0

        # --- 1. Initialization (Outside Graph) ---

        # Reset loop counters
        self.keep_running.fill_(1)
        self.iter_count.zero_()

        # Compute |b|^2 for relative tolerance
        b_view = b.reshape((self.num_worlds, 1, self.vec_dim))
        self.tiled_dot.compute(b_view, b_view, self.b_sq)

        # r = b - Ax
        A.matvec(x, b, self.r, alpha=-1.0, beta=1.0)

        # z := M^-1 r
        if M is None:
            wp.copy(dest=self.z, src=self.r)
        else:
            M.matvec(self.r, self.z, self.z, alpha=1.0, beta=0.0)

        # Az = A * z
        A.matvec(self.z, self.Az, self.Az, alpha=1.0, beta=0.0)

        # p = z
        wp.copy(dest=self.p, src=self.z)

        # Ap = Az
        wp.copy(dest=self.Ap, src=self.Az)

        # y handling (alias or separate buffer)
        y_vec = self.Ap
        if M is not None:
            y_vec = self.y

        # Initial zAz
        self.tiled_dot.compute(
            self.z.reshape((self.num_worlds, 1, self.vec_dim)),
            self.Az.reshape((self.num_worlds, 1, self.vec_dim)),
            self.zAz_new.reshape((self.num_worlds, 1)),
        )

        # --- 2. GPU Loop Definition ---

        def solver_step():
            # Optimistically assume we will finish this step
            # If any residual > tol, the kernel will set this back to 1.
            self.keep_running.zero_()

            # --- CR Iteration ---
            wp.copy(dest=self.zAz_old, src=self.zAz_new)

            if M is not None:
                M.matvec(self.Ap, self.y, self.y, alpha=1.0, beta=0.0)

            # y_Ap = dot(y, Ap)
            self.tiled_dot.compute(
                y_vec.reshape((self.num_worlds, 1, self.vec_dim)),
                self.Ap.reshape((self.num_worlds, 1, self.vec_dim)),
                self.y_Ap.reshape((self.num_worlds, 1)),
            )

            # Update x, r, z
            wp.launch(
                kernel=_cr_kernel_1,
                dim=(self.num_worlds, self.vec_dim),
                device=self.device,
                inputs=[self.zAz_old, self.y_Ap, x, self.r, self.z, self.p, self.Ap, y_vec],
            )

            # Az = A * z
            A.matvec(self.z, self.Az, self.Az, alpha=1.0, beta=0.0)

            # zAz_new = dot(z, Az)
            self.tiled_dot.compute(
                self.z.reshape((self.num_worlds, 1, self.vec_dim)),
                self.Az.reshape((self.num_worlds, 1, self.vec_dim)),
                self.zAz_new.reshape((self.num_worlds, 1)),
            )

            # Update p, Ap
            wp.launch(
                kernel=_cr_kernel_2,
                dim=(self.num_worlds, self.vec_dim),
                device=self.device,
                inputs=[self.zAz_old, self.zAz_new, self.z, self.p, self.Az, self.Ap],
            )

            # --- Convergence Check ---

            # Compute |r|^2
            r_view = self.r.reshape((self.num_worlds, 1, self.vec_dim))
            self.tiled_dot.compute(r_view, r_view, self.r_sq)

            # Check residuals and update iteration count
            wp.launch(
                kernel=_check_convergence_kernel,
                dim=(self.num_worlds),
                device=self.device,
                inputs=[
                    self.r_sq,
                    self.b_sq,
                    tol_sq,
                    atol_sq,
                    self.iter_count,
                    iters,
                    self.keep_running,
                ],
            )

        # --- 3. Capture and Launch ---

        # This captures the loop into a CUDA graph and executes it until keep_running is 0
        wp.capture_while(self.keep_running, solver_step)

        # # Print iteration count from GPU
        # wp.launch(
        #     kernel=_print_stats,
        #     dim=(1,),
        #     device=self.device,
        #     inputs=[self.iter_count],
        # )
