from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

import warp as wp
from warp.optim.linear import LinearOperator
from warp.utils import array_inner

if TYPE_CHECKING:
    from axion import HDF5Logger
# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})


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


def cr_solver(
    A: LinearOperator,
    b: wp.array,
    x: wp.array,
    iters: int,
    preconditioner: Optional[LinearOperator] = None,
    logger: Optional[HDF5Logger] = None,
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
    """

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
    for i in range(iters):
        # The value from the previous iteration becomes the "old" value for this one.
        wp.copy(dest=zAz_old, src=zAz_new)

        if preconditioner is not None:
            preconditioner.matvec(Ap, y, y, alpha=1.0, beta=0.0)
        else:
            wp.copy(dest=y, src=Ap)

        # dot(y, Ap) is the denominator for alpha
        array_inner(y, Ap, out=y_Ap)

        # Always use the robust CR update kernel
        wp.launch(
            kernel=_cr_kernel_1_fixed,
            dim=vec_dim,
            device=device,
            inputs=[zAz_old, y_Ap, x, r, z, p, Ap, y],
        )

        # Az = A * z
        A.matvec(z, Az, Az, alpha=1.0, beta=0.0)
        # zAz_new = dot(z, Az)
        array_inner(z, Az, out=zAz_new)

        # Update p, Ap (this part is the same for both)
        wp.launch(
            kernel=_cr_kernel_2_fixed,
            dim=vec_dim,
            device=device,
            inputs=[zAz_old, zAz_new, z, p, Az, Ap],
        )

        # if logger:
        #     with logger.scope(f"linear_iter_{i:02d}"):
        #         logger.log_dataset("x", x.numpy())
        #         logger.log_dataset("residual", r.numpy())
