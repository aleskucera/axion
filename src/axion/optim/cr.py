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
    """Graph-compatible CR kernel part 1: Updates x, r, and z."""
    world_idx, i = wp.tid()

    # Unconditionally compute alpha, with a safe division
    alpha = zAz_old.dtype(0.0)
    if y_Ap[world_idx] > 0.0:
        alpha = zAz_old[world_idx] / y_Ap[world_idx]

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
    """Graph-compatible CR kernel part 2: Updates p and Ap."""
    world_idx, i = wp.tid()

    # Unconditionally compute beta, with a safe division
    beta = zAz_old.dtype(0.0)
    if zAz_old[world_idx] > 0.0:
        beta = zAz_new[world_idx] / zAz_old[world_idx]

    p[world_idx, i] = z[world_idx, i] + beta * p[world_idx, i]
    Ap[world_idx, i] = Az[world_idx, i] + beta * Ap[world_idx, i]


def cr(
    A: LinearOperator,
    b: wp.array,
    x: wp.array,
    iters: int,
    M: Optional[LinearOperator] = None,
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
    num_worlds = x.shape[0]
    vec_dim = x.shape[1]

    tiled_dot = TiledDot(
        shape=(num_worlds, 1, vec_dim),
        dtype=scalar_dtype,
        tile_size=1024,
        device=device,
    )

    # ============== 1. Initialization ==========================================
    # This part runs once before the main loop.

    # Residual r = b - Ax
    r = wp.empty_like(b)
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    # Notations below follow the Conjugate Residual method pseudo-code.
    # z := M^-1 r
    # y := M^-1 Ap
    if M is None:
        z = wp.clone(r)
    else:
        z = wp.zeros_like(r)
        M.matvec(r, z, z, alpha=1.0, beta=0.0)

    Az = wp.zeros_like(b)
    A.matvec(z, Az, Az, alpha=1.0, beta=0.0)

    p = wp.clone(z)
    Ap = wp.clone(Az)

    if M is None:
        y = Ap
    else:
        y = wp.zeros_like(Ap)

    # Scalar values stored in single-element arrays
    zAz_old = wp.empty(shape=(num_worlds), dtype=scalar_dtype, device=device)
    zAz_new = wp.empty(shape=(num_worlds), dtype=scalar_dtype, device=device)
    y_Ap = wp.empty(shape=(num_worlds), dtype=scalar_dtype, device=device)

    # Initial dot product: zAz = dot(z, Az)
    tiled_dot.compute(
        z.reshape((num_worlds, 1, vec_dim)),
        Az.reshape((num_worlds, 1, vec_dim)),
        zAz_new.reshape((num_worlds, 1)),
    )
    # array_inner(z, Az, out=zAz_new)

    # ============== 2. Fixed Iteration Loop ====================================
    for i in range(iters):
        # The value from the previous iteration becomes the "old" value for this one.
        wp.copy(dest=zAz_old, src=zAz_new)

        if M is not None:
            M.matvec(Ap, y, y, alpha=1.0, beta=0.0)
        else:
            wp.copy(dest=y, src=Ap)

        # dot(y, Ap) is the denominator for alpha
        tiled_dot.compute(
            y.reshape((num_worlds, 1, vec_dim)),
            Ap.reshape((num_worlds, 1, vec_dim)),
            y_Ap.reshape((num_worlds, 1)),
        )
        # array_inner(y, Ap, out=y_Ap)

        # Always use the robust CR update kernel
        wp.launch(
            kernel=_cr_kernel_1,
            dim=(num_worlds, vec_dim),
            device=device,
            inputs=[zAz_old, y_Ap, x, r, z, p, Ap, y],
        )

        # Az = A * z
        A.matvec(z, Az, Az, alpha=1.0, beta=0.0)
        # zAz_new = dot(z, Az)
        tiled_dot.compute(
            z.reshape((num_worlds, 1, vec_dim)),
            Az.reshape((num_worlds, 1, vec_dim)),
            zAz_new.reshape((num_worlds, 1)),
        )
        # array_inner(z, Az, out=zAz_new)

        # Update p, Ap (this part is the same for both)
        wp.launch(
            kernel=_cr_kernel_2,
            dim=(num_worlds, vec_dim),
            device=device,
            inputs=[zAz_old, zAz_new, z, p, Az, Ap],
        )
