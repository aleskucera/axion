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


class CRSolver:
    """
    A persistent Conjugate Residual solver that pre-allocates memory
    to avoid overhead during simulation steps.
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

        # Pre-allocate Dot Product helper
        self.tiled_dot = TiledDot(
            shape=(num_worlds, 1, vec_dim),
            dtype=self.scalar_dtype,
            tile_size=1024,
            device=device,
        )

    def solve(
        self,
        A: LinearOperator,
        b: wp.array,
        x: wp.array,
        iters: int,
        M: Optional[LinearOperator] = None,
    ):
        """
        Solves Ax = b using Conjugate Residual method with reused memory.
        """

        # Ensure internal buffers match the request size.
        # If your engine changes number of bodies dynamically, you might need check_resize logic here.
        if x.shape[0] != self.num_worlds or x.shape[1] != self.vec_dim:
            raise ValueError(
                f"CRSolver dimension mismatch. Expected ({self.num_worlds}, {self.vec_dim}), got {x.shape}"
            )

        # --- 1. Initialization ---

        # r = b - Ax
        # Note: matvec(x, y, z, alpha, beta) -> z = alpha*x + beta*y
        # Here: r = -1.0 * Ax + 1.0 * b
        A.matvec(x, b, self.r, alpha=-1.0, beta=1.0)

        # z := M^-1 r
        if M is None:
            wp.copy(dest=self.z, src=self.r)
        else:
            # Zero out z before M.matvec usually isn't strict requirement if beta=0,
            # but good practice if the LinearOperator assumes accumulation.
            # Assuming M.matvec overwrites if beta=0.
            M.matvec(self.r, self.z, self.z, alpha=1.0, beta=0.0)

        # Az = A * z
        A.matvec(self.z, self.Az, self.Az, alpha=1.0, beta=0.0)

        # p = z
        wp.copy(dest=self.p, src=self.z)

        # Ap = Az
        wp.copy(dest=self.Ap, src=self.Az)

        # Prepare y (M^-1 Ap) pointer or copy
        # We use Python variable `y_vec` to handle the aliasing of Ap and y
        y_vec = self.Ap
        if M is not None:
            # If we have a separate y buffer, use it
            y_vec = self.y

        # zAz_new = dot(z, Az)
        self.tiled_dot.compute(
            self.z.reshape((self.num_worlds, 1, self.vec_dim)),
            self.Az.reshape((self.num_worlds, 1, self.vec_dim)),
            self.zAz_new.reshape((self.num_worlds, 1)),
        )

        # --- 2. Fixed Iteration Loop ---
        for i in range(iters):

            # zAz_old = zAz_new
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
