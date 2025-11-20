"""
Jacobi preconditioner for the system matrix A = J M⁻¹ Jᵀ + C.
"""
import warp as wp
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


@wp.kernel
def compute_inv_diag_kernel(
    body_M_inv: wp.array(dtype=SpatialInertia),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Output array
    P_inv_diag: wp.array(dtype=wp.float32),
):
    """
    Computes the inverse of the diagonal of the system matrix A = J M⁻¹ Jᵀ + C.
    The result P_inv_diag[i] = 1.0 / A[i,i] is stored.
    """
    constraint_idx = wp.tid()

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    result = 0.0
    if body_1 >= 0:
        M_inv_1 = body_M_inv[body_1]
        J_1 = J_values[constraint_idx, 0]
        result += wp.dot(J_1, to_spatial_momentum(M_inv_1, J_1))
    if body_2 >= 0:
        M_inv_2 = body_M_inv[body_2]
        J_2 = J_values[constraint_idx, 1]
        result += wp.dot(J_2, to_spatial_momentum(M_inv_2, J_2))

    # Add diagonal compliance term C[i,i]
    diag_A = result + C_values[constraint_idx]

    # Compute and store inverse, with stabilization
    P_inv_diag[constraint_idx] = 1.0 / (diag_A + 1e-6)


@wp.kernel
def apply_preconditioner_kernel(
    P_inv_diag: wp.array(dtype=wp.float32),
    vec_x: wp.array(dtype=wp.float32),
    vec_y: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    out_vec_z: wp.array(dtype=wp.float32),
):
    """Applies the Jacobi preconditioner: z = beta*y + alpha * P⁻¹ * x"""
    i = wp.tid()
    preconditioned_x = P_inv_diag[i] * vec_x[i]

    if beta == 0.0:
        out_vec_z[i] = alpha * preconditioned_x
    else:
        out_vec_z[i] = beta * vec_y[i] + alpha * preconditioned_x


class JacobiPreconditioner(LinearOperator):
    """
    A Jacobi (diagonal) preconditioner for the system matrix A = J M⁻¹ Jᵀ + C.

    This class provides a .matvec() method that applies the inverse of the
    diagonal of A, for use with Warp's iterative solvers.
    """

    def __init__(self, engine):
        super().__init__(
            shape=(engine.dims.N_c, engine.dims.N_c),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,  # Will be set later
        )
        self.engine = engine

        # Storage for the inverse diagonal elements
        self._P_inv_diag = wp.zeros(engine.dims.N_c, dtype=wp.float32, device=self.device)

    def update(self):
        """
        Re-computes the preconditioner's data. This must be called each time
        the Jacobian (J) or compliance (C) values change.
        """
        wp.launch(
            kernel=compute_inv_diag_kernel,
            dim=self.engine.dims.N_c,
            inputs=[
                self.engine.data.world_M_inv,
                self.engine.data.J_values.full,
                self.engine.data.C_values.full,
                self.engine.data.constraint_body_idx.full,
            ],
            outputs=[self._P_inv_diag],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        """
        Performs the preconditioning operation z = beta*y + alpha*(M⁻¹@x),
        where M⁻¹ is the inverse diagonal matrix stored in `_P_inv_diag`.
        """
        wp.launch(
            kernel=apply_preconditioner_kernel,
            dim=self.engine.dims.N_c,
            inputs=[self._P_inv_diag, x, y, alpha, beta, z],
            device=self.device,
        )
