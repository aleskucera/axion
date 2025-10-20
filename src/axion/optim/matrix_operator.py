"""
This module provides a linear operator that explicitly builds the full dense
system matrix, mirroring the behavior of a direct matrix-based solver.

The core component is the DenseSystemOperator class, which implements the
matrix-vector product by first constructing the system matrix A, where:

    A = (J M⁻¹ Jᵀ + C)

- J: The constraint Jacobian matrix.
- M: The block-diagonal mass matrix (inverse is M⁻¹).
- C: A diagonal compliance/regularization matrix.

This operator is primarily useful for debugging, validation, or for smaller
systems where the memory cost of the dense matrix is acceptable.
"""
import warp as wp
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import aslinearoperator
from warp.optim.linear import LinearOperator


@wp.func
def _compute_Aij(
    M_inv: wp.array(dtype=SpatialInertia),
    body_i: int,
    body_j: int,
    J_i: wp.spatial_vector,
    J_j: wp.spatial_vector,
):
    """Computes a block of the system matrix A = J M⁻¹ Jᵀ."""
    # This term is non-zero only if the two Jacobian rows act on the same body
    if body_i != body_j or body_i < 0 or body_j < 0:
        return 0.0

    Minv = M_inv[body_i]

    MinvJ_j = to_spatial_momentum(Minv, J_j)

    return wp.dot(J_i, MinvJ_j)


@wp.kernel
def update_system_matrix_kernel(
    M_inv: wp.array(dtype=SpatialInertia),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32, ndim=2),
):
    """Builds the full dense system matrix A = J M⁻¹ Jᵀ + C."""
    i, j = wp.tid()

    body_a_i = constraint_body_idx[i, 0]
    body_b_i = constraint_body_idx[i, 1]
    J_ia = J_values[i, 0]
    J_ib = J_values[i, 1]

    body_a_j = constraint_body_idx[j, 0]
    body_b_j = constraint_body_idx[j, 1]
    J_ja = J_values[j, 0]
    J_jb = J_values[j, 1]

    if body_a_i == body_b_i or body_a_j == body_b_j:
        A[i, j] = 0.0

    A_ij = 0.0

    # Term 1: body_a_i vs body_a_j
    if body_a_i >= 0 and body_a_i == body_a_j:
        A_ij += _compute_Aij(M_inv, body_a_i, body_a_j, J_ia, J_ja)

    # Term 2: body_a_i vs body_b_j
    if body_a_i >= 0 and body_a_i == body_b_j:
        A_ij += _compute_Aij(M_inv, body_a_i, body_b_j, J_ia, J_jb)

    # Term 3: body_b_i vs body_a_j
    if body_b_i >= 0 and body_b_i == body_a_j:
        A_ij += _compute_Aij(M_inv, body_b_i, body_a_j, J_ib, J_ja)

    # Term 4: body_b_i vs body_b_j
    if body_b_i >= 0 and body_b_i == body_b_j:
        A_ij += _compute_Aij(M_inv, body_b_i, body_b_j, J_ib, J_jb)

    # Add compliance term C_ij (only on diagonal)
    if i == j:
        A_ij += C_values[i]

    A[i, j] = A_ij


class MatrixSystemOperator(LinearOperator):
    """
    A linear operator that explicitly builds the dense system matrix A = J M⁻¹ Jᵀ + C
    and uses it for matrix-vector products.
    """

    def __init__(self, engine):
        super().__init__(
            shape=(engine.dims.con_dim, engine.dims.con_dim),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,
        )
        self.engine = engine

        # Allocate memory for the full dense matrix
        self._A = wp.zeros(self.shape, dtype=self.dtype, device=self.device)

    def update(self):
        """Re-computes the dense system matrix _A using the latest system state."""
        self._A.zero_()
        wp.launch(
            kernel=update_system_matrix_kernel,
            dim=self.shape,
            inputs=[
                self.engine.data.inv_sp_inertia,
                self.engine.data.constraint_body_idx,
                self.engine.data.J_values,
                self.engine.data.C_values,
            ],
            outputs=[self._A],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        """Computes z = beta * y + alpha * (A @ x) via dense matrix-vector product."""
        A = aslinearoperator(self._A)
        A.matvec(x=x, y=y, z=z, alpha=alpha, beta=beta)
