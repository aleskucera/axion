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
from axion.utils.constraints import get_constraint_body_index
from warp.optim.linear import aslinearoperator
from warp.optim.linear import LinearOperator


@wp.func
def _compute_Aij(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    body_i: int,
    body_j: int,
    J_i: wp.spatial_vector,
    J_j: wp.spatial_vector,
):
    """Computes a block of the system matrix A = J M⁻¹ Jᵀ."""
    # This term is non-zero only if the two Jacobian rows act on the same body
    if body_i != body_j or body_i < 0 or body_j < 0:
        return 0.0

    J_i_ang = wp.spatial_top(J_i)
    J_i_lin = wp.spatial_bottom(J_i)
    J_j_ang = wp.spatial_top(J_j)
    J_j_lin = wp.spatial_bottom(J_j)

    # Compute the angular part
    A_ij_ang = wp.dot(J_i_ang, body_inertia_inv[body_i] @ J_j_ang)
    # Compute the linear part
    A_ij_lin = wp.dot(J_i_lin, body_mass_inv[body_i] * J_j_lin)

    return A_ij_ang + A_ij_lin


@wp.kernel
def update_system_matrix_kernel(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: wp.int32,
    J_n_offset: wp.int32,
    J_f_offset: wp.int32,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32, ndim=2),
):
    """Builds the full dense system matrix A = J M⁻¹ Jᵀ + C."""
    i, j = wp.tid()

    body_a_i, body_b_i = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        i,
    )
    J_ia = J_values[i, 0]
    J_ib = J_values[i, 1]

    body_a_j, body_b_j = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        j,
    )
    J_ja = J_values[j, 0]
    J_jb = J_values[j, 1]

    if body_a_i == body_b_i or body_a_j == body_b_j:
        A[i, j] = 0.0

    A_ij = 0.0

    # Term 1: body_a_i vs body_a_j
    if body_a_i >= 0 and body_a_i == body_a_j:
        A_ij += _compute_Aij(
            body_mass_inv, body_inertia_inv, body_a_i, body_a_j, J_ia, J_ja
        )

    # Term 2: body_a_i vs body_b_j
    if body_a_i >= 0 and body_a_i == body_b_j:
        A_ij += _compute_Aij(
            body_mass_inv, body_inertia_inv, body_a_i, body_b_j, J_ia, J_jb
        )

    # Term 3: body_b_i vs body_a_j
    if body_b_i >= 0 and body_b_i == body_a_j:
        A_ij += _compute_Aij(
            body_mass_inv, body_inertia_inv, body_b_i, body_a_j, J_ib, J_ja
        )

    # Term 4: body_b_i vs body_b_j
    if body_b_i >= 0 and body_b_i == body_b_j:
        A_ij += _compute_Aij(
            body_mass_inv, body_inertia_inv, body_b_i, body_b_j, J_ib, J_jb
        )

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
            dtype=engine._lambda.dtype,
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
                self.engine.body_inv_mass,
                self.engine.body_inv_inertia,
                self.engine.joint_parent,
                self.engine.joint_child,
                self.engine._contact_body_a,
                self.engine._contact_body_b,
                self.engine.dims.J_j_offset,
                self.engine.dims.J_n_offset,
                self.engine.dims.J_f_offset,
                self.engine._J_values,
                self.engine._C_values,
            ],
            outputs=[self._A],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        """Computes z = beta * y + alpha * (A @ x) via dense matrix-vector product."""
        A = aslinearoperator(self._A)
        A.matvec(x=x, y=y, z=z, alpha=alpha, beta=beta)
