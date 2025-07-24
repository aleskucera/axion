import warp as wp
from axion.utils.constraints import get_constraint_body_index
from warp.optim.linear import LinearOperator


@wp.func
def compute_JM_inv_JT_i(
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_idx: int,
    J_i: wp.spatial_vector,
):
    """Computes the scalar result of Jᵢ M⁻¹ Jᵢᵀ for a single body."""
    if body_idx < 0:
        return 0.0

    # Angular part: J_ang * H_inv * J_angᵀ
    J_ang = wp.spatial_top(J_i)
    H_inv = body_inv_inertia[body_idx]
    diag_ang = wp.dot(J_ang, H_inv @ J_ang)

    # Linear part: J_lin * (m_inv * I) * J_linᵀ
    J_lin = wp.spatial_bottom(J_i)
    m_inv = body_inv_mass[body_idx]
    diag_lin = wp.dot(J_lin, J_lin) * m_inv

    return diag_ang + diag_lin


@wp.kernel
def compute_inv_diag_kernel(
    # System data from engine
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: int,
    J_n_offset: int,
    J_f_offset: int,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
    # Output array
    P_inv_diag: wp.array(dtype=wp.float32),
):
    """
    Computes the inverse of the diagonal of the system matrix A = J M⁻¹ Jᵀ + C.
    The result P_inv_diag[i] = 1.0 / A[i,i] is stored.
    """
    constraint_idx = wp.tid()

    body_a, body_b = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        constraint_idx,
    )

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    # Compute diagonal term from J M⁻¹ Jᵀ
    diag_JMJ = compute_JM_inv_JT_i(body_inv_mass, body_inv_inertia, body_a, J_ia)
    diag_JMJ += compute_JM_inv_JT_i(body_inv_mass, body_inv_inertia, body_b, J_ib)

    # Add diagonal compliance term C[i,i]
    diag_A = diag_JMJ + C_values[constraint_idx]

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
            shape=(engine.con_dim, engine.con_dim),
            dtype=engine._lambda.dtype,
            device=engine.device,
            matvec=None,  # Will be set later
        )
        self.engine = engine

        # Storage for the inverse diagonal elements
        self._P_inv_diag = wp.zeros(
            engine.con_dim, dtype=wp.float32, device=self.device
        )

    def update(self):
        """
        Re-computes the preconditioner's data. This must be called each time
        the Jacobian (J) or compliance (C) values change.
        """
        wp.launch(
            kernel=compute_inv_diag_kernel,
            dim=self.engine.con_dim,
            inputs=[
                self.engine.body_inv_mass,
                self.engine.body_inv_inertia,
                self.engine.joint_parent,
                self.engine.joint_child,
                self.engine._contact_body_a,
                self.engine._contact_body_b,
                self.engine.J_j_offset,
                self.engine.J_n_offset,
                self.engine.J_f_offset,
                self.engine._J_values,
                self.engine._C_values,
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
            dim=self.engine.con_dim,
            inputs=[self._P_inv_diag, x, y, alpha, beta, z],
            device=self.device,
        )
