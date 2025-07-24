"""
This module provides a matrix-free linear operator for solving the mixed-integer
system of equations that arises in velocity-based physics simulation.

The core component is the SystemOperator class, which implements the matrix-vector
product for the system matrix A, where:

    A = (J M⁻¹ Jᵀ + C)

- J: The constraint Jacobian matrix.
- M: The block-diagonal mass matrix (inverse is M⁻¹).
- C: A diagonal compliance/regularization matrix.

This operator is designed to be used with iterative linear solvers like Conjugate
Residual (CR) or Conjugate Gradient (CG), allowing the system to be solved
without ever forming the potentially very large and dense matrix A explicitly.
"""
import warp as wp
from axion.utils.constraints import get_constraint_body_index
from warp.optim.linear import LinearOperator


# @wp.kernel
# def kernel_J_transpose_matvec(
#     # Constraint layout information
#     joint_parent: wp.array(dtype=wp.int32),
#     joint_child: wp.array(dtype=wp.int32),
#     contact_body_a: wp.array(dtype=wp.int32),
#     contact_body_b: wp.array(dtype=wp.int32),
#     J_j_offset: int,
#     J_n_offset: int,
#     J_f_offset: int,
#     # Jacobian and vector data
#     J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
#     vec_x: wp.array(dtype=wp.float32),
#     # Output array
#     out_vec: wp.array(dtype=wp.float32),
# ):
#     """
#     Computes the matrix-vector product: out_vec = Jᵀ @ vec_x.
#
#     This kernel iterates over each constraint (the dimension of vec_x) and
#     scatters the results into the dynamics-space vector (out_vec) using atomic adds.
#
#     Args:
#         vec_x: A vector in constraint space (e.g., delta_lambda).
#         out_vec: A vector in dynamics space (size num_bodies * 6) to store the result.
#                  This vector MUST be zero-initialized before calling this kernel.
#     """
#     constraint_idx = wp.tid()
#
#     body_a, body_b = get_constraint_body_index(
#         joint_parent,
#         joint_child,
#         contact_body_a,
#         contact_body_b,
#         J_j_offset,
#         J_n_offset,
#         J_f_offset,
#         constraint_idx,
#     )
#     J_ia = J_values[constraint_idx, 0]
#     J_ib = J_values[constraint_idx, 1]
#     x_i = vec_x[constraint_idx]
#
#     # Scatter the product Jᵢᵀ * xᵢ into the appropriate body locations
#     if body_a >= 0:
#         for i in range(6):
#             wp.atomic_add(out_vec, body_a * 6 + i, J_ia[i] * x_i)
#     if body_b >= 0:
#         for i in range(6):
#             wp.atomic_add(out_vec, body_b * 6 + i, J_ib[i] * x_i)


@wp.kernel
def kernel_J_transpose_matvec(
    # Constraint layout information
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: int,
    J_n_offset: int,
    J_f_offset: int,
    # Jacobian and vector data
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    vec_x: wp.array(dtype=wp.float32),
    # Output array
    out_vec: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    # Get body indices and data
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
    x_i = vec_x[constraint_idx]

    # Compute branchless multipliers
    multiplier_a = wp.where(body_a >= 0, 1.0, 0.0)
    multiplier_b = wp.where(body_b >= 0, 1.0, 0.0)

    # Scatter contributions without branching
    for i in range(wp.static(6)):
        st_i = wp.static(i)

        # For body_a
        index_a = wp.where(body_a >= 0, body_a * 6 + st_i, st_i)
        wp.atomic_add(out_vec, index_a, J_ia[st_i] * x_i * multiplier_a)

        # For body_b
        index_b = wp.where(body_b >= 0, body_b * 6 + st_i, st_i)
        wp.atomic_add(out_vec, index_b, J_ib[st_i] * x_i * multiplier_b)


@wp.kernel
def kernel_inv_mass_matvec(
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    in_vec: wp.array(dtype=wp.float32),
    out_vec: wp.array(dtype=wp.float32),
):
    """
    Computes the matrix-vector product: out_vec = M⁻¹ @ in_vec.

    M⁻¹ is the block-diagonal inverse mass matrix, composed of a 3x3 inverse
    inertia tensor and a scalar inverse mass for each body.

    Args:
        in_vec: A vector in dynamics space (e.g., the result of Jᵀ @ x).
        out_vec: The resulting vector in dynamics space.
    """
    body_idx = wp.tid()

    # Angular part (top 3 components)
    w = wp.vec3(
        in_vec[body_idx * 6 + 0], in_vec[body_idx * 6 + 1], in_vec[body_idx * 6 + 2]
    )
    H_inv = body_inv_inertia[body_idx]
    w_out = H_inv @ w
    out_vec[body_idx * 6 + 0] = w_out[0]
    out_vec[body_idx * 6 + 1] = w_out[1]
    out_vec[body_idx * 6 + 2] = w_out[2]

    # Linear part (bottom 3 components)
    v = wp.vec3(
        in_vec[body_idx * 6 + 3], in_vec[body_idx * 6 + 4], in_vec[body_idx * 6 + 5]
    )
    m_inv = body_inv_mass[body_idx]
    v_out = m_inv * v
    out_vec[body_idx * 6 + 3] = v_out[0]
    out_vec[body_idx * 6 + 4] = v_out[1]
    out_vec[body_idx * 6 + 5] = v_out[2]


# @wp.kernel
# def kernel_J_matvec(
#     # Constraint layout information
#     joint_parent: wp.array(dtype=wp.int32),
#     joint_child: wp.array(dtype=wp.int32),
#     contact_body_a: wp.array(dtype=wp.int32),
#     contact_body_b: wp.array(dtype=wp.int32),
#     J_j_offset: int,
#     J_n_offset: int,
#     J_f_offset: int,
#     # Jacobian and vector data
#     J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
#     in_vec: wp.array(dtype=wp.float32),
#     # Output array
#     out_vec: wp.array(dtype=wp.float32),
# ):
#     """
#     Computes the matrix-vector product: out_vec = J @ in_vec.
#
#     This kernel iterates over each constraint and gathers values from the
#     dynamics-space vector (in_vec) to produce the constraint-space vector (out_vec).
#
#     Args:
#         in_vec: A vector in dynamics space (e.g., M⁻¹ @ Jᵀ @ x).
#         out_vec: The resulting vector in constraint space.
#     """
#     constraint_idx = wp.tid()
#
#     body_a, body_b = get_constraint_body_index(
#         joint_parent,
#         joint_child,
#         contact_body_a,
#         contact_body_b,
#         J_j_offset,
#         J_n_offset,
#         J_f_offset,
#         constraint_idx,
#     )
#
#     J_ia = J_values[constraint_idx, 0]
#     J_ib = J_values[constraint_idx, 1]
#
#     result = 0.0
#
#     # Gather from body_a
#     if body_a >= 0:
#         vel_a = wp.spatial_vector(
#             wp.vec3(
#                 in_vec[body_a * 6 + 0], in_vec[body_a * 6 + 1], in_vec[body_a * 6 + 2]
#             ),
#             wp.vec3(
#                 in_vec[body_a * 6 + 3], in_vec[body_a * 6 + 4], in_vec[body_a * 6 + 5]
#             ),
#         )
#         result += wp.dot(J_ia, vel_a)
#
#     # Gather from body_b
#     if body_b >= 0:
#         vel_b = wp.spatial_vector(
#             wp.vec3(
#                 in_vec[body_b * 6 + 0], in_vec[body_b * 6 + 1], in_vec[body_b * 6 + 2]
#             ),
#             wp.vec3(
#                 in_vec[body_b * 6 + 3], in_vec[body_b * 6 + 4], in_vec[body_b * 6 + 5]
#             ),
#         )
#         result += wp.dot(J_ib, vel_b)
#
#     out_vec[constraint_idx] = result


@wp.kernel
def kernel_J_matvec(
    # Constraint layout information
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: int,
    J_n_offset: int,
    J_f_offset: int,
    # Jacobian and vector data
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    in_vec: wp.array(dtype=wp.float32),
    # Output array
    out_vec: wp.array(dtype=wp.float32),
):
    """
    Computes the matrix-vector product: out_vec = J @ in_vec.

    This kernel iterates over each constraint and gathers values from the
    dynamics-space vector (in_vec) to produce the constraint-space vector (out_vec).

    Args:
        in_vec: A vector in dynamics space (e.g., M⁻¹ @ Jᵀ @ x).
        out_vec: The resulting vector in constraint space.
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

    # Compute masks and base indices for body_a
    mask_a = wp.where(body_a >= 0, 1.0, 0.0)
    base_a = wp.where(body_a >= 0, body_a * 6, 0)

    # Construct vel_a with masking
    vel_a_linear = wp.vec3(
        in_vec[base_a + 0] * mask_a,
        in_vec[base_a + 1] * mask_a,
        in_vec[base_a + 2] * mask_a,
    )
    vel_a_angular = wp.vec3(
        in_vec[base_a + 3] * mask_a,
        in_vec[base_a + 4] * mask_a,
        in_vec[base_a + 5] * mask_a,
    )
    vel_a = wp.spatial_vector(vel_a_linear, vel_a_angular)

    # Compute masks and base indices for body_b
    mask_b = wp.where(body_b >= 0, 1.0, 0.0)
    base_b = wp.where(body_b >= 0, body_b * 6, 0)

    # Construct vel_b with masking
    vel_b_linear = wp.vec3(
        in_vec[base_b + 0] * mask_b,
        in_vec[base_b + 1] * mask_b,
        in_vec[base_b + 2] * mask_b,
    )
    vel_b_angular = wp.vec3(
        in_vec[base_b + 3] * mask_b,
        in_vec[base_b + 4] * mask_b,
        in_vec[base_b + 5] * mask_b,
    )
    vel_b = wp.spatial_vector(vel_b_linear, vel_b_angular)

    # Compute the result
    result = wp.dot(J_ia, vel_a) + wp.dot(J_ib, vel_b)

    out_vec[constraint_idx] = result


@wp.kernel
def kernel_finalize_matvec(
    J_M_inv_Jt_x: wp.array(dtype=wp.float32),
    C_values: wp.array(dtype=wp.float32),
    vec_x: wp.array(dtype=wp.float32),
    vec_y: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    out_vec_z: wp.array(dtype=wp.float32),
):
    """
    Performs the final step of the matvec computation:
    z = beta * y + alpha * ( (J M⁻¹ Jᵀ @ x) + (C @ x) )

    Args:
        J_M_inv_Jt_x: The result of J @ M⁻¹ @ Jᵀ @ x.
        C_values: The diagonal entries of the compliance matrix C.
        vec_x: The original input vector 'x' to the matvec operation.
        vec_y: The input vector 'y'.
        alpha: Scalar multiplier for the A@x term.
        beta: Scalar multiplier for the y term.
        out_vec_z: The final output vector z.
    """
    i = wp.tid()
    c_times_x = C_values[i] * vec_x[i]
    a_times_x = J_M_inv_Jt_x[i] + c_times_x

    # The crucial change is here: including beta * y
    if beta == 0.0:
        out_vec_z[i] = alpha * a_times_x
    else:
        out_vec_z[i] = beta * vec_y[i] + alpha * a_times_x


class SystemOperator(LinearOperator):
    """
    A matrix-free linear operator for the system A = J M⁻¹ Jᵀ + C.

    This class provides a .matvec() method that computes the matrix-vector
    product A @ x without explicitly constructing the matrix A. It is intended
    to be used with iterative linear solvers like `cr_solver`.
    """

    def __init__(self, engine):
        """
        Initializes the operator with data from the main physics engine.

        Args:
            engine: An instance of the main physics engine (e.g., NSNEngine)
                    that holds all the necessary system data (Jacobians,
                    masses, constraint info, etc.).
        """
        super().__init__(
            shape=(engine.con_dim, engine.con_dim),
            dtype=engine._lambda.dtype,
            device=engine.device,
            matvec=None,  # Will be set later
        )
        self.engine = engine

        # Pre-allocate temporary buffers for intermediate calculations.
        self._tmp_dyn_vec = wp.zeros(
            engine.dyn_dim, dtype=wp.float32, device=engine.device
        )
        self._tmp_con_vec = wp.zeros(
            engine.con_dim, dtype=wp.float32, device=engine.device
        )

    def matvec(self, x, y, z, alpha, beta):
        """
        Computes the matrix-vector product: z = beta * y + alpha * (A @ x).
        """
        # --- Step 1: Compute v₁ = Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        wp.launch(
            kernel=kernel_J_transpose_matvec,
            dim=self.engine.con_dim,
            inputs=[
                self.engine.joint_parent,
                self.engine.joint_child,
                self.engine._contact_body_a,
                self.engine._contact_body_b,
                self.engine.J_j_offset,
                self.engine.J_n_offset,
                self.engine.J_f_offset,
                self.engine._J_values,
                x,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 2: Compute v₂ = M⁻¹ @ v₁ ---
        wp.launch(
            kernel=kernel_inv_mass_matvec,
            dim=self.engine.N_b,
            inputs=[
                self.engine.body_inv_mass,
                self.engine.body_inv_inertia,
                self._tmp_dyn_vec,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 3: Compute v₃ = J @ v₂ ---
        wp.launch(
            kernel=kernel_J_matvec,
            dim=self.engine.con_dim,
            inputs=[
                self.engine.joint_parent,
                self.engine.joint_child,
                self.engine._contact_body_a,
                self.engine._contact_body_b,
                self.engine.J_j_offset,
                self.engine.J_n_offset,
                self.engine.J_f_offset,
                self.engine._J_values,
                self._tmp_dyn_vec,
            ],
            outputs=[self._tmp_con_vec],
            device=self.device,
        )

        # --- Step 4: Compute z = beta * y + alpha * (v₃ + C @ x) ---
        wp.launch(
            kernel=kernel_finalize_matvec,
            dim=self.engine.con_dim,
            inputs=[
                self._tmp_con_vec,  # This is J M⁻¹ Jᵀ @ x
                self.engine._C_values,
                x,  # original x vector
                y,  # original y vector
                alpha,  # alpha scalar
                beta,  # beta scalar
                z,  # The final output vector
            ],
            device=self.device,
        )
