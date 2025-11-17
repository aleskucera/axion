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
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


@wp.kernel
def kernel_J_transpose_matvec(
    in_vec: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Output array
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the matrix-vector product: out_vec = Jᵀ @ vec_x.

    This kernel iterates over each constraint (the dimension of vec_x) and
    scatters the results into the dynamics-space vector (out_vec) using atomic adds.

    Args:
        vec_x: A vector in constraint space (e.g., delta_lambda).
        out_vec: A vector in dynamics space (size num_bodies * 6) to store the result.
                 This vector MUST be zero-initialized before calling this kernel.
    """
    constraint_idx = wp.tid()

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]
    x_i = in_vec[constraint_idx]

    if body_1 >= 0:
        out_vec[body_1] += x_i * J_1
    if body_2 >= 0:
        out_vec[body_2] += x_i * J_2


@wp.kernel
def kernel_inv_mass_matvec(
    in_vec: wp.array(dtype=wp.spatial_vector),
    body_M_inv: wp.array(dtype=SpatialInertia),
    out_vec: wp.array(dtype=wp.spatial_vector),
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

    M_inv = body_M_inv[body_idx]
    out_vec[body_idx] = to_spatial_momentum(M_inv, in_vec[body_idx])


@wp.kernel
def kernel_invM_Jt_matvec(
    # Inputs
    x: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    body_M_inv: wp.array(dtype=SpatialInertia),
    # Output array
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the fused operation: out_vec = M⁻¹ @ Jᵀ @ x.

    This is equivalent to merging kernel_J_transpose_matvec and kernel_inv_mass_matvec.
    It works by applying the M⁻¹ multiplication to each constraint's contribution
    *before* atomically adding it to the final output vector.

    Args:
        x: A vector in constraint space (e.g., lambda impulses).
        out_vec: The resulting vector in dynamics space (velocity change).
                 This vector MUST be zero-initialized before calling.
    """
    constraint_idx = wp.tid()

    body_1_idx = constraint_body_idx[constraint_idx, 0]
    body_2_idx = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]
    x_i = x[constraint_idx]

    if body_1_idx >= 0:
        # Calculate the momentum contribution from this constraint
        jt_x_1 = x_i * J_1
        # Convert momentum to velocity change (v = M⁻¹ * p)
        # and atomically add it to the output velocity vector
        m_inv_1 = body_M_inv[body_1_idx]
        delta_v_1 = to_spatial_momentum(m_inv_1, jt_x_1)
        wp.atomic_add(out_vec, body_1_idx, delta_v_1)

    if body_2_idx >= 0:
        # Repeat for the second body
        jt_x_2 = x_i * J_2
        m_inv_2 = body_M_inv[body_2_idx]
        delta_v_2 = to_spatial_momentum(m_inv_2, jt_x_2)
        wp.atomic_add(out_vec, body_2_idx, delta_v_2)


@wp.kernel
def kernel_J_matvec(
    in_vec: wp.array(dtype=wp.spatial_vector),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
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

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]

    result = 0.0
    if body_1 >= 0:
        result += wp.dot(J_1, in_vec[body_1])
    if body_2 >= 0:
        result += wp.dot(J_2, in_vec[body_2])

    out_vec[constraint_idx] = result


@wp.kernel
def kernel_finalize_matvec(
    J_M_inv_Jt_x: wp.array(dtype=wp.float32),
    C_values: wp.array(dtype=wp.float32),
    vec_x: wp.array(dtype=wp.float32),
    vec_y: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    regularization: float,
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
    # TODO: Fix the regularization
    c_times_x = (C_values[i] + regularization) * vec_x[i]
    a_times_x = J_M_inv_Jt_x[i] + c_times_x

    # The crucial change is here: including beta * y
    if beta == 0.0:
        out_vec_z[i] = alpha * a_times_x
    else:
        out_vec_z[i] = beta * vec_y[i] + alpha * a_times_x


@wp.kernel
def kernel_J_matvec_and_finalize(
    # Inputs for J @ v
    dyn_vec: wp.array(dtype=wp.spatial_vector),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Inputs for finalize
    C_values: wp.array(dtype=wp.float32),
    vec_x: wp.array(dtype=wp.float32),
    vec_y: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    regularization: float,
    # Output array
    out_vec_z: wp.array(dtype=wp.float32),
):
    """
    Computes J @ dyn_vec and immediately uses it in the finalization step.
    This fused kernel calculates:
    z = beta * y + alpha * ( (J @ dyn_vec) + (C + reg) @ x )
    """
    constraint_idx = wp.tid()

    # --- Part 1: Compute J @ dyn_vec for this constraint (from kernel_J_matvec) ---
    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]

    j_m_inv_jt_x = 0.0
    if body_1 >= 0:
        j_m_inv_jt_x += wp.dot(J_1, dyn_vec[body_1])
    if body_2 >= 0:
        j_m_inv_jt_x += wp.dot(J_2, dyn_vec[body_2])

    # --- Part 2: Use the result immediately (from kernel_finalize_matvec) ---
    c_times_x = (C_values[constraint_idx] + regularization) * vec_x[constraint_idx]
    a_times_x = j_m_inv_jt_x + c_times_x

    if beta == 0.0:
        out_vec_z[constraint_idx] = alpha * a_times_x
    else:
        out_vec_z[constraint_idx] = beta * vec_y[constraint_idx] + alpha * a_times_x


class MatrixFreeSystemOperator(LinearOperator):
    """
    A matrix-free linear operator for the system A = J M⁻¹ Jᵀ + C.

    This class provides a .matvec() method that computes the matrix-vector
    product A @ x without explicitly constructing the matrix A. It is intended
    to be used with iterative linear solvers like `cr_solver`.
    """

    def __init__(self, engine, regularization: float = 1e-5):
        """
        Initializes the operator with data from the main physics engine.

        Args:
            engine: An instance of the main physics engine (e.g., NSNEngine)
                    that holds all the necessary system data (Jacobians,
                    masses, constraint info, etc.).
        """
        super().__init__(
            shape=(engine.dims.N_c, engine.dims.N_c),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,
        ),
        self.engine = engine
        self.regularization = regularization

        # Pre-allocate temporary buffers for intermediate calculations.
        self._tmp_dyn_vec = wp.zeros(engine.dims.N_b, dtype=wp.spatial_vector, device=engine.device)
        self._tmp_con_vec = wp.zeros(engine.dims.N_c, dtype=wp.float32, device=engine.device)

    def matvec(self, x, y, z, alpha, beta):
        """
        Computes the matrix-vector product: z = beta * y + alpha * (A @ x).
        """
        # --- Step 1: Compute v₁ = Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        # wp.launch(
        #     kernel=kernel_J_transpose_matvec,
        #     dim=self.engine.dims.N_c,
        #     inputs=[
        #         x,
        #         self.engine.data.J_values,
        #         self.engine.data.constraint_body_idx,
        #     ],
        #     outputs=[self._tmp_dyn_vec],
        #     device=self.device,
        # )
        #
        # # --- Step 2: Compute v₂ = M⁻¹ @ v₁ ---
        # wp.launch(
        #     kernel=kernel_inv_mass_matvec,
        #     dim=self.engine.dims.N_b,
        #     inputs=[
        #         self._tmp_dyn_vec,
        #         self.engine.data.world_M_inv,
        #     ],
        #     outputs=[self._tmp_dyn_vec],
        #     device=self.device,
        # )
        #
        # # --- Step 3: Compute v₃ = J @ v₂ ---
        # wp.launch(
        #     kernel=kernel_J_matvec,
        #     dim=self.engine.dims.N_c,
        #     inputs=[
        #         self._tmp_dyn_vec,
        #         self.engine.data.J_values,
        #         self.engine.data.constraint_body_idx,
        #     ],
        #     outputs=[self._tmp_con_vec],
        #     device=self.device,
        # )
        #
        # # --- Step 4: Compute z = beta * y + alpha * (v₃ + C @ x) ---
        # wp.launch(
        #     kernel=kernel_finalize_matvec,
        #     dim=self.engine.dims.N_c,
        #     inputs=[
        #         self._tmp_con_vec,  # This is J M⁻¹ Jᵀ @ x
        #         self.engine.data.C_values,
        #         x,  # original x vector
        #         y,  # original y vector
        #         alpha,  # alpha scalar
        #         beta,  # beta scalar
        #         self.regularization,
        #     ],
        #     outputs=[z],
        #     device=self.device,
        # )

        # --- Step 1 & 2 (Fused): Compute v₂ = M⁻¹ @ Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        wp.launch(
            kernel=kernel_invM_Jt_matvec,  # The new fused kernel
            dim=self.engine.dims.N_c,
            inputs=[
                x,
                self.engine.data.J_values,
                self.engine.data.constraint_body_idx,
                self.engine.data.world_M_inv,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 3 & 4 (Fused): Compute z = final_op(J @ v₂) ---
        wp.launch(
            kernel=kernel_J_matvec_and_finalize,  # Fused kernel from previous answer
            dim=self.engine.dims.N_c,
            inputs=[
                self._tmp_dyn_vec,  # This is v₂ = M⁻¹Jᵀx
                self.engine.data.J_values,
                self.engine.data.constraint_body_idx,
                self.engine.data.C_values,
                x,
                y,
                alpha,
                beta,
                self.regularization,
            ],
            outputs=[z],
            device=self.device,
        )
