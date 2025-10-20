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
        wp.launch(
            kernel=kernel_J_transpose_matvec,
            dim=self.engine.dims.N_c,
            inputs=[
                x,
                self.engine.data.J_values,
                self.engine.data.constraint_body_idx,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 2: Compute v₂ = M⁻¹ @ v₁ ---
        wp.launch(
            kernel=kernel_inv_mass_matvec,
            dim=self.engine.dims.N_b,
            inputs=[
                self._tmp_dyn_vec,
                self.engine.data.body_M_inv,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 3: Compute v₃ = J @ v₂ ---
        wp.launch(
            kernel=kernel_J_matvec,
            dim=self.engine.dims.N_c,
            inputs=[
                self._tmp_dyn_vec,
                self.engine.data.J_values,
                self.engine.data.constraint_body_idx,
            ],
            outputs=[self._tmp_con_vec],
            device=self.device,
        )

        # --- Step 4: Compute z = beta * y + alpha * (v₃ + C @ x) ---
        wp.launch(
            kernel=kernel_finalize_matvec,
            dim=self.engine.dims.N_c,
            inputs=[
                self._tmp_con_vec,  # This is J M⁻¹ Jᵀ @ x
                self.engine.data.C_values,
                x,  # original x vector
                y,  # original y vector
                alpha,  # alpha scalar
                beta,  # beta scalar
                self.regularization,
            ],
            outputs=[z],
            device=self.device,
        )
