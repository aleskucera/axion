"""
This module provides a matrix-free linear operator for the Axion physics engine.

The core of many modern physics simulators involves solving a large linear system of equations
at each step of a Newton iteration. This system can be represented as `Ax = b`, where the
matrix `A` is the system matrix, often a Schur complement of the form:

\[
    A = J M^{-1} J^T + C
\]

Where:

- `J` is the constraint Jacobian, which maps generalized body velocities to relative velocities at the constraints.
- `M` is the block-diagonal generalized mass matrix.
- `C` is a diagonal matrix representing compliance and regularization terms.

For complex scenes, the matrix `A` can become very large and dense, making its explicit
construction and storage computationally expensive and memory-intensive.

Matrix-free methods avoid this problem by never forming `A`. Instead, they provide a function
that directly computes the matrix-vector product `A @ x`. This is sufficient for iterative linear
solvers (like Conjugate Gradient or Conjugate Residual), which only need this product to find a solution.

This module implements `MatrixFreeSystemOperator`, which computes `A @ x` through a sequence
of kernel calls that apply `J^T`, `M^{-1}`, and `J` in succession.
"""
import warp as wp
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


@wp.kernel
def kernel_J_transpose_matvec(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    vec_x: wp.array(dtype=wp.float32),
    # Output array
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the matrix-vector product `out_vec = Jᵀ @ vec_x` (scatter).

    This kernel performs the action of the Jacobian transpose. It takes a vector `vec_x`
    from constraint-space (e.g., constraint impulses `λ`) and "scatters" its effects
    into `out_vec` in generalized-coordinate-space (e.g., generalized forces on bodies).

    Each thread corresponds to a single constraint `i`. It multiplies the scalar `x_i` by the
    corresponding Jacobian columns (`J_ia`, `J_ib`) and atomically adds the results to the
    output vectors for the affected bodies (`body_a`, `body_b`).

    Args:
        constraint_body_idx: A `(num_constraints, 2)` array mapping each constraint to the indices of the two bodies it connects.
        J_values: A `(num_constraints, 2)` array of `wp.spatial_vector` holding the Jacobian blocks for each constraint.
        vec_x: An input vector in constraint-space, with size `num_constraints`.
        out_vec: The output vector in generalized-coordinate-space, with size `num_bodies`. This vector MUST be zero-initialized before calling.
    """
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]
    x_i = vec_x[constraint_idx]

    if body_a >= 0:
        wp.atomic_add(out_vec, body_a, x_i * J_ia)
    if body_b >= 0:
        wp.atomic_add(out_vec, body_b, x_i * J_ib)


@wp.kernel
def kernel_inv_mass_matvec(
    gen_inv_mass: wp.array(dtype=SpatialInertia),
    in_vec: wp.array(dtype=wp.spatial_vector),
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the matrix-vector product `out_vec = M⁻¹ @ in_vec`.

    This kernel applies the inverse generalized mass matrix `M⁻¹` to a vector. Since `M⁻¹`
    is block-diagonal (one `6x6` block per body), this operation is highly parallel. Each
    thread handles one body, applying its inverse mass and inverse inertia tensor to the
    corresponding part of `in_vec`.

    Args:
        gen_inv_mass: An array of `SpatialInertia` structs, one for each body, containing its inverse mass and inverse inertia tensor.
        in_vec: A vector in generalized-coordinate-space (e.g., generalized forces).
        out_vec: The resulting vector in generalized-coordinate-space (e.g., generalized accelerations).
    """
    body_idx = wp.tid()

    out_vec[body_idx] = to_spatial_momentum(gen_inv_mass[body_idx], in_vec[body_idx])


@wp.kernel
def kernel_J_matvec(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    in_vec: wp.array(dtype=wp.spatial_vector),
    # Output array
    out_vec: wp.array(dtype=wp.float32),
):
    """
    Computes the matrix-vector product `out_vec = J @ in_vec` (gather).

    This kernel performs the action of the Jacobian matrix. It takes a vector `in_vec`
    from generalized-coordinate-space (e.g., generalized velocities) and "gathers"
    values to compute `out_vec` in constraint-space (e.g., relative velocities at constraints).

    Each thread corresponds to a single constraint `i`. It reads the generalized vectors
    for the two bodies involved and computes the dot product with the corresponding Jacobian
    rows to produce a single scalar value in `out_vec`.

    Args:
        constraint_body_idx: A `(num_constraints, 2)` array mapping each constraint to the indices of the two bodies it connects.
        J_values: A `(num_constraints, 2)` array of `wp.spatial_vector` holding the Jacobian blocks for each constraint.
        in_vec: An input vector in generalized-coordinate-space, with size `num_bodies`.
        out_vec: The output vector in constraint-space, with size `num_constraints`.
    """
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    result = 0.0
    if body_a >= 0:
        result += wp.dot(J_ia, in_vec[body_a])
    if body_b >= 0:
        result += wp.dot(J_ib, in_vec[body_b])

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
    Performs the final generalized matrix-vector product (GEMV) update.

    This kernel computes the final result `z` of the linear operator:
    `z = β*y + α*A*x`, where `A*x = (J*M⁻¹*Jᵀ*x) + (C*x)`.

    It combines the result of the `J M⁻¹ Jᵀ @ x` computation with the diagonal
    compliance term `C @ x` and scales the result by `alpha`, finally adding
    the scaled `y` term. This `alpha` and `beta` structure is standard for
    BLAS routines and iterative solvers.

    Args:
        J_M_inv_Jt_x: The result of `J @ M⁻¹ @ Jᵀ @ vec_x`.
        C_values: The diagonal entries of the compliance/regularization matrix `C`.
        vec_x: The original input vector 'x' to the matvec operation.
        vec_y: The input vector 'y' to be accumulated.
        alpha: Scalar multiplier for the `A@x` term.
        beta: Scalar multiplier for the `y` term.
        out_vec_z: The final output vector `z`.
    """
    i = wp.tid()
    c_times_x = C_values[i] * vec_x[i]
    a_times_x = J_M_inv_Jt_x[i] + c_times_x

    if beta == 0.0:
        out_vec_z[i] = alpha * a_times_x
    else:
        out_vec_z[i] = beta * vec_y[i] + alpha * a_times_x


class MatrixFreeSystemOperator(LinearOperator):
    """
    A matrix-free `LinearOperator` for the system matrix `A = J M⁻¹ Jᵀ + C`.

    This class adheres to the `warp.optim.linear.LinearOperator` interface, allowing it
    to be used seamlessly with iterative solvers like `cr_solver` or `cg_solver`. Instead of
    building the large, dense system matrix `A` in memory, it provides a `matvec` method
    that computes the product `A @ x` by applying the constituent matrices (`J`, `M⁻¹`, `Jᵀ`, `C`)
    as a sequence of efficient GPU kernel calls.

    Attributes:
        engine: A reference to the main Axion engine instance which holds all the live
            simulation data (Jacobians, masses, etc.).
    """

    def __init__(self, engine):
        """
        Initializes the operator with data from the main physics engine.

        Args:
            engine: An instance of `AxionEngine` that holds the system data like
                    Jacobians, masses, and dimension information.
        """
        super().__init__(
            shape=(engine.dims.con_dim, engine.dims.con_dim),
            dtype=wp.float32,
            device=engine.device,
            matvec=self.matvec,  # Hook up our matvec implementation
        )
        self.engine = engine

        # Pre-allocate temporary buffers for intermediate matvec calculations to avoid
        # re-allocation inside the solver loop.
        self._tmp_dyn_vec = wp.zeros(
            shape=engine.dims.N_b, dtype=wp.spatial_vector, device=engine.device
        )
        self._tmp_con_vec = wp.zeros(
            shape=engine.dims.con_dim, dtype=wp.float32, device=engine.device
        )

    def matvec(self, x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float):
        """
        Computes the matrix-vector product `z = β*y + α*(A @ x)`.

        This method implements the core logic of the matrix-free operator. It calculates
        the product `A @ x` where `A = J M⁻¹ Jᵀ + C` through a four-step process:
        1.  `v₁ = Jᵀ @ x`: Scatter from constraint-space to generalized-space.
        2.  `v₂ = M⁻¹ @ v₁`: Apply inverse mass per body.
        3.  `v₃ = J @ v₂`: Gather from generalized-space back to constraint-space.
        4.  `z = β*y + α*(v₃ + C@x)`: Final combination and scaling.

        Args:
            x: The input vector for the matrix-vector product.
            y: The vector to be accumulated.
            z: The output vector to store the result.
            alpha: The scaling factor for the `A@x` term.
            beta: The scaling factor for the `y` term.
        """
        # --- Step 1: Compute v₁ = Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        wp.launch(
            kernel=kernel_J_transpose_matvec,
            dim=self.engine.dims.con_dim,
            inputs=[
                self.engine.data.constraint_body_idx,
                self.engine.data.J_values,
                x,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 2: Compute v₂ = M⁻¹ @ v₁ ---
        wp.launch(
            kernel=kernel_inv_mass_matvec,
            dim=self.engine.dims.N_b,
            inputs=[
                self.engine.data.gen_inv_mass,
                self._tmp_dyn_vec,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # --- Step 3: Compute v₃ = J @ v₂ ---
        wp.launch(
            kernel=kernel_J_matvec,
            dim=self.engine.dims.con_dim,
            inputs=[
                self.engine.data.constraint_body_idx,
                self.engine.data.J_values,
                self._tmp_dyn_vec,
            ],
            outputs=[self._tmp_con_vec],
            device=self.device,
        )

        # --- Step 4: Compute z = β*y + α*(v₃ + C@x) ---
        wp.launch(
            kernel=kernel_finalize_matvec,
            dim=self.engine.dims.con_dim,
            inputs=[
                self._tmp_con_vec,  # This is J M⁻¹ Jᵀ @ x
                self.engine.data.C_values,  # Diagonal of C
                x,  # The original input vector 'x'
                y,  # The vector 'y' to add
                alpha,  # Scalar alpha
                beta,  # Scalar beta
            ],
            outputs=[z],  # The final output vector z
            device=self.device,
        )
