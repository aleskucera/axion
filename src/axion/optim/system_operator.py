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
from axion.tiled import TiledSum
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


@wp.kernel
def kernel_invM_Jt_matvec_scatter(
    # Inputs
    x: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    body_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # Output array
    out_vec_buffer: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    world_idx, constraint_idx = wp.tid()

    # Initialize a random state for this thread
    state = wp.rand_init(seed=constraint_idx)
    # Randomly select a sub-bin to write to
    sub_bin_index = wp.randi(state, 0, out_vec_buffer.shape[-1])

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]
    x_i = x[world_idx, constraint_idx]

    if body_1 >= 0:
        # Calculate the momentum contribution from this constraint
        jt_x_1 = x_i * J_1
        # Convert momentum to velocity change (v = M⁻¹ * p)
        # and atomically add it to the output velocity vector
        m_inv_1 = body_M_inv[world_idx, body_1]
        delta_v_1 = to_spatial_momentum(m_inv_1, jt_x_1)
        wp.atomic_add(out_vec_buffer, world_idx, body_1, sub_bin_index, delta_v_1)

    if body_2 >= 0:
        # Repeat for the second body
        jt_x_2 = x_i * J_2
        m_inv_2 = body_M_inv[world_idx, body_2]
        delta_v_2 = to_spatial_momentum(m_inv_2, jt_x_2)
        wp.atomic_add(out_vec_buffer, world_idx, body_2, sub_bin_index, delta_v_2)


@wp.kernel
def kernel_J_matvec_and_finalize(
    # Inputs for J @ v
    dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    # Inputs for finalize
    C_values: wp.array(dtype=wp.float32, ndim=2),
    vec_x: wp.array(dtype=wp.float32, ndim=2),
    vec_y: wp.array(dtype=wp.float32, ndim=2),
    alpha: float,
    beta: float,
    regularization: float,
    # Output array
    out_vec_z: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Computes J @ dyn_vec and immediately uses it in the finalization step.
    This fused kernel calculates:
    z = beta * y + alpha * ( (J @ dyn_vec) + (C + reg) @ x )
    """
    world_idx, constraint_idx = wp.tid()

    # --- Part 1: Compute J @ dyn_vec for this constraint (from kernel_J_matvec) ---
    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]

    j_m_inv_jt_x = 0.0
    if body_1 >= 0:
        j_m_inv_jt_x += wp.dot(J_1, dyn_vec[world_idx, body_1])
    if body_2 >= 0:
        j_m_inv_jt_x += wp.dot(J_2, dyn_vec[world_idx, body_2])

    # --- Part 2: Use the result immediately (from kernel_finalize_matvec) ---
    c_times_x = (C_values[world_idx, constraint_idx] + regularization) * vec_x[
        world_idx, constraint_idx
    ]
    a_times_x = j_m_inv_jt_x + c_times_x

    if beta == 0.0:
        out_vec_z[world_idx, constraint_idx] = alpha * a_times_x
    else:
        out_vec_z[world_idx, constraint_idx] = (
            beta * vec_y[world_idx, constraint_idx] + alpha * a_times_x
        )


class SystemOperator(LinearOperator):
    """
    A matrix-free linear operator for the system A = J M⁻¹ Jᵀ + C.

    This class provides a .matvec() method that computes the matrix-vector
    product A @ x without explicitly constructing the matrix A. It is intended
    to be used with iterative linear solvers like `cr_solver`.
    """

    def __init__(
        self,
        engine,
        regularization: float = 1e-5,
        scatter_buffer_size: int = 512,
        reducer_tile_size: int = 256,
        reducer_block_threads: int = 256,
    ):
        """
        Initializes the operator with data from the main physics engine.

        Args:
            engine: An instance of the main physics engine (e.g., NSNEngine)
                    that holds all the necessary system data (Jacobians,
                    masses, constraint info, etc.).
        """
        super().__init__(
            shape=(engine.dims.N_w, engine.dims.N_c, engine.dims.N_c),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,
        ),
        self.engine = engine
        self.regularization = regularization
        self.scatter_buffer_size = scatter_buffer_size

        # Pre-allocate temporary buffers for intermediate calculations.
        with wp.ScopedDevice(self.device):
            self._tmp_dyn_vec_buffer = wp.zeros(
                shape=(engine.dims.N_w, engine.dims.N_b, self.scatter_buffer_size),
                dtype=wp.spatial_vector,
            )
            self._tmp_dyn_vec = wp.zeros(
                (engine.dims.N_w, engine.dims.N_b), dtype=wp.spatial_vector
            )
            self._tmp_con_vec = wp.zeros((engine.dims.N_w, engine.dims.N_c), dtype=wp.float32)

        self.tiled_reducer = TiledSum(
            shape=(engine.dims.N_w, engine.dims.N_b, self.scatter_buffer_size),
            dtype=wp.spatial_vector,
            tile_size=reducer_tile_size,
            block_threads=reducer_block_threads,
            device=engine.device,
        )

        self.events = [wp.Event(enable_timing=True) for _ in range(6)]

    def matvec(self, x, y, z, alpha, beta):
        """
        Computes the matrix-vector product: z = beta * y + alpha * (A @ x).
        """
        # --- Step 1a: Compute v₂ = M⁻¹ @ Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        self._tmp_dyn_vec_buffer.zero_()

        # wp.record_event(self.events[0])
        wp.launch(
            kernel=kernel_invM_Jt_matvec_scatter,
            dim=(self.engine.dims.N_w, self.engine.dims.N_c),
            inputs=[
                x,
                self.engine.data.J_values.full,
                self.engine.data.constraint_body_idx.full,
                self.engine.data.world_M_inv,
            ],
            outputs=[self._tmp_dyn_vec_buffer],
            device=self.device,
        )

        # wp.record_event(self.events[1])

        # --- Step 1b: (REDUCE) v₁ = sum(v₁_scattered) ---
        self.tiled_reducer.compute(a=self._tmp_dyn_vec_buffer, out=self._tmp_dyn_vec)
        # wp.record_event(self.events[2])

        wp.launch(
            kernel=kernel_J_matvec_and_finalize,
            dim=(self.engine.dims.N_w, self.engine.dims.N_c),
            inputs=[
                self._tmp_dyn_vec,  # This is v₂ = M⁻¹ @ Jᵀ @ x
                self.engine.data.J_values.full,
                self.engine.data.constraint_body_idx.full,
                self.engine.data.C_values.full,
                x,
                y,
                alpha,
                beta,
                self.regularization,
            ],
            outputs=[z],
            device=self.device,
        )

        # wp.record_event(self.events[3])

        # return [
        #     wp.get_event_elapsed_time(self.events[0], self.events[1]),
        #     wp.get_event_elapsed_time(self.events[1], self.events[2]),
        #     wp.get_event_elapsed_time(self.events[2], self.events[3]),
        # ]
