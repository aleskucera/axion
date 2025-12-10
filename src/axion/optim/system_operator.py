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
from dataclasses import dataclass

import warp as wp
from axion.tiled import TiledSum
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


# Define the interface required by the operator
@dataclass
class SystemLinearData:
    """Holds references to the specific arrays needed for A = JM^{-1}J^T + C"""

    # Dimensions
    N_w: int  # Num worlds
    N_c: int  # Num constraints
    N_b: int  # Num bodies

    # Data Arrays (References)
    J_values: wp.array  # shape=(N_w, N_c, 2) dtype=spatial
    constraint_body_idx: wp.array  # shape=(N_w, N_c, 2) dtype=int32
    constraint_active_mask: wp.array  # shape=(N_w, N_c) dtype=float
    C_values: wp.array  # shape=(N_w, N_c) dtype=float
    M_inv: wp.array  # shape=(N_w, N_b) dtype=SpatialInertia

    # Helper to extract this from your main engine
    @classmethod
    def from_engine(cls, engine):
        return cls(
            N_w=engine.dims.N_w,
            N_c=engine.dims.N_c,
            N_b=engine.dims.N_b,
            J_values=engine.data.J_values.full,
            constraint_body_idx=engine.data.constraint_body_idx.full,
            constraint_active_mask=engine.data.constraint_active_mask.full,
            C_values=engine.data.C_values.full,
            M_inv=engine.data.world_M_inv,
        )


@wp.kernel
def kernel_invM_Jt_matvec_scatter(
    # Inputs
    x: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # Output array
    out: wp.array(dtype=wp.spatial_vector, ndim=3),  # Buffer
):
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        return

    # Initialize a random state for this thread
    state = wp.rand_init(seed=constraint_idx)
    # Randomly select a sub-bin to write to
    sub_bin_index = wp.randi(state, 0, out.shape[-1])

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
        m_inv_1 = M_inv[world_idx, body_1]
        delta_v_1 = to_spatial_momentum(m_inv_1, jt_x_1)
        wp.atomic_add(out, world_idx, body_1, sub_bin_index, delta_v_1)

    if body_2 >= 0:
        # Repeat for the second body
        jt_x_2 = x_i * J_2
        m_inv_2 = M_inv[world_idx, body_2]
        delta_v_2 = to_spatial_momentum(m_inv_2, jt_x_2)
        wp.atomic_add(out, world_idx, body_2, sub_bin_index, delta_v_2)


@wp.kernel
def kernel_J_matvec_and_finalize(
    # Inputs for J @ v
    dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
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

    is_active = constraint_active_mask[world_idx, constraint_idx]

    # The result of A @ x.
    # Default is 0.0 (implied for inactive constraints)
    Ax_i = 0.0

    if is_active > 0.0:
        body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
        body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
        J_1 = J_values[world_idx, constraint_idx, 0]
        J_2 = J_values[world_idx, constraint_idx, 1]

        j_m_inv_jt_x = 0.0
        if body_1 >= 0:
            j_m_inv_jt_x += wp.dot(J_1, dyn_vec[world_idx, body_1])
        if body_2 >= 0:
            j_m_inv_jt_x += wp.dot(J_2, dyn_vec[world_idx, body_2])

        c_times_x = (C_values[world_idx, constraint_idx] + regularization) * vec_x[
            world_idx, constraint_idx
        ]
        Ax_i = j_m_inv_jt_x + c_times_x

    # Final composition.
    if beta == 0.0:
        out_vec_z[world_idx, constraint_idx] = alpha * Ax_i
    else:
        out_vec_z[world_idx, constraint_idx] = (
            beta * vec_y[world_idx, constraint_idx] + alpha * Ax_i
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
        data: SystemLinearData,
        device: wp.context.Device,
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
            shape=(data.N_w, data.N_c, data.N_c),
            dtype=wp.float32,
            device=device,
            matvec=None,
        ),
        self.data = data
        self.regularization = regularization

        # Pre-allocate temporary buffers for intermediate calculations.
        with wp.ScopedDevice(self.device):
            self._scatter_buffer = wp.zeros(
                (data.N_w, data.N_b, scatter_buffer_size), dtype=wp.spatial_vector
            )
            self._tmp_dyn_vec = wp.zeros((data.N_w, data.N_b), dtype=wp.spatial_vector)
            self._tmp_con_vec = wp.zeros((data.N_w, data.N_c), dtype=wp.float32)

        self.tiled_reducer = TiledSum(
            shape=(data.N_w, data.N_b, scatter_buffer_size),
            dtype=wp.spatial_vector,
            tile_size=reducer_tile_size,
            block_threads=reducer_block_threads,
            device=self.device,
        )

        self.events = [wp.Event(enable_timing=True) for _ in range(6)]

    def matvec(self, x, y, z, alpha, beta):
        """
        Computes the matrix-vector product: z = beta * y + alpha * (A @ x).
        """
        # --- Step 1a: Compute v₂ = M⁻¹ @ Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        self._scatter_buffer.zero_()

        # wp.record_event(self.events[0])

        wp.launch(
            kernel=kernel_invM_Jt_matvec_scatter,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                x,
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                self.data.M_inv,
            ],
            outputs=[self._scatter_buffer],
            device=self.device,
        )

        # wp.record_event(self.events[1])

        # --- Step 1b: (REDUCE) v₁ = sum(v₁_scattered) ---
        self.tiled_reducer.compute(a=self._scatter_buffer, out=self._tmp_dyn_vec)
        # wp.record_event(self.events[2])

        wp.launch(
            kernel=kernel_J_matvec_and_finalize,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                self._tmp_dyn_vec,  # This is v₂ = M⁻¹ @ Jᵀ @ x
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                self.data.C_values,
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
