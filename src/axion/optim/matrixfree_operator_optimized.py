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
import warp.context as wpc
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from warp.optim.linear import LinearOperator


def create_spatial_tiled_sum_kernel(tile_size: int):
    """
    FACTORY FUNCTION: Creates a specialized tiled sum kernel for a given tile_size.
    The tile_size is a compile-time constant for the returned kernel.
    """

    @wp.kernel
    def spatial_tiled_sum_kernel(
        inp: wp.array(dtype=wp.spatial_vector, ndim=2),
        out: wp.array(dtype=wp.spatial_vector, ndim=2),
    ):
        i, j = wp.tid()
        # tile_load uses the compile-time constant 'tile_size' from the factory closure
        tile = wp.tile_load(inp, (1, tile_size), (i, j * tile_size))
        tile = wp.tile_sum(tile)
        tile = wp.tile_reshape(tile, shape=(1, 1))
        wp.tile_store(out, tile, offset=(i, j))

    return spatial_tiled_sum_kernel


@wp.kernel
def spatial_atomic_sum_kernel(
    inp: wp.array(dtype=wp.spatial_vector, ndim=2),
    result: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    i, j = wp.tid()
    wp.atomic_add(result, i, inp[i, j])


class TiledSpatialVectorSum:
    def __init__(
        self,
        shape: int | tuple | list,
        tile_size: int = 256,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        self.shape = shape
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        if isinstance(self.shape, int):
            self.unsqueeze = True
            self.shape = (self.shape,)
            N = self.shape[0]
            M = 1
        elif len(self.shape) == 1:
            self.unsqueeze = True
            N = self.shape[0]
            M = 1
        elif len(self.shape) == 2:
            self.unsqueeze = False
            M, N = self.shape
        else:
            raise ValueError("Unknown shape")

        self.num_blocks = N // self.tile_size
        self.extra_block = 1 if N % self.tile_size > 0 else 0
        self.num_blocks_total = self.num_blocks + self.extra_block
        self.partial_sums = wp.empty(
            (M, self.num_blocks_total), dtype=wp.spatial_vector, device=device
        )

        sum_kernel: wp.Kernel = create_spatial_tiled_sum_kernel(self.tile_size)

        a = wp.empty((M, N), dtype=wp.spatial_vector, device=self.device)
        out = wp.empty(M, dtype=wp.spatial_vector, device=self.device)
        if self.num_blocks > 0:
            self.dot_launch: wp.Launch = wp.launch_tiled(
                kernel=sum_kernel,
                dim=(M, self.num_blocks),
                inputs=[a],
                outputs=[self.partial_sums],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )
        if self.extra_block:
            self.dot_extra_launch: wp.Launch = wp.launch_tiled(
                kernel=sum_kernel,
                dim=(M, 1),
                inputs=[
                    a[:, self.tile_size * self.num_blocks :],
                ],
                outputs=[self.partial_sums[:, -1:]],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )
        if self.num_blocks > 0:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=spatial_atomic_sum_kernel,
                dim=(M, self.num_blocks_total),
                inputs=[self.partial_sums],
                outputs=[out],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )

    def compute(self, a: wp.array, out: wp.array):
        """
        Computes the sum of the input array: out = sum(a) along the last axis.
        Args:
            a: Input array.
            out: Output array to store the result.
        """

        assert (
            a.shape == self.shape
        ), f"Shapes do not match. a.shape = {a.shape}, expected {self.shape}"
        assert a.shape[0] == out.shape[0] or (
            a.ndim == 1 and out.shape[0] == 1
        ), f"Inupt and output dimensions do not match. a.shape = {a.shape}, out.shape = {out.shape}"

        # Use wp.zeros to initialize instead of fill_() to be CUDA graph compatible
        self.partial_sums.zero_()
        if self.unsqueeze:
            a = a.reshape((1, -1))

        if self.num_blocks > 0:
            self.dot_launch.set_param_at_index(0, a)
            self.dot_launch.set_param_at_index(1, self.partial_sums)
            self.dot_launch.launch()

        if self.extra_block:
            self.dot_extra_launch.set_param_at_index(0, a[:, self.tile_size * self.num_blocks :])
            self.dot_extra_launch.set_param_at_index(1, self.partial_sums[:, -1:])
            self.dot_extra_launch.launch()

        if self.num_blocks > 0:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, 0])


@wp.kernel
def kernel_J_transpose_scatter(
    in_vec: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Output array
    out_vec_2d: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    """
    SCATTER step.
    Computes Jᵀ @ x by scattering results into a 2D intermediate buffer
    to avoid atomic contention.
    """
    constraint_idx = wp.tid()

    # Initialize a random state for this thread
    state = wp.rand_init(seed=constraint_idx)
    # Randomly select a sub-bin to write to
    sub_bin_index = wp.randi(state, 0, out_vec_2d.shape[1])

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]
    x_i = in_vec[constraint_idx]

    if body_1 >= 0:
        wp.atomic_add(out_vec_2d, body_1, sub_bin_index, x_i * J_1)
    if body_2 >= 0:
        wp.atomic_add(out_vec_2d, body_2, sub_bin_index, x_i * J_2)


@wp.kernel
def kernel_J_transpose_reduce(
    in_vec_2d: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Final output array (1D)
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    """
    REDUCE step.
    Sums the results from the intermediate 2D buffer into the final 1D vector.
    This kernel is launched with one thread per body and has no atomics.
    """
    body_idx = wp.tid()

    # Each thread sums up the sub-bins for its assigned body
    total_vec = wp.spatial_vector()  # Initializes to zero
    for i in range(in_vec_2d.shape[1]):
        total_vec += in_vec_2d[body_idx, i]

    out_vec[body_idx] = total_vec


@wp.kernel
def kernel_J_transpose_matvec(
    in_vec: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    out_vec: wp.array(dtype=wp.spatial_vector),
):
    constraint_idx = wp.tid()
    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]
    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]
    x_i = in_vec[constraint_idx]
    if body_1 >= 0:
        wp.atomic_add(out_vec, body_1, x_i * J_1)
    if body_2 >= 0:
        wp.atomic_add(out_vec, body_2, x_i * J_2)


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
            shape=(engine.dims.N_c, engine.dims.N_c),
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
                shape=(engine.dims.N_b, self.scatter_buffer_size),
                dtype=wp.spatial_vector,
            )
            self._tmp_dyn_vec = wp.zeros(engine.dims.N_b, dtype=wp.spatial_vector)
            self._tmp_con_vec = wp.zeros(engine.dims.N_c, dtype=wp.float32)

        self.tiled_reducer = TiledSpatialVectorSum(
            shape=(engine.dims.N_b, self.scatter_buffer_size),
            tile_size=reducer_tile_size,
            block_threads=reducer_block_threads,
            device=engine.device,
        )

        self.events = [wp.Event(enable_timing=True) for _ in range(6)]

    def matvec(self, x, y, z, alpha, beta):
        """
        Computes the matrix-vector product: z = beta * y + alpha * (A @ x).
        """
        # --- Step 1: Compute v₁ = Jᵀ @ x ---
        self._tmp_dyn_vec.zero_()
        self._tmp_dyn_vec_buffer.zero_()

        # wp.record_event(self.events[0])

        wp.launch(
            kernel=kernel_J_transpose_scatter,
            dim=self.engine.dims.N_c,
            inputs=[
                x,
                self.engine.data.J_values,
                self.engine.data.constraint_body_idx,
            ],
            outputs=[self._tmp_dyn_vec_buffer],
            device=self.device,
        )
        # wp.record_event(self.events[1])

        # --- Step 1b: (REDUCE) v₁ = sum(v₁_scattered) ---
        # wp.launch(
        #     kernel=kernel_J_transpose_reduce,
        #     dim=self.engine.dims.N_b,
        #     inputs=[
        #         self._tmp_dyn_vec_buffer,
        #     ],
        #     outputs=[self._tmp_dyn_vec],
        #     device=self.device,
        # )
        self.tiled_reducer.compute(a=self._tmp_dyn_vec_buffer, out=self._tmp_dyn_vec)
        # wp.record_event(self.events[2])

        # --- Step 2: Compute v₂ = M⁻¹ @ v₁ ---
        wp.launch(
            kernel=kernel_inv_mass_matvec,
            dim=self.engine.dims.N_b,
            inputs=[
                self._tmp_dyn_vec,
                self.engine.data.world_M_inv,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # wp.record_event(self.events[3])

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

        # wp.record_event(self.events[4])

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

        # wp.record_event(self.events[5])

        # return [
        #     wp.get_event_elapsed_time(self.events[0], self.events[1]),
        #     wp.get_event_elapsed_time(self.events[1], self.events[2]),
        #     wp.get_event_elapsed_time(self.events[2], self.events[3]),
        #     wp.get_event_elapsed_time(self.events[3], self.events[4]),
        #     wp.get_event_elapsed_time(self.events[4], self.events[5]),
        # ]
