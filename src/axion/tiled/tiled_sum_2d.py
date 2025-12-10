import warp as wp
import warp.context as wpc

# ------------------------------------------------------------------
# Kernel Definitions
# ------------------------------------------------------------------


def create_tiled_sum_kernel_2d(tile_size: int, dtype: type):
    """
    Creates a kernel for 2D data that effectively sums 'strips' of the array.
    Input: (Rows, Cols)
    Output: (Rows, Num_Blocks) -> Stores partial sums of each strip.
    """

    @wp.kernel
    def tiled_sum_kernel_2d(inp: wp.array(dtype=dtype, ndim=2), out: wp.array(dtype=dtype, ndim=2)):
        # tid is (row, block_index)
        row, block_idx = wp.tid()

        # Calculate the starting column for this tile
        start_col = block_idx * tile_size

        # Load a horizontal tile of shape (1, tile_size)
        # Offset is (row, start_col)
        tile = wp.tile_load(inp, shape=(1, tile_size), offset=(row, start_col))

        # Sum the tile. Result is a tile of shape (1, 1)
        tile_sum_val = wp.tile_sum(tile)
        tile_sum_val_2d = wp.tile_reshape(tile_sum_val, shape=(1, 1))

        # Store the (1,1) result into the partials array at (row, block_idx)
        wp.tile_store(out, tile_sum_val_2d, offset=(row, block_idx))

    return tiled_sum_kernel_2d


def create_atomic_sum_kernel_2d(dtype: type):
    """
    Creates a kernel that accumulates 2D partial sums into a 1D vector.
    Input: (Rows, Num_Blocks)
    Output: (Rows,)
    """

    @wp.kernel
    def atomic_sum_kernel_2d(
        partials: wp.array(dtype=dtype, ndim=2), out_vec: wp.array(dtype=dtype, ndim=1)
    ):
        # tid is (row, block_index)
        row, block_idx = wp.tid()

        # Read the partial sum calculated by the specific block
        val = partials[row, block_idx]

        # Atomically add to the row's total in the 1D output vector
        wp.atomic_add(out_vec, row, val)

    return atomic_sum_kernel_2d


# ------------------------------------------------------------------
# TiledSum Class (2D -> 1D)
# ------------------------------------------------------------------


class TiledSum:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 256,  # 1024 is often too wide for standard 2D bitmaps, 256 is safer
        block_threads: int = 128,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled sum computation for 2D arrays.
        Sums the columns of each row (reduction along axis 1).

        Args:
            shape: Shape of the 2D input array (ROWS, COLS).
            dtype: Data type (default wp.float32).
            tile_size: Width of the tile to sum in one go.
            block_threads: CUDA threads per block.
            device: Computing device.
        """
        self.shape = shape
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        assert len(self.shape) == 2, "Input shape must be 2D (ROWS, COLS)"
        ROWS, COLS = self.shape

        # Calculate how many blocks cover the columns
        self.NUM_BLOCKS = (COLS + self.tile_size - 1) // self.tile_size

        # Intermediate buffer: (ROWS, NUM_BLOCKS)
        self.partial_sums = wp.empty((ROWS, self.NUM_BLOCKS), dtype=self.dtype, device=self.device)

        # Create Kernels
        sum_kernel = create_tiled_sum_kernel_2d(self.tile_size, self.dtype)
        atomic_kernel = create_atomic_sum_kernel_2d(self.dtype)

        # Dummy arrays for graph recording match the shapes:
        # Input: (ROWS, COLS) -> 2D
        # Output: (ROWS) -> 1D
        a_dummy = wp.empty((ROWS, COLS), dtype=self.dtype, device=self.device)
        out_dummy = wp.empty((ROWS,), dtype=self.dtype, device=self.device)

        # 1. Tiled Reduction Launch configuration
        # Grid dimensions: (ROWS, NUM_BLOCKS)
        self.dot_launch = wp.launch_tiled(
            kernel=sum_kernel,
            dim=(ROWS, self.NUM_BLOCKS),
            inputs=[a_dummy],
            outputs=[self.partial_sums],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )

        # 2. Atomic Accumulation Launch configuration
        # Grid dimensions: (ROWS, NUM_BLOCKS)
        if self.NUM_BLOCKS > 0:
            self.atomic_sum_launch = wp.launch(
                kernel=atomic_kernel,
                dim=(ROWS, self.NUM_BLOCKS),
                inputs=[self.partial_sums],
                outputs=[out_dummy],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )

    def compute(self, a: wp.array, out: wp.array):
        """
        Computes out = sum(a, axis=1).

        Args:
            a: Input 2D array (ROWS, COLS).
            out: Output 1D array (ROWS,).
        """
        # Validations
        assert self.dtype == a.dtype == out.dtype, "Data types do not match."
        assert a.shape == self.shape, f"Input shape {a.shape} mismatch. Expected {self.shape}"
        assert out.shape == (
            self.shape[0],
        ), f"Output shape {out.shape} mismatch. Expected {(self.shape[0],)}"

        # 1. Clear accumulators
        out.zero_()

        # 2. Run Tiled Reduction (Input 2D -> Partial Sums 2D)
        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, self.partial_sums)
        self.dot_launch.launch()

        # 3. Accumulated Partials (Partial Sums 2D -> Output 1D)
        if self.NUM_BLOCKS > 0:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
