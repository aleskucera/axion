import axion.tiled.tiled_kernels as tk
import warp as wp
import warp.context as wpc


class TiledDot:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled dot product computation.

        Args:
            shape: Shape of the input arrays. The shape is (B, M, N). Where B is batch size, N is the
                   length of the vectors and M is the number of rows.
            dtype: Data type of the input arrays. Defaults to wp.float32.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

        self.shape = shape
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        assert len(self.shape) == 3, "Input shape must be 3D (BATCH, ROWS, COLS)"
        BATCH, ROWS, COLS = self.shape

        self.NUM_BLOCKS = (COLS + self.tile_size - 1) // self.tile_size
        self.partial_sums = wp.empty(
            (BATCH, ROWS, self.NUM_BLOCKS), dtype=self.dtype, device=self.device
        )

        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel(self.dtype)
        dot_kernel: wp.Kernel = tk.create_tiled_dot_kernel(self.tile_size, self.dtype)

        a = wp.empty((BATCH, ROWS, COLS), dtype=self.dtype, device=self.device)
        b = wp.empty((BATCH, ROWS, COLS), dtype=self.dtype, device=self.device)
        out = wp.empty((BATCH, ROWS), dtype=self.dtype, device=self.device)
        self.dot_launch: wp.Launch = wp.launch_tiled(
            kernel=dot_kernel,
            dim=(BATCH, ROWS, self.NUM_BLOCKS),
            inputs=[a, b],
            outputs=[self.partial_sums],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )
        if self.NUM_BLOCKS > 1:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=atomic_sum_kernel,
                dim=(BATCH, ROWS, self.NUM_BLOCKS),
                inputs=[self.partial_sums],
                outputs=[out],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )

    def compute(self, a: wp.array, b: wp.array, out: wp.array):
        """
        Computes the dot product of two arrays: out = sum(a * b) along the last axis.

        Args:
            a: First input array.
            b: Second input array.
            out: 1D output array to store the result.
        """
        assert self.dtype == a.dtype == b.dtype == out.dtype, "Data types do not match."
        assert (
            a.shape == b.shape == self.shape
        ), f"Input and expoected shapes do not match. a.shape = {a.shape}, b.shape = {b.shape}, expected {self.shape}"
        assert (self.shape[0] == out.shape[0]) and (
            self.shape[1] == out.shape[1]
        ), f"Output shape does not match. out.shape = {out.shape}, expected {(self.shape[0], self.shape[1])}"

        out.zero_()

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, b)
        self.dot_launch.set_param_at_index(2, self.partial_sums)
        self.dot_launch.launch()

        if self.NUM_BLOCKS > 1:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, :, 0])


class TiledSqrNorm(TiledDot):
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled squared norm computation.

        Args:
            shape: Shape of the input array. The shape is (B, M, N). Where B is the batch size, N is the
                length of the vectors and M is the number of rows.
            dtype: Data type of the input arrays. Defaults to wp.float32.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

        super().__init__(shape, dtype, tile_size, block_threads, device)
        self.compute_dot = super().compute

    def compute(self, a: wp.array, out: wp.array):
        """
        Computes the squared norm of the input array: out = sum(a * a) along the last axis.

        Args:
            a: Input array.
            out: 1D output array to store the result.
        """
        self.compute_dot(a, a, out)


class TiledSum:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled sum computation.

        Args:
            shape: Shape of the input array. The shape is (B, M, N). Where B is the batch size, N is the
                   length of the vectors and M is the number of rows.
            dtype: Data type of the input array. Defaults to wp.float32.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

        self.shape = shape
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        assert len(self.shape) == 3, "Input shape must be 3D (BATCH, ROWS, COLS)"
        BATCH, ROWS, COLS = self.shape

        self.NUM_BLOCKS = (COLS + self.tile_size - 1) // self.tile_size
        self.partial_sums = wp.empty(
            (BATCH, ROWS, self.NUM_BLOCKS), dtype=self.dtype, device=self.device
        )
        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel(self.dtype)
        sum_kernel: wp.Kernel = tk.create_tiled_sum_kernel(self.tile_size, self.dtype)

        a = wp.empty((BATCH, ROWS, COLS), dtype=self.dtype, device=self.device)
        out = wp.empty((BATCH, ROWS), dtype=self.dtype, device=self.device)
        self.dot_launch: wp.Launch = wp.launch_tiled(
            kernel=sum_kernel,
            dim=(BATCH, ROWS, self.NUM_BLOCKS),
            inputs=[a],
            outputs=[self.partial_sums],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )
        if self.NUM_BLOCKS > 1:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=atomic_sum_kernel,
                dim=(BATCH, ROWS, self.NUM_BLOCKS),
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
            out: 1D output array to store the result.
        """

        assert self.dtype == a.dtype == out.dtype, "Data types do not match."
        assert (
            a.shape == self.shape
        ), f"Input and expected shapes do not match. a.shape = {a.shape}, expected {self.shape}"
        assert (self.shape[0] == out.shape[0]) and (
            self.shape[1] == out.shape[1]
        ), f"Output shape does not match. out.shape = {out.shape}, expected {(self.shape[0], self.shape[1])}"

        out.zero_()

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, self.partial_sums)
        self.dot_launch.launch()

        if self.NUM_BLOCKS > 1:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, :, 0])
