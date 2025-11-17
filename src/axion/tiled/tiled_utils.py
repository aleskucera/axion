import axion.tiled.tiled_kernels as tk
import warp as wp
import warp.context as wpc


class TiledDot:
    def __init__(
        self,
        shape: int | tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled dot product computation.

        Args:
            shape: Shape of the input arrays. Can be N or (M, N). Where N is the
                   length of the vectors and M is the batch size.
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

        self.num_blocks = (N + self.tile_size - 1) // self.tile_size
        self.partial_sums = wp.empty((M, self.num_blocks), dtype=self.dtype, device=self.device)

        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel(self.dtype)
        dot_kernel: wp.Kernel = tk.create_tiled_dot_kernel(self.tile_size, self.dtype)

        a = wp.empty((M, N), dtype=self.dtype, device=self.device)
        b = wp.empty((M, N), dtype=self.dtype, device=self.device)
        out = wp.empty(M, dtype=self.dtype, device=self.device)
        self.dot_launch: wp.Launch = wp.launch_tiled(
            kernel=dot_kernel,
            dim=(M, self.num_blocks),
            inputs=[a, b],
            outputs=[self.partial_sums],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )
        if self.num_blocks > 1:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=atomic_sum_kernel,
                dim=(M, self.num_blocks),
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
        ), f"Shapes do not match. a.shape = {a.shape}, b.shape = {b.shape}, expected {self.shape}"
        assert a.shape[0] == out.shape[0] or (
            a.ndim == 1 and out.shape[0] == 1
        ), f"Input and output dimensions do not match. a.shape = {a.shape}, out.shape = {out.shape}"

        out.zero_()
        if self.unsqueeze:
            a = a.reshape((1, -1))
            b = b.reshape((1, -1))

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, b)
        self.dot_launch.set_param_at_index(2, self.partial_sums)
        self.dot_launch.launch()

        if self.num_blocks > 1:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, 0])


class TiledSqrNorm(TiledDot):
    def __init__(
        self,
        shape: int | tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled squared norm computation.

        Args:
            shape: Shape of the input array. Can be N or (M, N). Where N is the
                length of the vectors and M is the batch size.
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
        shape: int | tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled sum computation.

        Args:
            shape: Shape of the input array. Can be N or (M, N). Where N is the
                   length of the vectors and M is the batch size.
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

        self.num_blocks = (N + self.tile_size - 1) // self.tile_size
        self.partial_sums = wp.empty((M, self.num_blocks), dtype=self.dtype, device=self.device)

        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel(self.dtype)
        sum_kernel: wp.Kernel = tk.create_tiled_sum_kernel(self.tile_size, self.dtype)

        a = wp.empty((M, N), dtype=self.dtype, device=self.device)
        out = wp.empty(M, dtype=self.dtype, device=self.device)
        self.dot_launch: wp.Launch = wp.launch_tiled(
            kernel=sum_kernel,
            dim=(M, self.num_blocks),
            inputs=[a],
            outputs=[self.partial_sums],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )
        if self.num_blocks > 1:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=atomic_sum_kernel,
                dim=(M, self.num_blocks),
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
        ), f"Shapes do not match. a.shape = {a.shape}, expected {self.shape}"
        assert a.shape[0] == out.shape[0] or (
            a.ndim == 1 and out.shape[0] == 1
        ), f"Inupt and output dimensions do not match. a.shape = {a.shape}, out.shape = {out.shape}"

        out.zero_()
        if self.unsqueeze:
            a = a.reshape((1, -1))

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, self.partial_sums)
        self.dot_launch.launch()

        if self.num_blocks > 1:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, 0])
