import axion.tiled.tiled_kernels as tk
import warp as wp
import warp.context as wpc


class TiledDot:
    def __init__(
        self,
        shape: int | tuple | list,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled dot product computation.

        Args:
            shape: Shape of the input arrays. Can be N or (M, N). Where N is the
                   length of the vectors and M is the batch size.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

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
        self.partial_sums = wp.empty((M, self.num_blocks_total), dtype=wp.float32, device=device)

        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel()
        dot_kernel: wp.Kernel = tk.create_tiled_dot_kernel(self.tile_size)

        a = wp.empty((M, N), dtype=wp.float32, device=self.device)
        b = wp.empty((M, N), dtype=wp.float32, device=self.device)
        out = wp.empty(M, dtype=wp.float32, device=self.device)
        if self.num_blocks > 0:
            self.dot_launch: wp.Launch = wp.launch_tiled(
                kernel=dot_kernel,
                dim=(M, self.num_blocks),
                inputs=[a, b],
                outputs=[self.partial_sums],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )
        if self.extra_block:
            self.dot_extra_launch: wp.Launch = wp.launch_tiled(
                kernel=dot_kernel,
                dim=(M, 1),
                inputs=[
                    a[:, self.tile_size * self.num_blocks :],
                    b[:, self.tile_size * self.num_blocks :],
                ],
                outputs=[self.partial_sums[:, -1:]],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )
        if self.num_blocks > 0:
            self.atomic_sum_launch: wp.Launch = wp.launch(
                kernel=atomic_sum_kernel,
                dim=(M, self.num_blocks_total),
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
            out: Output array to store the result.
        """

        assert (
            a.shape == b.shape == self.shape
        ), f"Shapes do not match. a.shape = {a.shape}, b.shape = {b.shape}, expected {self.shape}"
        assert a.shape[0] == out.shape[0] or (
            a.ndim == 1 and out.shape[0] == 1
        ), f"Input and output dimensions do not match. a.shape = {a.shape}, out.shape = {out.shape}"

        out.fill_(0.0)
        if self.unsqueeze:
            a = a.reshape((1, -1))
            b = b.reshape((1, -1))

        if self.num_blocks > 0:
            self.dot_launch.set_param_at_index(0, a)
            self.dot_launch.set_param_at_index(1, b)
            self.dot_launch.set_param_at_index(2, self.partial_sums)
            self.dot_launch.launch()

        if self.extra_block:
            self.dot_extra_launch.set_param_at_index(0, a[:, self.tile_size * self.num_blocks :])
            self.dot_extra_launch.set_param_at_index(1, b[:, self.tile_size * self.num_blocks :])
            self.dot_extra_launch.set_param_at_index(2, self.partial_sums[:, -1:])
            self.dot_extra_launch.launch()

        if self.num_blocks > 0:
            self.atomic_sum_launch.set_param_at_index(0, self.partial_sums)
            self.atomic_sum_launch.set_param_at_index(1, out)
            self.atomic_sum_launch.launch()
        else:
            wp.copy(out, self.partial_sums[:, 0])


class TiledSqrNorm(TiledDot):
    def __init__(self, shape, tile_size=1024, block_threads=512, device="cuda"):
        """
        Tiled squared norm computation.

        Args:
            shape: Shape of the input array. Can be N or (M, N). Where N is the
                length of the vectors and M is the batch size.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

        super().__init__(shape, tile_size, block_threads, device)
        self.compute_dot = super().compute

    def compute(self, a: wp.array, out: wp.array):
        """
        Computes the squared norm of the input array: out = sum(a * a) along the last axis.

        Args:
            a: Input array.
            out: Output array to store the result.
        """
        self.compute_dot(a, a, out)


class TiledSum:
    def __init__(
        self,
        shape: int | tuple | list,
        tile_size: int = 1024,
        block_threads: int = 512,
        device: wpc.Device | str = "cuda",
    ):
        """
        Tiled sum computation.

        Args:
            shape: Shape of the input array. Can be N or (M, N). Where N is the
                   length of the vectors and M is the batch size.
            tile_size: Size of the tiles. Defaults to 1024.
            block_threads: Number of threads per block. Defaults to 512.
            device: Device to run the computation on. Defaults to "cuda". Can be
                    a warp Device or a string.
        """

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
        self.partial_sums = wp.empty((M, self.num_blocks_total), dtype=wp.float32, device=device)

        atomic_sum_kernel: wp.Kernel = tk.create_atomic_sum_kernel()
        sum_kerenl: wp.Kernel = tk.create_tiled_sum_kernel(self.tile_size)

        a = wp.empty((M, N), dtype=wp.float32, device=self.device)
        out = wp.empty(M, dtype=wp.float32, device=self.device)
        if self.num_blocks > 0:
            self.dot_launch: wp.Launch = wp.launch_tiled(
                kernel=sum_kerenl,
                dim=(M, self.num_blocks),
                inputs=[a],
                outputs=[self.partial_sums],
                block_dim=self.block_threads,
                device=self.device,
                record_cmd=True,
            )
        if self.extra_block:
            self.dot_extra_launch: wp.Launch = wp.launch_tiled(
                kernel=sum_kerenl,
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
                kernel=atomic_sum_kernel,
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

        out.fill_(0.0)
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
