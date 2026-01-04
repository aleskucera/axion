import numpy as np
import warp as wp

TILE_M = wp.constant(2)
TILE_N = wp.constant(2)
HALF_M = wp.constant(TILE_M // 2)
HALF_N = wp.constant(TILE_N // 2)


@wp.kernel
def compute(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()

    evens = wp.tile_arange(HALF_M, dtype=int, storage="shared") * 2

    t0 = wp.tile_load_indexed(
        x,
        indices=evens,
        shape=(HALF_M, TILE_N),
        offset=(i * TILE_M, j * TILE_N),
        axis=0,
        storage="register",
    )
    wp.tile_store(y, t0, offset=(i * HALF_M, j * TILE_N))


M = TILE_M * 2
N = TILE_N * 2

arr = np.arange(M * N).reshape(M, N)

x = wp.array(arr, dtype=float)
y = wp.zeros((M // 2, N), dtype=float)

wp.launch_tiled(compute, dim=[2, 2], inputs=[x], outputs=[y], block_dim=32)

print(x.numpy())
print(y.numpy())
