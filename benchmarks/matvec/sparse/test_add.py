import numpy as np
import warp as wp

wp.init()

TILE_SIZE = 12
HALF_TILE_SIZE = 2


@wp.kernel
def repeat_kernel2(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    i, block_idx = wp.tid()

    a_tile = wp.tile_load(
        a,
        TILE_SIZE,
        offset=i * TILE_SIZE,
    )
    b_tile = wp.tile_load(
        b,
        HALF_TILE_SIZE,
        offset=i * HALF_TILE_SIZE,
    )

    b_reshaped = wp.tile_reshape(b_tile, shape=(HALF_TILE_SIZE, 1))
    b_broadcasted = wp.tile_broadcast(b_reshaped, shape=(HALF_TILE_SIZE, 2))
    b_flat = wp.tile_reshape(b_broadcasted, shape=(TILE_SIZE,))

    b_i = wp.untile(b_flat)
    wp.printf("Thread idx: %d, b val: %d\n", block_idx, b_i)

    out_tile = wp.tile_map(wp.add, a_tile, b_flat)
    out_i = wp.untile(out_tile)
    wp.printf("Thread idx: %d, out val: %d\n", block_idx, out_i)

    wp.tile_store(out, out_tile)


@wp.kernel
def repeat_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    tid, block_idx = wp.tid()

    a_tile = wp.tile_load(
        a,
        TILE_SIZE,
        offset=tid * TILE_SIZE,
    )

    b_val = b[block_idx // 2]
    out_tile = a_tile * b_val
    wp.tile_store(out, out_tile, offset=tid * TILE_SIZE)


if __name__ == "__main__":

    # --- Launch tiled kernel ---
    a_np = np.arange(TILE_SIZE)
    a = wp.from_numpy(a_np, dtype=wp.int32)  # [0, 1, 2, 3]

    b_np = np.arange(1, HALF_TILE_SIZE + 1)
    b = wp.from_numpy(b_np, dtype=wp.int32)  # [0, 1]

    out = wp.zeros(TILE_SIZE, dtype=wp.int32)

    # a + b.repeat(2) = [0, 1, 2, 3] + [0, 0, 1, 1]
    wp.launch_tiled(kernel=repeat_kernel, dim=1, block_dim=TILE_SIZE, inputs=[a, b], outputs=[out])
    wp.synchronize()
    print(out.numpy())
