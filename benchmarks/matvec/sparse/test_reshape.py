import numpy as np
import warp as wp

wp.init()

TILE_SIZE = 4
HALF_TILE_SIZE = 2


@wp.kernel
def repeat_kernel(
    arr: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    i, block_idx = wp.tid()

    t = wp.tile_load(
        arr,
        HALF_TILE_SIZE,
        offset=i * HALF_TILE_SIZE,
    )

    t_reshaped = wp.tile_reshape(t, shape=(HALF_TILE_SIZE, 1))
    t_broadcasted = wp.tile_broadcast(t_reshaped, shape=(HALF_TILE_SIZE, 2))
    t_flat = wp.tile_reshape(t_broadcasted, shape=(TILE_SIZE,))

    wp.tile_store(out, t_flat)


if __name__ == "__main__":

    # --- Launch tiled kernel ---
    arr_np = np.arange(HALF_TILE_SIZE)
    arr = wp.from_numpy(arr_np, dtype=wp.int32)
    out = wp.zeros(TILE_SIZE, dtype=wp.int32)

    wp.launch_tiled(kernel=repeat_kernel, dim=1, block_dim=TILE_SIZE, inputs=[arr], outputs=[out])
    wp.synchronize()
    print(out.numpy())
