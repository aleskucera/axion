import numpy as np
import warp as wp

wp.init()

TILE_SIZE = 16


@wp.kernel
def compute():
    i = wp.tid()

    # create block-wide tile
    t = wp.tile(i) * 2

    # convert back to per-thread values
    s = wp.untile(t)

    wp.printf("%d \n", s)


def make_compute2():
    @wp.kernel
    def compute2(arr: wp.array(dtype=wp.int32)):
        i, block_idx = wp.tid()

        t = wp.tile_load(
            arr,
            TILE_SIZE,
            offset=i * TILE_SIZE,
        )

        # convert back to per-thread values
        s = wp.untile(t)

        wp.printf("%d \n", s)

    return compute2


if __name__ == "__main__":
    # --- First kernel (normal block width = 32) ---
    wp.launch(kernel=compute, dim=32, block_dim=32)
    wp.synchronize()

    # --- Create compute2 lazily ---
    compute2 = make_compute2()

    # --- Launch tiled kernel ---
    arr_np = np.arange(32)
    arr = wp.from_numpy(arr_np, dtype=wp.int32)

    wp.launch_tiled(kernel=compute2, dim=2, block_dim=TILE_SIZE, inputs=[arr])
    wp.synchronize()
