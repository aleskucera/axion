import numpy as np
import warp as wp

TILE_SIZE = 4
HALF_TILE_SIZE = 2


def create_tile_load_kernel():
    @wp.kernel
    def tile_load_kernel(
        arr: wp.array(dtype=wp.int32, ndim=2),
    ):
        tid, block_idx = wp.tid()

        t = wp.tile_load(arr, shape=(2, HALF_TILE_SIZE), offset=(tid * 2, 0))

        ti = wp.untile(wp.tile_reshape(t, shape=(TILE_SIZE,)))

        wp.printf("TID: %d, block_idx %d, value: %d\n", tid, block_idx, ti)

    return tile_load_kernel


def main():
    n = 10
    arr_np = np.arange(n).reshape(-1, TILE_SIZE // 2)
    arr = wp.array(arr_np, dtype=wp.int32)
    k = create_tile_load_kernel()

    wp.launch_tiled(
        kernel=k,
        dim=(len(arr) // 2,),
        inputs=[
            arr,
        ],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()


if __name__ == "__main__":
    main()
