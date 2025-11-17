import warp as wp


def create_atomic_sum_kernel(dtype: type):
    @wp.kernel
    def atomic_sum_kernel(
        inp: wp.array(dtype=dtype, ndim=2), result: wp.array(dtype=dtype, ndim=1)
    ):
        i, j = wp.tid()
        wp.atomic_add(result, i, inp[i, j])

    return atomic_sum_kernel


def create_tiled_sum_kernel(tile_size: int, dtype: type):
    @wp.kernel
    def tiled_sum_kernel(inp: wp.array(dtype=dtype, ndim=2), out: wp.array(dtype=dtype, ndim=2)):
        i, j = wp.tid()
        tile = wp.tile_load(inp, (1, tile_size), (i, j * tile_size))
        tile = wp.tile_sum(tile)
        tile = wp.tile_reshape(tile, shape=(1, 1))
        wp.tile_store(out, tile, offset=(i, j))

    return tiled_sum_kernel


def create_tiled_dot_kernel(tile_size: int, dtype: type):
    @wp.kernel
    def tiled_dot_kernel(
        a: wp.array(dtype=dtype, ndim=2),
        b: wp.array(dtype=dtype, ndim=2),
        result: wp.array(dtype=dtype, ndim=2),
    ):
        i, j = wp.tid()
        start_j = j * tile_size
        a_tile = wp.tile_load(a, shape=(1, tile_size), offset=(i, start_j))
        b_tile = wp.tile_load(b, shape=(1, tile_size), offset=(i, start_j))
        prod_tile = wp.tile_map(wp.mul, a_tile, b_tile)
        block_sum = wp.tile_sum(prod_tile)
        block_sum_reshaped = wp.tile_reshape(block_sum, shape=(1, 1))
        wp.tile_store(result, block_sum_reshaped, offset=(i, j))

    return tiled_dot_kernel
