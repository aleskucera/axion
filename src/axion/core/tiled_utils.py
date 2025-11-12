import warp as wp
import warp.context as wpc


TILE_SIZE = 1024
BLOCK_THREADS = 512


@wp.kernel
def atomic_sum_1d_kernel(
    inp: wp.array(dtype=wp.float32, ndim=1), result: wp.array(dtype=wp.float32, ndim=1)
):
    i = wp.tid()
    wp.atomic_add(result, 0, inp[i])


@wp.kernel
def atomic_sum_2d_kernel(
    inp: wp.array(dtype=wp.float32, ndim=2), result: wp.array(dtype=wp.float32, ndim=1)
):
    i, j = wp.tid()
    wp.atomic_add(result, i, inp[i, j])


@wp.kernel
def tiled_sum_1d_kernel(
    inp: wp.array(dtype=wp.float32, ndim=1), out: wp.array(dtype=wp.float32, ndim=1)
):
    i = wp.tid()
    tile = wp.tile_load(inp, (TILE_SIZE,), (i * TILE_SIZE,))
    tile = wp.tile_sum(tile)
    wp.tile_store(out, tile, offset=(i,))


@wp.kernel
def tiled_sum_2d_kernel(
    inp: wp.array(dtype=wp.float32, ndim=2), out: wp.array(dtype=wp.float32, ndim=2)
):
    i, j = wp.tid()
    tile = wp.tile_load(inp, (1, TILE_SIZE), (i, j * TILE_SIZE))
    tile = wp.tile_sum(tile)
    tile = wp.tile_reshape(tile, shape=(1, 1))
    wp.tile_store(out, tile, offset=(i, j))


@wp.kernel
def tiled_dot_1d_kernel(
    a: wp.array(dtype=wp.float32, ndim=1),
    b: wp.array(dtype=wp.float32, ndim=1),
    result: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    start_i = i * TILE_SIZE
    a_tile = wp.tile_load(a, shape=(TILE_SIZE,), offset=(start_i,))
    b_tile = wp.tile_load(b, shape=(TILE_SIZE,), offset=(start_i,))
    prod_tile = wp.tile_map(wp.mul, a_tile, b_tile)
    block_sum = wp.tile_sum(prod_tile)
    wp.tile_store(result, block_sum, offset=(i,))


@wp.kernel
def tiled_dot_2d_kernel(
    a: wp.array(dtype=wp.float32, ndim=2),
    b: wp.array(dtype=wp.float32, ndim=2),
    result: wp.array(dtype=wp.float32, ndim=2),
):
    i, j = wp.tid()
    start_j = j * TILE_SIZE
    a_tile = wp.tile_load(a, shape=(1, TILE_SIZE), offset=(i, start_j))
    b_tile = wp.tile_load(b, shape=(1, TILE_SIZE), offset=(i, start_j))
    prod_tile = wp.tile_map(wp.mul, a_tile, b_tile)
    block_sum = wp.tile_sum(prod_tile)
    block_sum_reshaped = wp.tile_reshape(block_sum, shape=(1, 1))
    wp.tile_store(result, block_sum_reshaped, offset=(i, j))


def tiled_sum_1d(x: wp.array, device: wpc.Device) -> float:
    N = x.shape[0]
    NUM_BLOCKS = N // TILE_SIZE
    EXTRA_BLOCK = 1 if N % TILE_SIZE > 0 else 0
    NUM_BLOCKS_TOTAL = NUM_BLOCKS + EXTRA_BLOCK
    partial_sums = wp.empty(NUM_BLOCKS_TOTAL, dtype=wp.float32, device=device)
    final_sum = wp.zeros(1, dtype=wp.float32, device=device)
    if NUM_BLOCKS > 0:
        wp.launch_tiled(
            kernel=tiled_sum_1d_kernel,
            dim=NUM_BLOCKS,
            inputs=[x],
            outputs=[partial_sums],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if EXTRA_BLOCK:
        wp.launch_tiled(
            kernel=tiled_sum_1d_kernel,
            dim=1,
            inputs=[x[TILE_SIZE * NUM_BLOCKS :]],
            outputs=[partial_sums[-1:]],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if NUM_BLOCKS > 0:
        wp.synchronize()
        wp.launch(
            atomic_sum_1d_kernel,
            dim=NUM_BLOCKS_TOTAL,
            inputs=[partial_sums],
            outputs=[final_sum],
            device=device,
        )
    else:
        final_sum = partial_sums
    return final_sum


def tiled_sum_2d(x: wp.array, device: wpc.Device) -> float:
    M, N = x.shape
    NUM_BLOCKS = N // TILE_SIZE
    EXTRA_BLOCK = 1 if N % TILE_SIZE > 0 else 0
    NUM_BLOCKS_TOTAL = NUM_BLOCKS + EXTRA_BLOCK
    partial_sums = wp.empty((M, NUM_BLOCKS_TOTAL), dtype=wp.float32, device=device)
    final_sum = wp.zeros(M, dtype=wp.float32, device=device)
    if NUM_BLOCKS > 0:
        wp.launch_tiled(
            kernel=tiled_sum_2d_kernel,
            dim=(M, NUM_BLOCKS),
            inputs=[x],
            outputs=[partial_sums],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if EXTRA_BLOCK:
        wp.launch_tiled(
            kernel=tiled_sum_2d_kernel,
            dim=(M, 1),
            inputs=[x[:, TILE_SIZE * NUM_BLOCKS :]],
            outputs=[partial_sums[:, -1:]],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if NUM_BLOCKS > 0:
        wp.synchronize()
        wp.launch(
            atomic_sum_2d_kernel,
            dim=(M, NUM_BLOCKS_TOTAL),
            inputs=[partial_sums],
            outputs=[final_sum],
            device=device,
        )
    else:
        final_sum = partial_sums[:, 0]
    return final_sum


def tiled_dot_1d(a: wp.array, b: wp.array, device: wpc.Device) -> wp.array:
    assert a.ndim == 1, "Input arrays must be 1D."
    assert a.shape == b.shape, "Input arrays must match."
    N = a.shape[0]
    NUM_BLOCKS = N // TILE_SIZE
    EXTRA_BLOCK = 1 if N % TILE_SIZE > 0 else 0
    NUM_BLOCKS_TOTAL = NUM_BLOCKS + EXTRA_BLOCK
    partial_sums = wp.zeros(NUM_BLOCKS_TOTAL, dtype=float, device=device)
    final_sum = wp.zeros(1, dtype=float, device=device)
    if NUM_BLOCKS > 0:
        wp.launch_tiled(
            kernel=tiled_dot_1d_kernel,
            dim=NUM_BLOCKS,
            inputs=[a, b],
            outputs=[partial_sums],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if EXTRA_BLOCK:
        wp.launch_tiled(
            kernel=tiled_dot_1d_kernel,
            dim=1,
            inputs=[a[TILE_SIZE * NUM_BLOCKS :], b[TILE_SIZE * NUM_BLOCKS :]],
            outputs=[partial_sums[-1:]],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if NUM_BLOCKS > 0:
        wp.synchronize()
        wp.launch(
            kernel=atomic_sum_1d_kernel,
            dim=NUM_BLOCKS_TOTAL,
            inputs=[partial_sums],
            outputs=[final_sum],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    else:
        final_sum = partial_sums
    return final_sum


def tiled_dot_2d(a: wp.array, b: wp.array, device: wpc.Device) -> wp.array:
    assert a.ndim == 2, "Input arrays must be 2D."
    assert a.shape == b.shape, "Input arrays must match."
    M, N = a.shape
    NUM_BLOCKS = N // TILE_SIZE
    EXTRA_BLOCK = 1 if N % TILE_SIZE > 0 else 0
    NUM_BLOCKS_TOTAL = NUM_BLOCKS + EXTRA_BLOCK
    partial_sums = wp.zeros((M, NUM_BLOCKS_TOTAL), dtype=float, device=device)
    final_sums = wp.zeros(M, dtype=float, device=device)
    if NUM_BLOCKS > 0:
        wp.launch_tiled(
            kernel=tiled_dot_2d_kernel,
            dim=(M, NUM_BLOCKS),
            inputs=[a, b],
            outputs=[partial_sums],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if EXTRA_BLOCK:
        wp.launch_tiled(
            kernel=tiled_dot_2d_kernel,
            dim=(M, 1),
            inputs=[a[:, TILE_SIZE * NUM_BLOCKS :], b[:, TILE_SIZE * NUM_BLOCKS :]],
            outputs=[partial_sums[:, -1:]],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    if NUM_BLOCKS > 0:
        wp.synchronize()
        wp.launch(
            kernel=atomic_sum_2d_kernel,
            dim=(M, NUM_BLOCKS_TOTAL),
            inputs=[partial_sums],
            outputs=[final_sums],
            block_dim=BLOCK_THREADS,
            device=device,
        )
    else:
        final_sums = partial_sums[:, 0]
    return final_sums


def tiled_sq_norm_1d(a: wp.array, device: wpc.Device) -> wp.array:
    return tiled_dot_1d(a, a, device)


def tiled_sq_norm_2d(a: wp.array, device: wpc.Device) -> wp.array:
    return tiled_dot_2d(a, a, device)
