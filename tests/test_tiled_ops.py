import numpy as np
import pytest
import warp as wp
from axion.tiled import TiledArgMin
from axion.tiled import TiledDot
from axion.tiled import TiledSqNorm
from axion.tiled import TiledSum

wp.init()


@pytest.mark.parametrize(
    "shape", [(128, 256), (10, 20, 128), (2, 5, 10, 64), (1024,)]  # 2D  # 3D  # 4D  # 1D
)
def test_tiled_sum(shape):
    device = "cuda" if wp.is_cuda_available() else "cpu"

    # Generate random data
    np_a = np.random.rand(*shape).astype(np.float32)

    # Expected result (sum along last axis)
    np_out = np.sum(np_a, axis=-1).astype(np.float32)

    # Warp Arrays
    # Force scalar float32 dtype to avoid Warp inferring vectors/matrices
    wp_a = wp.from_numpy(np_a, dtype=wp.float32, device=device)
    # Output shape is shape[:-1]
    out_shape = shape[:-1]

    # TiledSum output is array.
    if len(shape) == 1:
        # For 1D input, output is (1,) to match Warp limitation
        wp_out = wp.zeros((1,), dtype=wp.float32, device=device)
        tiled_op = TiledSum(shape, device=device)
        tiled_op.compute(wp_a, wp_out)
        res = wp_out.numpy()[0]  # Extract scalar from (1,) array
        assert np.allclose(res, np_out, atol=1e-4)
    else:
        wp_out = wp.zeros(out_shape, dtype=wp.float32, device=device)
        tiled_op = TiledSum(shape, device=device)
        tiled_op.compute(wp_a, wp_out)
        res = wp_out.numpy()
        assert np.allclose(res, np_out, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (128, 256),
        (10, 20, 128),
    ],
)
def test_tiled_dot(shape):
    device = "cuda" if wp.is_cuda_available() else "cpu"

    np_a = np.random.rand(*shape).astype(np.float32)
    np_b = np.random.rand(*shape).astype(np.float32)

    # Expected: sum(a*b, axis=-1)
    np_out = np.sum(np_a * np_b, axis=-1).astype(np.float32)

    wp_a = wp.from_numpy(np_a, dtype=wp.float32, device=device)
    wp_b = wp.from_numpy(np_b, dtype=wp.float32, device=device)
    wp_out = wp.zeros(shape[:-1], dtype=wp.float32, device=device)

    tiled_op = TiledDot(shape, device=device)
    tiled_op.compute(wp_a, wp_b, wp_out)

    assert np.allclose(wp_out.numpy(), np_out, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (128, 256),
        (5, 5, 64),
    ],
)
def test_tiled_sqr_norm(shape):
    device = "cuda" if wp.is_cuda_available() else "cpu"

    np_a = np.random.rand(*shape).astype(np.float32)

    # Expected: sum(a*a, axis=-1)
    np_out = np.sum(np_a * np_a, axis=-1).astype(np.float32)

    wp_a = wp.from_numpy(np_a, dtype=wp.float32, device=device)
    wp_out = wp.zeros(shape[:-1], dtype=wp.float32, device=device)

    tiled_op = TiledSqNorm(shape, device=device)
    tiled_op.compute(wp_a, wp_out)

    assert np.allclose(wp_out.numpy(), np_out, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (128, 200),  # 2D (matches linesearch typical case)
        (10, 20, 64),  # 3D
        (256,),  # 1D
    ],
)
def test_tiled_argmin(shape):
    device = "cuda" if wp.is_cuda_available() else "cpu"

    # Generate random data
    np_a = np.random.rand(*shape).astype(np.float32)

    # Expected results (argmin and min along last axis)
    np_argmin = np.argmin(np_a, axis=-1).astype(np.int32)
    np_min = np.min(np_a, axis=-1).astype(np.float32)

    # Warp Arrays
    wp_a = wp.from_numpy(np_a, dtype=wp.float32, device=device)

    out_shape = shape[:-1]
    if len(shape) == 1:
        out_shape = (1,)

    wp_idx = wp.zeros(out_shape, dtype=wp.int32, device=device)
    wp_val = wp.zeros(out_shape, dtype=wp.float32, device=device)

    tiled_op = TiledArgMin(shape, tile_size=shape[-1], device=device)
    tiled_op.compute(wp_a, wp_idx, wp_val)

    # Verify
    if len(shape) == 1:
        assert wp_idx.numpy()[0] == np_argmin
        assert np.allclose(wp_val.numpy()[0], np_min, atol=1e-4)
    else:
        assert np.array_equal(wp_idx.numpy(), np_argmin)
        assert np.allclose(wp_val.numpy(), np_min, atol=1e-4)


if __name__ == "__main__":
    # Manual run if needed
    test_tiled_sum((128, 256))
    test_tiled_argmin((128, 200))
    print("Tests passed!")

