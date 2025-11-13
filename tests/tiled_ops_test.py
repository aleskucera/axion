from tqdm import tqdm
import warp as wp
import numpy as np
import src.axion.core.tiled_utils as tu

np.random.seed(0)
TS = tu.TILE_SIZE
Ns = [1, 10, TS - 1, TS, TS + 1, 10 * TS, 100 * TS, 100 * TS - 1, 100 * TS + 1]
Ms = [1, 3, 10]

for N in tqdm(Ns):
    for i in range(30):
        for M in Ms:
            a_np = np.random.rand(M, N).astype(np.float32) * 2 - 1
            b_np = np.random.rand(M, N).astype(np.float32) * 2 - 1
            a = wp.array(a_np, dtype=wp.float32, device="cuda:0")
            b = wp.array(b_np, dtype=wp.float32, device="cuda:0")

            res_tiled = tu.tiled_sq_norm_2d(a, device="cuda:0").numpy()
            res_np = np.sum(a_np * a_np, axis=1)
            assert np.allclose(
                res_tiled, res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

            res_tiled = tu.tiled_dot_2d(a, b, device="cuda:0").numpy()
            res_np = np.sum(a_np * b_np, axis=1)
            assert np.allclose(
                res_tiled, res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

            res_tiled = tu.tiled_sum_2d(a, device="cuda:0").numpy()
            res_np = np.sum(a_np, axis=1)
            assert np.allclose(
                res_tiled, res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

        a_np = np.random.rand(N).astype(np.float32) * 2 - 1
        b_np = np.random.rand(N).astype(np.float32) * 2 - 1
        a = wp.array(a_np, dtype=wp.float32, device="cuda:0")
        b = wp.array(b_np, dtype=wp.float32, device="cuda:0")

        res_tiled = tu.tiled_sq_norm_1d(a, device="cuda:0").numpy()[0]
        res_np = np.sum(a_np * a_np)
        assert np.isclose(
            res_tiled, res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

        res_tiled = tu.tiled_dot_1d(a, b, device="cuda:0").numpy()[0]
        res_np = np.sum(a_np * b_np)
        assert np.isclose(
            res_tiled, res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

        res_tiled = tu.tiled_sum_1d(a, device="cuda:0").numpy()[0]
        res_np = np.sum(a_np)
        assert np.isclose(
            res_tiled, res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

print("All tests passed!")
