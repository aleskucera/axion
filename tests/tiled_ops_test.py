from tqdm import tqdm
import warp as wp
import numpy as np
import axion.tiled.tiled_utils as tu

TS = 1024
np.random.seed(0)
Ns = [1, 10, TS - 1, TS, TS + 1, 10 * TS, 100 * TS, 100 * TS - 1, 100 * TS + 1]
Ms = [1, 3, 10]
TESTS = 30

for i in tqdm(range(TESTS)):
    for N in Ns:
        for M in Ms:
            tiled_dot = tu.TiledDot((M, N), TS)
            tiled_norm = tu.TiledSqrNorm((M, N), TS)
            tiled_sum = tu.TiledSum((M, N), TS)

            a_np = np.random.rand(M, N).astype(np.float32) * 2 - 1
            b_np = np.random.rand(M, N).astype(np.float32) * 2 - 1
            a_wp = wp.array(a_np, dtype=wp.float32, device="cuda:0")
            b_wp = wp.array(b_np, dtype=wp.float32, device="cuda:0")
            res_tiled = wp.zeros(M, dtype=wp.float32, device="cuda:0")

            tiled_norm.compute(a_wp, res_tiled)
            res_np = np.sum(a_np * a_np, axis=1)
            assert np.allclose(
                res_tiled.numpy(), res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

            tiled_dot.compute(a_wp, b_wp, res_tiled)
            res_np = np.sum(a_np * b_np, axis=1)
            assert np.allclose(
                res_tiled.numpy(), res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

            tiled_sum.compute(a_wp, res_tiled)
            res_np = np.sum(a_np, axis=1)
            assert np.allclose(
                res_tiled.numpy(), res_np, atol=1e-3
            ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, M={M}, N={N}"

        tiled_dot = tu.TiledDot(N, TS)
        tiled_norm = tu.TiledSqrNorm(N, TS)
        tiled_sum = tu.TiledSum(N, TS)

        a_np = np.random.rand(N).astype(np.float32) * 2 - 1
        b_np = np.random.rand(N).astype(np.float32) * 2 - 1
        a_wp = wp.array(a_np, dtype=wp.float32, device="cuda:0")
        b_wp = wp.array(b_np, dtype=wp.float32, device="cuda:0")
        res_tiled = wp.zeros(1, dtype=wp.float32, device="cuda:0")

        tiled_norm.compute(a_wp, res_tiled)
        res_np = np.sum(a_np * a_np)
        assert np.isclose(
            res_tiled.numpy()[0], res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

        tiled_dot.compute(a_wp, b_wp, res_tiled)
        res_np = np.sum(a_np * b_np)
        assert np.isclose(
            res_tiled.numpy()[0], res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

        tiled_sum.compute(a_wp, res_tiled)
        res_np = np.sum(a_np)
        assert np.isclose(
            res_tiled.numpy()[0], res_np, atol=1e-3
        ), f"\nMismatch:\nTiled: {res_tiled}\nNumpy: {res_np}\nat: iteration {i}, N={N}"

print("All tests passed!")
