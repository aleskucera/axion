import gc
import time

import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 1. NAIVE TILED KERNELS (Baseline)
# =================================================================================


@wp.func
def compute_joint_lambda_idx(J_idx: int):
    return wp.rshift(J_idx + 1, 1)


def create_naive_tiled_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_joints_naive(
        lambda_padded: wp.array(dtype=wp.float32),
        W_flat: wp.array(dtype=wp.spatial_vector),
        body_to_constraints: wp.array(dtype=wp.int32),
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        tile_offset = tile_idx * tile_size

        # Gather Indices (Random Access possible)
        J_indices = wp.tile_load(
            body_to_constraints, shape=(tile_size,), offset=(tile_offset,), storage="shared"
        )

        lam_indices = wp.tile_map(compute_joint_lambda_idx, J_indices)

        # Gather Data (INDIRECT LOAD)
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        # Compute
        u_tile = wp.tile_map(wp.mul, W_tile, lam_tile)
        u_matrix = wp.tile_reshape(u_tile, shape=(bodies_in_tile, constraints_per_body))
        u_sum = wp.tile_reduce(wp.add, u_matrix, axis=1)

        wp.tile_store(u, u_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_joints_naive


@wp.kernel
def kernel_precompute_joint_W_naive(
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W_flat: wp.array(dtype=wp.spatial_vector),
):
    flat_idx = wp.tid()
    if flat_idx == 0:
        return

    idx = flat_idx - 1
    c_idx = idx // 2
    side = idx % 2

    b_idx = constraint_to_body[c_idx, side]
    W_flat[flat_idx] = M_inv[b_idx] * J_flat[flat_idx]


# =================================================================================
# 2. OPTIMIZED TILED KERNELS (1D Flattened / Baked)
# =================================================================================


@wp.kernel
def kernel_bake_joint_data_flat(
    # Inputs
    J_flat: wp.array(dtype=wp.spatial_vector),
    body_to_constraints: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # Outputs (1D Flattened)
    W_binned_flat: wp.array(dtype=wp.spatial_vector),
    lam_indices_binned_flat: wp.array(dtype=wp.int32),
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]

    # Manually compute flat offset base
    base_offset = body_idx * stride

    # Loop up to stride (constraints_per_body)
    for i in range(stride):
        flat_idx = body_to_constraints[body_idx, i]
        write_idx = base_offset + i

        if flat_idx > 0:
            # Bake W = M_inv * J
            W_binned_flat[write_idx] = m_inv_local * J_flat[flat_idx]
            # Pre-calc Lambda Index
            lam_indices_binned_flat[write_idx] = wp.rshift(flat_idx + 1, 1)
        else:
            # Padding
            W_binned_flat[write_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            lam_indices_binned_flat[write_idx] = 0


def create_optimized_tiled_solver_flat(bodies_in_tile, constraints_per_body):
    # Total elements in a tile = bodies * constraints per body
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_joints_optimized_flat(
        lambda_padded: wp.array(dtype=wp.float32),
        W_binned_flat: wp.array(dtype=wp.spatial_vector),  # 1D Input
        lam_indices_binned_flat: wp.array(dtype=wp.int32),  # 1D Input
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()

        # Since data is flattened (Body0_C0...C7, Body1_C0...C7),
        # A tile of 'bodies_in_tile' bodies is just a contiguous chunk of 'tile_size' elements.
        data_offset = tile_idx * tile_size

        # 1. Load Indices (DIRECT 1D LOAD)
        lam_idx_tile = wp.tile_load(
            lam_indices_binned_flat, shape=(tile_size,), offset=(data_offset,)
        )

        # 2. Load Vectors (DIRECT 1D LOAD - Coalesced)
        W_tile = wp.tile_load(W_binned_flat, shape=(tile_size,), offset=(data_offset,))

        # 3. Gather Lambda
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_idx_tile, shape=(tile_size,))

        # 4. Compute
        u_tile = wp.tile_map(wp.mul, W_tile, lam_tile)

        # Reshape the 1D tile back to (Bodies, Constraints) for reduction
        u_matrix = wp.tile_reshape(u_tile, shape=(bodies_in_tile, constraints_per_body))
        u_sum = wp.tile_reduce(wp.add, u_matrix, axis=1)

        wp.tile_store(u, u_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_joints_optimized_flat


# =================================================================================
# 3. HELPERS
# =================================================================================


def generate_safe_joint_map(num_joints, num_bodies, max_joints_per_body):
    rng = np.random.default_rng(42)
    map_j = np.zeros((num_joints, 2), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)
    available_bodies = list(range(num_bodies))

    for i in range(num_joints):
        if len(available_bodies) < 2:
            break
        b1, b2 = rng.choice(available_bodies, size=2, replace=False)
        map_j[i] = [b1, b2]
        counts[b1] += 1
        counts[b2] += 1
        if counts[b1] >= max_joints_per_body:
            available_bodies.remove(b1)
        if counts[b2] >= max_joints_per_body and b2 in available_bodies:
            available_bodies.remove(b2)
    return map_j


def build_joint_graph(num_bodies, max_constraints, J_np, map_np):
    num_cons = len(J_np)
    J_flat = np.zeros((1 + num_cons * 2, 6), dtype=np.float32)
    J_flat[1:] = J_np.reshape(-1, 6)
    body_to_flat = np.zeros((num_bodies, max_constraints), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)

    for c_i in range(num_cons):
        idx_A, idx_B = 1 + c_i * 2, 1 + c_i * 2 + 1
        b0, b1 = map_np[c_i]
        if counts[b0] < max_constraints:
            body_to_flat[b0, counts[b0]] = idx_A
            counts[b0] += 1
        if counts[b1] < max_constraints:
            body_to_flat[b1, counts[b1]] = idx_B
            counts[b1] += 1
    return J_flat, body_to_flat


def measure_throughput(launch_lambda, iterations=100):
    launch_lambda()
    wp.synchronize()

    with wp.ScopedCapture() as cap:
        for _ in range(10):
            launch_lambda()
    graph = cap.graph

    wp.synchronize()
    t0 = time.time()
    for _ in range(iterations // 10):
        wp.capture_launch(graph)
    wp.synchronize()
    t1 = time.time()

    return ((t1 - t0) / iterations) * 1000.0


# =================================================================================
# 4. MAIN BENCHMARK
# =================================================================================


def main():
    num_bodies = 16384
    j_per_body = 8
    tile_bodies = 8
    num_joints = int(num_bodies * (j_per_body / 2.0 * 0.9))

    print(f"Running Optimization Test: {num_bodies} Bodies, {num_joints} Joints")

    # 1. Data Gen
    M_inv_np = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T

    map_j_np = generate_safe_joint_map(num_joints, num_bodies, j_per_body)
    lam_j_np = np.random.randn(num_joints).astype(np.float32)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)

    # 2. Alloc
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)

    lam_j_pad_np = np.zeros(num_joints + 1, dtype=np.float32)
    lam_j_pad_np[1:] = lam_j_np
    lam_j = wp.array(lam_j_pad_np, dtype=wp.float32)

    J_j_flat_np, body_to_flat_np = build_joint_graph(num_bodies, j_per_body, J_j_np, map_j_np)

    # Arrays
    J_j_flat = wp.array(J_j_flat_np, dtype=wp.spatial_vector)

    # Explicit map for naive precompute (Fixes Illegal Address error)
    map_j_dev = wp.array(map_j_np, dtype=wp.int32)

    # Map for Naive Solver
    map_naive = wp.array(body_to_flat_np.flatten(), dtype=wp.int32)

    # Map for Baking Kernel (needs structure to read from)
    map_opt_struct = wp.array(body_to_flat_np, dtype=wp.int32, ndim=2)

    du_naive = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_opt = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # ---------------------------
    # SETUP NAIVE
    # ---------------------------
    W_naive = wp.zeros_like(J_j_flat)
    wp.launch(
        kernel_precompute_joint_W_naive,
        dim=len(J_j_flat),
        inputs=[J_j_flat, map_j_dev, M_inv],
        outputs=[W_naive],
    )
    k_naive = create_naive_tiled_solver(tile_bodies, j_per_body)

    # ---------------------------
    # SETUP OPTIMIZED (1D FLAT)
    # ---------------------------
    # Flattened Allocation: size = bodies * joints_per_body
    flat_size = num_bodies * j_per_body
    W_binned_flat = wp.zeros(flat_size, dtype=wp.spatial_vector)
    lam_idx_binned_flat = wp.zeros(flat_size, dtype=wp.int32)

    wp.launch(
        kernel_bake_joint_data_flat,
        dim=num_bodies,
        inputs=[J_j_flat, map_opt_struct, M_inv],
        outputs=[W_binned_flat, lam_idx_binned_flat, j_per_body],
    )

    k_opt_flat = create_optimized_tiled_solver_flat(tile_bodies, j_per_body)

    # ---------------------------
    # CORRECTNESS CHECK
    # ---------------------------
    wp.synchronize()

    wp.launch_tiled(
        k_naive,
        dim=[num_bodies // tile_bodies],
        inputs=[lam_j, W_naive, map_naive],
        outputs=[du_naive],
        block_dim=tile_bodies * j_per_body,
    )
    wp.launch_tiled(
        k_opt_flat,
        dim=[num_bodies // tile_bodies],
        inputs=[lam_j, W_binned_flat, lam_idx_binned_flat],
        outputs=[du_opt],
        block_dim=tile_bodies * j_per_body,
    )

    wp.synchronize()

    err = np.max(np.abs(du_naive.numpy() - du_opt.numpy()))
    print(f"\nCorrectness Check Error: {err:.6f}")

    if err > 1e-4:
        print("CRITICAL: Optimized kernel outputs different results!")
        return

    # ---------------------------
    # BENCHMARK
    # ---------------------------
    print("\nBenchmarking Solver Loop (ms per call)...")

    def run_naive():
        wp.launch_tiled(
            k_naive,
            dim=[num_bodies // tile_bodies],
            inputs=[lam_j, W_naive, map_naive],
            outputs=[du_naive],
            block_dim=tile_bodies * j_per_body,
        )

    def run_opt():
        # Notice: inputs are now 1D flat arrays
        wp.launch_tiled(
            k_opt_flat,
            dim=[num_bodies // tile_bodies],
            inputs=[lam_j, W_binned_flat, lam_idx_binned_flat],
            outputs=[du_opt],
            block_dim=tile_bodies * j_per_body,
        )

    t_naive = measure_throughput(run_naive)
    t_opt = measure_throughput(run_opt)

    print(f"{'Naive Tiled (Indirect)':<30} | {t_naive:.4f} ms")
    print(f"{'Optimized (Flat Coalesced)':<30} | {t_opt:.4f} ms")
    print("-" * 50)
    print(f"Speedup: {t_naive / t_opt:.2f}x")


if __name__ == "__main__":
    main()
