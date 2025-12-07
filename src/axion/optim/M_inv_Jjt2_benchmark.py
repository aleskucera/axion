import time

import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 0. CONFIGURATION
# =================================================================================

NUM_BODIES = 1024 * 4  # Large count to saturate GPU
NUM_CONSTRAINTS = NUM_BODIES * 4
CONSTRAINTS_PER_BODY = 16
BODIES_IN_TILE = 8
TILE_SIZE = BODIES_IN_TILE * CONSTRAINTS_PER_BODY

assert NUM_BODIES % BODIES_IN_TILE == 0

# =================================================================================
# 1. KERNELS DEFINITIONS
# =================================================================================


# --- Reference: Scatter Kernel ---
@wp.kernel
def kernel_M_inv_Jjt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    b0_idx = constraint_to_body[c_idx, 0]
    b1_idx = constraint_to_body[c_idx, 1]

    lam = dlambda[c_idx]

    if b0_idx >= 0:
        delta_impulse0 = J[c_idx, 0] * lam
        wp.atomic_add(du, b0_idx, M_inv[b0_idx] * delta_impulse0)

    if b1_idx >= 0:
        delta_impulse1 = J[c_idx, 1] * lam
        wp.atomic_add(du, b1_idx, M_inv[b1_idx] * delta_impulse1)


# --- Pre-computation Kernel (Run once during setup) ---
@wp.kernel
def kernel_precompute_W(
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W_flat: wp.array(dtype=wp.spatial_vector),  # Output
):
    flat_idx = wp.tid()
    if flat_idx == 0:  # Padding
        W_flat[0] = wp.vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

    # Map flat index back to logic
    idx = flat_idx - 1
    c_idx = idx // 2
    side = idx % 2
    body_idx = constraint_to_body[c_idx, side]

    if body_idx >= 0:
        W_flat[flat_idx] = M_inv[body_idx] * J_flat[flat_idx]
    else:
        W_flat[flat_idx] = wp.vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --- Optimized: Tiled Kernel ---
@wp.func
def compute_lambda_idx_aligned(J_idx: int):
    # Shift logic: (J_idx + 1) // 2
    return wp.rshift(J_idx + 1, 1)


def create_solver_tiled_optimized(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_solve_tiled_opt(
        lambda_padded: wp.array(dtype=wp.float32),
        W_flat: wp.array(dtype=wp.spatial_vector),  # Pre-computed
        body_to_constraints: wp.array(dtype=wp.int32),
        du: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        body_offset = tile_idx * bodies_in_tile
        map_offset = tile_idx * tile_size

        # 1. Load Topology
        J_indices = wp.tile_load(
            body_to_constraints, shape=(tile_size,), offset=(map_offset,), storage="shared"
        )

        # 2. Compute indices (Math)
        lam_indices = wp.tile_map(compute_lambda_idx_aligned, J_indices)

        # 3. Load Data (W instead of J + M_inv)
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        # 4. Math (Vector only)
        dv_tile = wp.tile_map(wp.mul, W_tile, lam_tile)

        # 5. Reduce
        dv_by_body = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))
        total_dv = wp.tile_reduce(wp.add, dv_by_body, axis=1)

        # 6. Store
        wp.tile_store(du, total_dv, offset=body_offset)

    return kernel_solve_tiled_opt


# =================================================================================
# 2. DATA GENERATION HELPERS
# =================================================================================


def generate_M_inv(n):
    m = np.random.rand(n, 6, 6).astype(np.float32)
    for i in range(n):
        m[i] = m[i] @ m[i].T
    return m


def generate_constr_to_body(nc, nb, max_c, seed=42):
    rng = np.random.default_rng(seed)
    counts = np.zeros(nb, dtype=np.int32)
    arr = np.zeros((nc, 2), dtype=np.int32)
    for i in range(nc):
        avail = np.where(counts < max_c)[0]
        if len(avail) < 2:
            raise ValueError("Topology fail")
        v1 = rng.choice(avail)
        avail2 = avail[avail != v1]
        v2 = rng.choice(avail2)
        arr[i] = [v1, v2]
        counts[[v1, v2]] += 1
    return arr


def build_tiled_graph_data(nb, max_c, J_np, constr_map_np):
    nc = len(J_np)
    # Flatten J
    J_flat_np = np.zeros((1 + nc * 2, 6), dtype=np.float32)
    J_flat_np[1:] = J_np.reshape(-1, 6)

    # Build Map
    body_to_flat = np.zeros((nb, max_c), dtype=np.int32)
    body_counts = np.zeros(nb, dtype=np.int32)

    for c_i in range(nc):
        idx_A, idx_B = 1 + c_i * 2, 1 + c_i * 2 + 1
        b0, b1 = constr_map_np[c_i]

        if b0 >= 0 and body_counts[b0] < max_c:
            body_to_flat[b0, body_counts[b0]] = idx_A
            body_counts[b0] += 1
        if b1 >= 0 and body_counts[b1] < max_c:
            body_to_flat[b1, body_counts[b1]] = idx_B
            body_counts[b1] += 1

    return J_flat_np, body_to_flat.flatten()


# =================================================================================
# 3. BENCHMARK SETUP
# =================================================================================


def build_data():
    # A. Raw Data
    M_inv_np = generate_M_inv(NUM_BODIES)
    dlambda_np = np.random.randn(NUM_CONSTRAINTS).astype(np.float32)
    J_np = np.random.randn(NUM_CONSTRAINTS, 2, 6).astype(np.float32)
    constr_map_np = generate_constr_to_body(NUM_CONSTRAINTS, NUM_BODIES, CONSTRAINTS_PER_BODY)

    # B. Scatter Inputs
    scatter_inputs = [
        wp.array(dlambda_np, dtype=wp.float32),
        wp.array(J_np, dtype=wp.spatial_vector),
        wp.array(constr_map_np, dtype=wp.int32),
        wp.array(M_inv_np, dtype=wp.spatial_matrix),
        wp.zeros(NUM_BODIES, dtype=wp.spatial_vector),
    ]

    # C. Tiled Inputs (Optimized)
    J_flat_np, body_to_J_np = build_tiled_graph_data(
        NUM_BODIES, CONSTRAINTS_PER_BODY, J_np, constr_map_np
    )

    # 1. Pad Lambda (N+1)
    lam_padded = np.zeros(NUM_CONSTRAINTS + 1, dtype=np.float32)
    lam_padded[1:] = dlambda_np

    # 2. Warp Arrays
    lambda_wp = wp.array(lam_padded, dtype=wp.float32)
    J_flat_wp = wp.array(J_flat_np, dtype=wp.spatial_vector)
    body_to_J_wp = wp.array(body_to_J_np, dtype=wp.int32)
    W_flat_wp = wp.zeros_like(J_flat_wp)  # Container for Pre-calc
    du_tiled = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)

    # 3. PRE-COMPUTE W (Run Once)
    wp.launch(
        kernel_precompute_W,
        dim=len(J_flat_np),
        inputs=[J_flat_wp, scatter_inputs[2], scatter_inputs[3]],  # J_flat, map, M_inv
        outputs=[W_flat_wp],
    )
    wp.synchronize()

    # 4. Create Kernel
    tiled_kernel = create_solver_tiled_optimized(BODIES_IN_TILE, CONSTRAINTS_PER_BODY)

    # Inputs: [lambda, W_flat, body_map, output]
    # Note: M_inv and J are NOT passed here.
    tiled_inputs = [lambda_wp, W_flat_wp, body_to_J_wp, du_tiled]

    return scatter_inputs, tiled_inputs, tiled_kernel


# =================================================================================
# 4. TIMING LOGIC
# =================================================================================


def measure_scatter(inputs, use_graph=False, steps=100):
    dl, J, cmap, Minv, du = inputs

    def run():
        du.zero_()
        wp.launch(
            kernel_M_inv_Jjt_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[dl, J, cmap, Minv],
            outputs=[du],
        )

    # Warmup
    run()
    wp.synchronize()

    t0 = time.time()

    if use_graph:
        with wp.ScopedCapture() as cap:
            for _ in range(10):
                run()  # Graph batch
        graph = cap.graph
        for _ in range(steps):
            wp.capture_launch(graph)
        wp.synchronize()
        total_ops = steps * 10
    else:
        for _ in range(steps):
            run()
        wp.synchronize()
        total_ops = steps

    t1 = time.time()
    return (t1 - t0) / total_ops


def measure_tiled(inputs, kernel, use_graph=False, steps=100):
    lam, W, bmap, du = inputs
    num_tiles = NUM_BODIES // BODIES_IN_TILE

    def run():
        du.zero_()
        wp.launch_tiled(
            kernel, dim=[num_tiles], inputs=[lam, W, bmap], outputs=[du], block_dim=TILE_SIZE
        )

    # Warmup
    run()
    wp.synchronize()

    t0 = time.time()

    if use_graph:
        with wp.ScopedCapture() as cap:
            for _ in range(10):
                run()
        graph = cap.graph
        for _ in range(steps):
            wp.capture_launch(graph)
        wp.synchronize()
        total_ops = steps * 10
    else:
        for _ in range(steps):
            run()
        wp.synchronize()
        total_ops = steps

    t1 = time.time()
    return (t1 - t0) / total_ops


# =================================================================================
# 5. MAIN
# =================================================================================


def main():
    print(f"\n=== Benchmark: Scatter vs. Optimized Tiled (Pre-calc W) ===")
    print(f"Bodies: {NUM_BODIES}, Constraints: {NUM_CONSTRAINTS}")
    print(f"Tile Config: {BODIES_IN_TILE} bodies/tile, {CONSTRAINTS_PER_BODY} constr/body")

    scatter_in, tiled_in, tiled_kern = build_data()

    # --- Correctness ---
    # Run scatter
    wp.launch(
        kernel_M_inv_Jjt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=scatter_in[:-1],
        outputs=[scatter_in[-1]],
    )
    # Run tiled
    num_tiles = NUM_BODIES // BODIES_IN_TILE
    wp.launch_tiled(
        tiled_kern,
        dim=[num_tiles],
        inputs=tiled_in[:-1],
        outputs=[tiled_in[-1]],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    diff = np.max(np.abs(scatter_in[-1].numpy() - tiled_in[-1].numpy()))
    print(f"Validation Diff: {diff:.6e} (Should be small/epsilon)")
    if diff > 1e-3:
        print("WARNING: Mismatch detected!")
        return

    # --- Timing ---
    print("\n[ Standard Launch (Python Overhead included) ]")
    t_scat = measure_scatter(scatter_in, False) * 1e3
    t_tile = measure_tiled(tiled_in, tiled_kern, False) * 1e3
    print(f"Scatter: {t_scat:.4f} ms")
    print(f"Tiled:   {t_tile:.4f} ms  (Speedup: {t_scat/t_tile:.2f}x)")

    print("\n[ CUDA Graph Launch (Pure GPU Performance) ]")
    t_scat_g = measure_scatter(scatter_in, True) * 1e3
    t_tile_g = measure_tiled(tiled_in, tiled_kern, True) * 1e3
    print(f"Scatter: {t_scat_g:.4f} ms")
    print(f"Tiled:   {t_tile_g:.4f} ms  (Speedup: {t_scat_g/t_tile_g:.2f}x)")


if __name__ == "__main__":
    main()
