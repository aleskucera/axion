import gc
import time

import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 1. KERNEL DEFINITIONS (Standard & Optimized)
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


# --- Pre-computation Kernel ---
@wp.kernel
def kernel_precompute_W(
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W_flat: wp.array(dtype=wp.spatial_vector),
):
    flat_idx = wp.tid()
    if flat_idx == 0:
        W_flat[0] = wp.vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

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
    return wp.rshift(J_idx + 1, 1)


def create_solver_tiled_optimized(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_solve_tiled_opt(
        lambda_padded: wp.array(dtype=wp.float32),
        W_flat: wp.array(dtype=wp.spatial_vector),
        body_to_constraints: wp.array(dtype=wp.int32),
        du: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        body_offset = tile_idx * bodies_in_tile
        map_offset = tile_idx * tile_size

        J_indices = wp.tile_load(
            body_to_constraints, shape=(tile_size,), offset=(map_offset,), storage="shared"
        )
        lam_indices = wp.tile_map(compute_lambda_idx_aligned, J_indices)

        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        dv_tile = wp.tile_map(wp.mul, W_tile, lam_tile)
        dv_by_body = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))
        total_dv = wp.tile_reduce(wp.add, dv_by_body, axis=1)

        wp.tile_store(du, total_dv, offset=body_offset)

    return kernel_solve_tiled_opt


# =================================================================================
# 2. DATA GENERATION
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
        # Retry logic for dense graphs
        attempts = 0
        while attempts < 10:
            avail = np.where(counts < max_c)[0]
            if len(avail) < 2:
                break
            v1 = rng.choice(avail)
            avail2 = avail[avail != v1]
            if len(avail2) == 0:
                break
            v2 = rng.choice(avail2)
            arr[i] = [v1, v2]
            counts[[v1, v2]] += 1
            break
            attempts += 1
    return arr


def build_tiled_graph_data(nb, max_c, J_np, constr_map_np):
    nc = len(J_np)
    J_flat_np = np.zeros((1 + nc * 2, 6), dtype=np.float32)
    J_flat_np[1:] = J_np.reshape(-1, 6)

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
# 3. BENCHMARK UTILITIES
# =================================================================================


def run_test_case(num_bodies, constraints_per_body, bodies_in_tile):
    num_constraints = num_bodies * 4  # Assuming avg degree 4

    # --- Generate Data ---
    M_inv_np = generate_M_inv(num_bodies)
    dlambda_np = np.random.randn(num_constraints).astype(np.float32)
    J_np = np.random.randn(num_constraints, 2, 6).astype(np.float32)
    constr_map_np = generate_constr_to_body(num_constraints, num_bodies, constraints_per_body)

    # --- Setup Scatter ---
    scatter_inputs = [
        wp.array(dlambda_np, dtype=wp.float32),
        wp.array(J_np, dtype=wp.spatial_vector),
        wp.array(constr_map_np, dtype=wp.int32),
        wp.array(M_inv_np, dtype=wp.spatial_matrix),
        wp.zeros(num_bodies, dtype=wp.spatial_vector),
    ]

    # --- Setup Tiled ---
    J_flat_np, body_to_J_np = build_tiled_graph_data(
        num_bodies, constraints_per_body, J_np, constr_map_np
    )
    lam_padded = np.zeros(num_constraints + 1, dtype=np.float32)
    lam_padded[1:] = dlambda_np

    lambda_wp = wp.array(lam_padded, dtype=wp.float32)
    J_flat_wp = wp.array(J_flat_np, dtype=wp.spatial_vector)
    body_to_J_wp = wp.array(body_to_J_np, dtype=wp.int32)
    W_flat_wp = wp.zeros_like(J_flat_wp)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # Pre-calc W
    wp.launch(
        kernel_precompute_W,
        dim=len(J_flat_np),
        inputs=[J_flat_wp, scatter_inputs[2], scatter_inputs[3]],
        outputs=[W_flat_wp],
    )
    wp.synchronize()

    tiled_kernel = create_solver_tiled_optimized(bodies_in_tile, constraints_per_body)
    tiled_inputs = [lambda_wp, W_flat_wp, body_to_J_wp, du_tiled]

    # --- MEASUREMENT FUNCTION ---
    def measure(kernel, dims, inputs, is_tiled=False, steps=20):
        # Warmup
        if is_tiled:
            wp.launch_tiled(
                kernel,
                dim=[dims],
                inputs=inputs,
                block_dim=bodies_in_tile * constraints_per_body,
            )
        else:
            wp.launch(kernel, dim=dims, inputs=inputs)
        wp.synchronize()

        # Capture Graph
        with wp.ScopedCapture() as cap:
            for _ in range(5):
                if is_tiled:
                    wp.launch_tiled(
                        kernel,
                        dim=[dims],
                        inputs=inputs,
                        block_dim=bodies_in_tile * constraints_per_body,
                    )
                else:
                    inputs[-1].zero_()  # Reset output
                    wp.launch(kernel, dim=dims, inputs=inputs)

        graph = cap.graph
        t0 = time.time()
        for _ in range(steps):
            wp.capture_launch(graph)
        wp.synchronize()
        t1 = time.time()

        return ((t1 - t0) / (steps * 5)) * 1000.0  # Return ms

    # --- Run Measurements ---
    t_scatter = measure(
        kernel_M_inv_Jjt_matvec_scatter, num_constraints, scatter_inputs, is_tiled=False
    )
    t_tiled = measure(tiled_kernel, num_bodies // bodies_in_tile, tiled_inputs, is_tiled=True)

    return t_scatter, t_tiled


# =================================================================================
# 4. MAIN LOOP
# =================================================================================


def main():
    bodies_list = [256, 512, 1024, 2048, 4096, 8192, 16384]  # Scaling up

    constraints_per_body = 16
    bodies_in_tile = 8  # optimized for warp size

    print(f"{'# BODIES':<12} | {'SCATTER (ms)':<15} | {'TILED (ms)':<15} | {'SPEEDUP':<10}")
    print("-" * 65)

    for n_bodies in bodies_list:
        try:
            # Force cleanup to prevent VRAM accumulation
            gc.collect()

            t_s, t_t = run_test_case(n_bodies, constraints_per_body, bodies_in_tile)

            speedup = t_s / t_t
            print(f"{n_bodies:<12} | {t_s:<15.4f} | {t_t:<15.4f} | {speedup:<10.2f}x")

        except Exception as e:
            print(f"{n_bodies:<12} | {'FAILED':<15} | {str(e)}")


if __name__ == "__main__":
    main()
