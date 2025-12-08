import gc
import time

import axion.optim.A_matvec as k_f32
import axion.optim.A_matvec_f16 as k_f16
import numpy as np
import warp as wp

wp.init()


def measure_graph_throughput(launch_lambda, graph_batch_size=50, measure_launches=10):
    launch_lambda()
    wp.synchronize()

    with wp.ScopedCapture() as cap:
        for _ in range(graph_batch_size):
            launch_lambda()
    graph = cap.graph

    wp.synchronize()
    t0 = time.time()
    for _ in range(measure_launches):
        wp.capture_launch(graph)
    wp.synchronize()
    t1 = time.time()

    return ((t1 - t0) / (graph_batch_size * measure_launches)) * 1000.0


def run_benchmark(num_bodies):
    j_per_body = 16
    c_per_body = 32
    block_size = 8

    num_joints = int(num_bodies * (j_per_body / 2.0 * 0.9))
    num_contacts = num_bodies * c_per_body

    # ---------------------------
    # DATA
    # ---------------------------
    M_inv = wp.array(np.random.rand(num_bodies, 6, 6).astype(np.float32), dtype=wp.spatial_matrix)
    map_j_np = k_f32.generate_safe_joint_map(num_joints, num_bodies, j_per_body)
    J_j_flat_np, body_to_flat_np = k_f32.build_joint_graph(
        num_bodies, j_per_body, np.zeros((num_joints, 2, 6)), map_j_np
    )

    J_j_flat = wp.array(
        np.random.randn(len(J_j_flat_np), 6).astype(np.float32), dtype=wp.spatial_vector
    )
    map_j = wp.array(map_j_np, dtype=wp.int32)
    map_struct = wp.array(body_to_flat_np, dtype=wp.int32, ndim=2)
    J_c = wp.array(np.random.randn(num_contacts, 6).astype(np.float32), dtype=wp.spatial_vector)

    v_body = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # ---------------------------
    # SETUP F32 (Baseline)
    # ---------------------------
    x_j_32 = wp.array(np.random.randn(num_joints).astype(np.float32), dtype=wp.float32)
    x_c_32 = wp.array(np.random.randn(num_contacts).astype(np.float32), dtype=wp.float32)
    z_j_32 = wp.zeros_like(x_j_32)
    z_c_32 = wp.zeros_like(x_c_32)
    C_j_32 = wp.zeros_like(x_j_32)
    C_c_32 = wp.zeros_like(x_c_32)

    # Precompute F32 (Linear Layout)
    W_j_32 = wp.zeros(num_bodies * j_per_body, dtype=wp.spatial_vector)
    x_idx_32 = wp.zeros(num_bodies * j_per_body, dtype=wp.int32)
    W_c_32 = wp.zeros(num_contacts, dtype=wp.spatial_vector)

    wp.launch(
        k_f32.kernel_bake_joints_flat,
        dim=num_bodies,
        inputs=[J_j_flat, map_struct, M_inv],
        outputs=[W_j_32, x_idx_32, j_per_body],
    )
    wp.launch(
        k_f32.kernel_bake_contacts_flat,
        dim=num_bodies,
        inputs=[J_c, M_inv],
        outputs=[W_c_32, c_per_body],
    )

    k_gather_32 = k_f32.create_body_gather_kernel(block_size, j_per_body, c_per_body)
    k_apply_32 = k_f32.create_apply_contacts_kernel(c_per_body)

    # ---------------------------
    # SETUP F16 (SoA Optimized)
    # ---------------------------
    x_j_16 = wp.array(x_j_32.numpy().astype(np.float16), dtype=wp.float16)
    x_c_16 = wp.array(x_c_32.numpy().astype(np.float16), dtype=wp.float16)
    z_j_16 = wp.zeros_like(x_j_16)
    z_c_16 = wp.zeros_like(x_c_16)
    C_j_16 = wp.zeros_like(x_j_16)
    C_c_16 = wp.zeros_like(x_c_16)

    # Precompute F16 (SoA / Transposed Layout)
    # Shape: [Stride, Bodies, 6]
    W_j_16 = wp.zeros((j_per_body, num_bodies, 6), dtype=wp.float16)
    x_idx_16 = wp.zeros((j_per_body, num_bodies), dtype=wp.int32)
    W_c_16 = wp.zeros((c_per_body, num_bodies, 6), dtype=wp.float16)

    wp.launch(
        k_f16.kernel_bake_joints_f16_soa,
        dim=num_bodies,
        inputs=[J_j_flat, map_struct, M_inv],
        outputs=[W_j_16, x_idx_16, j_per_body],
    )
    wp.launch(
        k_f16.kernel_bake_contacts_f16_soa,
        dim=num_bodies,
        inputs=[J_c, M_inv],
        outputs=[W_c_16, c_per_body],
    )

    k_gather_16 = k_f16.create_gather_f16_soa_kernel(j_per_body, c_per_body)
    k_apply_16 = k_f16.create_apply_contacts_f16_kernel(c_per_body)

    # ---------------------------
    # RUNNERS
    # ---------------------------
    def run_f32():
        wp.launch_tiled(
            k_gather_32,
            dim=[num_bodies // block_size],
            inputs=[x_j_32, W_j_32, x_idx_32, x_c_32, W_c_32],
            outputs=[v_body],
            block_dim=block_size * max(j_per_body, c_per_body),
        )
        wp.launch(
            k_f32.kernel_apply_joints,
            dim=num_joints,
            inputs=[v_body, J_j_flat, map_j, C_j_32, x_j_32, z_j_32],
        )
        wp.launch(k_apply_32, dim=num_contacts, inputs=[v_body, J_c, C_c_32, x_c_32, z_c_32])

    def run_f16():
        # F16 uses SoA gather (one thread per body)
        wp.launch(
            k_gather_16, dim=num_bodies, inputs=[x_j_16, W_j_16, x_idx_16, x_c_16, W_c_16, v_body]
        )
        wp.launch(
            k_f16.kernel_apply_joints_f16,
            dim=num_joints,
            inputs=[v_body, J_j_flat, map_j, C_j_16, x_j_16, z_j_16],
        )
        wp.launch(k_apply_16, dim=num_contacts, inputs=[v_body, J_c, C_c_16, x_c_16, z_c_16])

    t_32 = measure_graph_throughput(run_f32)
    t_16 = measure_graph_throughput(run_f16)

    return t_32, t_16


def main():
    sizes = [1024, 2048, 4096, 8192]
    print(f"{'Bodies':<10} | {'Float32 (ms)':<15} | {'Float16 (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)
    for nb in sizes:
        gc.collect()
        t32, t16 = run_benchmark(nb)
        print(f"{nb:<10} | {t32:<15.4f} | {t16:<15.4f} | {t32/t16:<9.2f}x")


if __name__ == "__main__":
    main()
