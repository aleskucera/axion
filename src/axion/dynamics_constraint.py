import time

import numpy as np
import warp as wp
from warp.sim import ModelShapeGeometry


@wp.kernel
def unconstrained_dynamics_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    body_f: wp.array(dtype=wp.spatial_vector),  # [B]
    body_mass: wp.array(dtype=wp.float32),  # [B]
    body_mass_inv: wp.array(dtype=wp.float32),  # [B]
    body_inertia: wp.array(dtype=wp.mat33),  # [B]
    body_inertia_inv: wp.array(dtype=wp.mat33),  # [B]
    # --- Parameters ---
    dt: wp.float32,
    gravity: wp.vec3,  # [3]
    # --- Outputs ---
    g: wp.array(dtype=wp.float32),  # [B]
    H_inv_values: wp.array(dtype=wp.float32),  # [12B]
):
    tid = wp.tid()
    if tid >= body_qd.shape[0]:
        return

    w = wp.spatial_top(body_qd[tid])
    v = wp.spatial_bottom(body_qd[tid])
    w_prev = wp.spatial_top(body_qd_prev[tid])
    v_prev = wp.spatial_bottom(body_qd_prev[tid])
    t = wp.spatial_top(body_f[tid])
    f = wp.spatial_bottom(body_f[tid])

    m, I = body_mass[tid], body_inertia[tid]

    res_ang = I * (w - w_prev) - t * dt
    res_lin = m * (v - v_prev) - f * dt - m * gravity * dt

    # --- g --- [6B]
    g_b = wp.spatial_vector(res_ang, res_lin)
    for i in range(wp.static(6)):
        st_i = wp.static(i)
        g[tid * 6 + st_i] = g_b[st_i]

    m_inv, I_inv = body_mass_inv[tid], body_inertia_inv[tid]

    # --- H_inv --- [6B, 6B]
    # Angular part:
    for r in range(wp.static(3)):  # rows
        for c in range(wp.static(3)):  # columns
            st_r, st_c = wp.static(r), wp.static(c)
            flat_idx = st_r * 3 + st_c
            H_inv_values[tid * 12 + flat_idx] = I_inv[st_r, st_c]

    # Linear part:
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        flat_idx = 9 + st_i
        H_inv_values[tid * 12 + flat_idx] = m_inv


def setup_data(num_bodies, device):
    """Generates all necessary input and output arrays for the kernel."""
    B = num_bodies

    # Generate random inertia tensors that are symmetric positive-definite
    rand_mat = np.random.rand(B, 3, 3)
    # Make symmetric
    inertia_tensors = (rand_mat + rand_mat.transpose((0, 2, 1))) / 2.0
    # Add identity to ensure positive-definiteness for inversion
    inertia_tensors += np.expand_dims(np.identity(3), axis=0)
    inertia_inv_tensors = np.linalg.inv(inertia_tensors)

    mass = np.random.rand(B).astype(np.float32) + 1.0
    mass_inv = 1.0 / mass

    data = {
        # --- Inputs ---
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_f": wp.array(np.zeros((B, 6)), dtype=wp.spatial_vector, device=device),
        "body_mass": wp.array(mass, dtype=wp.float32, device=device),
        "body_mass_inv": wp.array(mass_inv, dtype=wp.float32, device=device),
        "body_inertia": wp.array(inertia_tensors, dtype=wp.mat33, device=device),
        "body_inertia_inv": wp.array(
            inertia_inv_tensors, dtype=wp.mat33, device=device
        ),
        # --- Parameters ---
        "params": {
            "dt": 1.0 / 60.0,
            "gravity": wp.vec3(0.0, -9.8, 0.0),
        },
        # --- Outputs ---
        "g": wp.zeros(B * 6, dtype=wp.float32, device=device),
        "H_inv_values": wp.zeros(B * 12, dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(num_bodies, num_iterations=200):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Unconstrained Dynamics Kernel: B={num_bodies}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, device)
    params = data["params"]

    # Assemble the list of arguments in the exact order the kernel expects.
    # This includes all inputs, parameters, and outputs.
    kernel_args = [
        data["body_qd"],
        data["body_qd_prev"],
        data["body_f"],
        data["body_mass"],
        data["body_mass_inv"],
        data["body_inertia"],
        data["body_inertia_inv"],
        params["dt"],
        params["gravity"],
        data["g"],
        data["H_inv_values"],
    ]

    # --- 1. Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    # Warm-up launch
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=num_bodies,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark (only on GPU) ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        with wp.ScopedCapture() as capture:
            wp.launch(
                kernel=unconstrained_dynamics_kernel,
                dim=num_bodies,
                inputs=kernel_args,
                device=device,
            )
        graph = capture.graph
        # Warm-up launch
        wp.capture_launch(graph)
        wp.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            wp.capture_launch(graph)
        wp.synchronize()
        graph_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000

        print(f"   Avg. Time: {graph_launch_ms:.3f} ms")
        if graph_launch_ms > 0:
            speedup = standard_launch_ms / graph_launch_ms
            print(f"   Speedup from Graph: {speedup:.2f}x")
    else:
        print("2. Skipping CUDA Graph benchmark (not on a GPU device).")

    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print("Warning: No CUDA device found. Performance will be poor.")

    run_benchmark(num_bodies=100)
    run_benchmark(num_bodies=500)
    run_benchmark(num_bodies=1000)
    run_benchmark(num_bodies=2000)
    run_benchmark(num_bodies=4000)
