import time

import numpy as np
import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.types import ContactInteraction
from axion.types import SpatialInertia


def setup_data(
    num_worlds,
    num_bodies_per_world,
    num_contacts_per_world,
    device,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Generates random batched input data for the kernel benchmark."""

    N_w = num_worlds
    N_b = num_bodies_per_world
    N_c = num_contacts_per_world

    np.random.seed(42)  # for reproducibility

    # --- Generate Penetration Depths ---
    penetration_depths = np.random.rand(N_w, N_c).astype(np.float32) * 0.1
    num_inactive = int(N_c * inactive_contact_ratio)
    if num_inactive > 0:
        inactive_indices = np.random.choice(N_c, num_inactive, replace=False)
        penetration_depths[:, inactive_indices] = -0.01  # Make them non-penetrating

    # --- Generate Body Indices ---
    body_a_indices = np.random.randint(0, N_b, size=(N_w, N_c), dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=(N_w, N_c), dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(0, N_b, size=np.sum(mask), dtype=np.int32)
        mask = body_a_indices == body_b_indices

    num_fixed = int(N_c * fixed_body_ratio)
    if num_fixed > 0:
        fixed_indices = np.random.choice(N_c, num_fixed, replace=False)
        chooser = np.random.randint(0, 2, size=num_fixed)
        body_a_indices[:, fixed_indices[chooser == 0]] = -1
        body_b_indices[:, fixed_indices[chooser == 1]] = -1

    # --- Generate Random Jacobians, coeffs, etc. ---
    J_a = (np.random.rand(N_w, N_c, 3, 6) - 0.5).astype(np.float32)
    J_b = (np.random.rand(N_w, N_c, 3, 6) - 0.5).astype(np.float32)
    restitution_coeffs = (np.random.rand(N_w, N_c) * 0.5).astype(np.float32)
    friction_coeffs = (np.random.rand(N_w, N_c) * 0.5 + 0.5).astype(np.float32)

    # --- Create Interactions Array ---
    interactions_list = []
    for w in range(N_w):
        for i in range(N_c):
            inter = ContactInteraction()
            inter.is_active = penetration_depths[w, i] > 0.0
            inter.body_a_idx = body_a_indices[w, i]
            inter.body_b_idx = body_b_indices[w, i]
            inter.penetration_depth = penetration_depths[w, i]
            inter.restitution_coeff = restitution_coeffs[w, i]
            inter.friction_coeff = friction_coeffs[w, i]

            inter.basis_a.normal = wp.spatial_vector(*J_a[w, i, 0])
            inter.basis_a.tangent1 = wp.spatial_vector(*J_a[w, i, 1])
            inter.basis_a.tangent2 = wp.spatial_vector(*J_a[w, i, 2])
            inter.basis_b.normal = wp.spatial_vector(*J_b[w, i, 0])
            inter.basis_b.tangent1 = wp.spatial_vector(*J_b[w, i, 1])
            inter.basis_b.tangent2 = wp.spatial_vector(*J_b[w, i, 2])
            interactions_list.append(inter)

    # --- Create Inverse Mass Array ---
    spatial_inertia_list = []
    for w in range(N_w):
        for i in range(N_b):
            inertia = SpatialInertia()
            inertia.m = np.random.rand() + 0.5  # mass from 0.5 to 1.5
            inertia.inertia = np.diag(np.random.rand(3) * 0.1 + 0.05)  # simple diagonal inertia
            spatial_inertia_list.append(inertia)

    # --- Assemble final data dictionary for Warp ---
    data = {
        "body_u": wp.from_numpy(
            (np.random.rand(N_w, N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_u_prev": wp.from_numpy(
            (np.random.rand(N_w, N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_lambda_n": wp.from_numpy(
            (np.random.rand(N_w, N_c) * 0.1).astype(np.float32), device=device, dtype=wp.float32
        ),
        "interactions": wp.array(
            interactions_list, dtype=ContactInteraction, device=device
        ).reshape((N_w, N_c)),
        "body_M_inv": wp.array(spatial_inertia_list, dtype=SpatialInertia, device=device).reshape(
            (N_w, N_b)
        ),
        "dt": 0.01,
        "stabilization_factor": 0.2,
        "fb_alpha": 0.25,
        "fb_beta": 0.25,
        "compliance": 1e-6,
        # Outputs
        "h_d": wp.zeros((N_w, N_b), dtype=wp.spatial_vector, device=device),
        "h_n": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
        "J_hat_n_values": wp.zeros((N_w, N_c, 2), dtype=wp.spatial_vector, device=device),
        "C_n_values": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
        "s_n": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(
    num_worlds,
    num_bodies,
    num_contacts,
    num_iterations_total=1000,
    num_iterations_in_graph=100,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    if num_iterations_total % num_iterations_in_graph != 0:
        raise ValueError("num_iterations_total must be divisible by num_iterations_in_graph")
    num_graph_launches = num_iterations_total // num_iterations_in_graph

    print(
        f"\n--- Benchmarking: Worlds={num_worlds}, Bodies/World={num_bodies}, Contacts/World={num_contacts} ---"
    )
    print(f"    Scenario: Inactive={inactive_contact_ratio:.0%}, Fixed={fixed_body_ratio:.0%}")
    print(
        f"    Configuration: Total Iterations={num_iterations_total}, Graph Capture Size={num_iterations_in_graph}, Graph Launches={num_graph_launches}"
    )

    device = wp.get_device()
    data = setup_data(
        num_worlds,
        num_bodies,
        num_contacts,
        device,
        inactive_contact_ratio,
        fixed_body_ratio,
    )
    kernel_args = [
        data["body_u"],
        data["body_u_prev"],
        data["body_lambda_n"],
        data["interactions"],
        data["body_M_inv"],
        data["dt"],
        data["stabilization_factor"],
        data["fb_alpha"],
        data["fb_beta"],
        data["compliance"],
        data["h_d"],
        data["h_n"],
        data["J_hat_n_values"],
        data["C_n_values"],
        data["s_n"],
    ]

    # --- Standard Launch Benchmark ---
    print(f"1. Benching Standard Kernel Launch ({num_iterations_total} launches)...")
    # Warm-up a single launch
    wp.launch(
        kernel=contact_constraint_kernel,
        dim=(num_worlds, num_contacts),
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations_total):
        # In a real solver, outputs from one iteration feed into the next.
        # Here we just zero one of the outputs to simulate work between launches.
        data["h_d"].zero_()
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=(num_worlds, num_contacts),
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_total_time = time.perf_counter() - start_time
    standard_launch_ms = standard_total_time / num_iterations_total * 1000
    print(f"   Avg. Time per iteration: {standard_launch_ms:.4f} ms")
    print(f"   Total Time for {num_iterations_total} iterations: {standard_total_time*1000:.2f} ms")

    # --- CUDA Graph Benchmark ---
    if device.is_cuda:
        print(
            f"2. Benching CUDA Graph Launch ({num_graph_launches} launches of a {num_iterations_in_graph}-iteration graph)..."
        )

        # --- Graph Capture ---
        print(f"   Capturing {num_iterations_in_graph} iterations into a graph...")
        wp.capture_begin()
        for _ in range(num_iterations_in_graph):
            data["h_d"].zero_()
            wp.launch(
                kernel=contact_constraint_kernel,
                dim=(num_worlds, num_contacts),
                inputs=kernel_args,
                device=device,
            )
        graph = wp.capture_end()
        print("   Graph capture complete.")

        # Warm-up launch of the entire graph
        wp.capture_launch(graph)
        wp.synchronize()

        # --- Timed Graph Launch ---
        start_time = time.perf_counter()
        for _ in range(num_graph_launches):
            wp.capture_launch(graph)
        wp.synchronize()
        graph_total_time = time.perf_counter() - start_time

        # Calculate the average time per equivalent single iteration
        graph_equivalent_iter_ms = graph_total_time / num_iterations_total * 1000

        # Calculate the average time per launch of the entire graph
        avg_graph_launch_ms = graph_total_time / num_graph_launches * 1000

        print(f"   Avg. Time per equivalent iteration: {graph_equivalent_iter_ms:.4f} ms")
        print(
            f"   Avg. Time per graph launch ({num_iterations_in_graph} iters): {avg_graph_launch_ms:.4f} ms"
        )
        print(
            f"   Total Time for {num_graph_launches} graph launches: {graph_total_time*1000:.2f} ms"
        )

        if graph_equivalent_iter_ms > 1e-9:
            speedup = standard_launch_ms / graph_equivalent_iter_ms
            print(f"   Speedup from Graph (per iteration): {speedup:.2f}x")
    else:
        print("2. Skipping CUDA Graph benchmark (not on a CUDA-enabled device).")

    print("-" * 70)


if __name__ == "__main__":
    # Initialize Warp.
    wp.init()
    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print(
            "Warning: No CUDA device found. Performance will be low and CUDA graph test will be skipped."
        )

    # Simulation parameters
    N_WORLDS = 20
    N_BODIES = 50
    N_CONTACTS = 500

    # Iteration parameters for the benchmark
    TOTAL_ITERATIONS = 1000
    ITERATIONS_PER_GRAPH = 50

    # --- Baseline ---
    run_benchmark(
        num_worlds=N_WORLDS,
        num_bodies=N_BODIES,
        num_contacts=N_CONTACTS,
        num_iterations_total=TOTAL_ITERATIONS,
        num_iterations_in_graph=ITERATIONS_PER_GRAPH,
        inactive_contact_ratio=0.0,
        fixed_body_ratio=0.0,
    )

    # --- High Divergence (many inactive contacts) ---
    run_benchmark(
        num_worlds=N_WORLDS,
        num_bodies=N_BODIES,
        num_contacts=N_CONTACTS,
        num_iterations_total=TOTAL_ITERATIONS,
        num_iterations_in_graph=ITERATIONS_PER_GRAPH,
        inactive_contact_ratio=0.5,
        fixed_body_ratio=0.0,
    )

    # --- Fixed Body Scenario ---
    run_benchmark(
        num_worlds=N_WORLDS,
        num_bodies=N_BODIES,
        num_contacts=N_CONTACTS,
        num_iterations_total=TOTAL_ITERATIONS,
        num_iterations_in_graph=ITERATIONS_PER_GRAPH,
        inactive_contact_ratio=0.0,
        fixed_body_ratio=0.2,
    )
