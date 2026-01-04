import time

import numpy as np
import warp as wp
from axion.constraints import friction_constraint_kernel
from axion.types import ContactInteraction


def setup_data(
    num_bodies,
    num_contacts,
    device,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """
    Generates random input and output arrays for the kernel benchmark.
    This is very similar to the setup for the normal contact constraint kernel.
    """
    N_b, N_c = num_bodies, num_contacts

    # --- Generate Penetration Depths ---
    # Start with all contacts penetrating, then make a fraction of them inactive.
    gaps = np.random.rand(N_c) * -0.1  # All negative (active)
    num_inactive = int(N_c * inactive_contact_ratio)
    if num_inactive > 0:
        inactive_indices = np.random.choice(N_c, num_inactive, replace=False)
        gaps[inactive_indices] = (
            np.random.rand(num_inactive) * 0.1
        )  # Make them positive
    penetration_depths = -gaps  # Positive for penetration

    # --- Generate Body Indices ---
    # Start with all valid body indices, ensuring no self-contacts.
    body_a_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(
            0, N_b, size=np.sum(mask), dtype=np.int32
        )
        mask = body_a_indices == body_b_indices

    # Now, introduce fixed bodies (-1) for a fraction of contacts.
    num_fixed = int(N_c * fixed_body_ratio)
    if num_fixed > 0:
        fixed_indices = np.random.choice(N_c, num_fixed, replace=False)
        # For each chosen contact, randomly set either body A or B to be fixed.
        # This avoids creating static-static (-1 vs -1) contacts.
        chooser = np.random.randint(0, 2, size=num_fixed)
        body_a_indices[fixed_indices[chooser == 0]] = -1
        body_b_indices[fixed_indices[chooser == 1]] = -1

    # --- Generate Random Jacobians (Basis Vectors) ---
    J_a = (np.random.rand(N_c, 3, 6) - 0.5).astype(np.float32)
    J_b = (np.random.rand(N_c, 3, 6) - 0.5).astype(np.float32)

    # --- Generate Restitution and Friction Coefficients ---
    restitution_coeffs = (np.random.rand(N_c) * 0.5).astype(np.float32)
    friction_coeffs = np.random.rand(N_c).astype(np.float32)

    # --- Create Interactions Array ---
    interactions_list = []
    for i in range(N_c):
        inter = ContactInteraction()
        inter.is_active = penetration_depths[i] > 0.0
        inter.body_a_idx = body_a_indices[i]
        inter.body_b_idx = body_b_indices[i]
        inter.penetration_depth = penetration_depths[i]
        inter.restitution_coeff = restitution_coeffs[i]
        inter.friction_coeff = friction_coeffs[i]

        # Set basis vectors from random Jacobians
        inter.basis_a.normal = wp.spatial_vector(*J_a[i, 0])
        inter.basis_a.tangent1 = wp.spatial_vector(*J_a[i, 1])
        inter.basis_a.tangent2 = wp.spatial_vector(*J_a[i, 2])
        inter.basis_b.normal = wp.spatial_vector(*J_b[i, 0])
        inter.basis_b.tangent1 = wp.spatial_vector(*J_b[i, 1])
        inter.basis_b.tangent2 = wp.spatial_vector(*J_b[i, 2])

        interactions_list.append(inter)

    interactions = wp.array(interactions_list, dtype=ContactInteraction, device=device)

    data = {
        "body_qd": wp.from_numpy(
            (np.random.rand(N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "interactions": interactions,
        "lambda_f": wp.from_numpy(
            (np.random.rand(2 * N_c) * 10.0).astype(np.float32), device=device
        ),
        "lambda_n_prev": wp.from_numpy(
            (np.random.rand(N_c) * 10.0).astype(np.float32), device=device
        ),
        "fb_alpha": 0.25,
        "fb_beta": 0.25,
        "compliance": 1e-6,
        "g": wp.zeros(N_b, dtype=wp.spatial_vector, device=device),
        "h_f": wp.zeros(2 * N_c, dtype=wp.float32, device=device),
        "J_f_values": wp.zeros((2 * N_c, 2), dtype=wp.spatial_vector, device=device),
        "C_f_values": wp.zeros(2 * N_c, dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(
    num_bodies,
    num_contacts,
    num_iterations=200,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking: N_b={num_bodies}, N_c={num_contacts}, Inactive={inactive_contact_ratio:.0%}, Fixed={fixed_body_ratio:.0%}, Iters={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(
        num_bodies, num_contacts, device, inactive_contact_ratio, fixed_body_ratio
    )

    kernel_args = [
        data["body_qd"],
        data["interactions"],
        data["lambda_f"],
        data["lambda_n_prev"],
        data["fb_alpha"],
        data["fb_beta"],
        data["compliance"],
        data["g"],
        data["h_f"],
        data["J_f_values"],
        data["C_f_values"],
    ]

    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        kernel=friction_constraint_kernel,
        dim=num_contacts,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=friction_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        wp.capture_begin()
        data["g"].zero_()
        wp.launch(
            kernel=friction_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
            device=device,
        )
        graph = wp.capture_end()

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
        print("2. Skipping CUDA Graph benchmark (not on a CUDA-enabled device).")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print(
            "\nWarning: No CUDA device found. Performance will be poor and CUDA graph test will be skipped."
        )

    # --- Baseline Benchmarks (all active contacts, no fixed bodies) ---
    print("\n>>> Running Baseline Benchmarks (0% Inactive, 0% Fixed)...")
    run_benchmark(num_bodies=400, num_contacts=800)

    # --- Divergence Benchmarks (inactive contacts) ---
    print("\n>>> Running Divergence Benchmarks (50% Inactive Contacts)...")
    run_benchmark(num_bodies=400, num_contacts=800, inactive_contact_ratio=0.5)

    # --- Fixed Body Benchmarks ---
    print("\n>>> Running Fixed Body Benchmarks (20% Fixed Bodies)...")
    run_benchmark(num_bodies=400, num_contacts=800, fixed_body_ratio=0.2)

    # --- Combined Scenario ---
    print("\n>>> Running Combined Scenario Benchmarks (50% Inactive, 20% Fixed)...")
    run_benchmark(
        num_bodies=400,
        num_contacts=800,
        inactive_contact_ratio=0.5,
        fixed_body_ratio=0.2,
    )
