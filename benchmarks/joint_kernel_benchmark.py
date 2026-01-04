import time

import numpy as np
import warp as wp
from axion.constraints import joint_constraint_kernel
from axion.types import JointInteraction


def _rand_quat():
    """Generate a random quaternion."""
    u, v, w = np.random.uniform(0, 1, 3)
    return np.array(
        [
            np.sqrt(1 - u) * np.sin(2 * np.pi * v),
            np.sqrt(1 - u) * np.cos(2 * np.pi * v),
            np.sqrt(u) * np.sin(2 * np.pi * w),
            np.sqrt(u) * np.cos(2 * np.pi * w),
        ]
    )


def setup_data(
    num_bodies,
    num_joints,
    device,
    disabled_joint_ratio=0.0,
    world_joint_ratio=0.0,
):
    """
    Generates random input and output arrays for the kernel benchmark.

    Args:
        num_bodies (int): Total number of bodies in the simulation.
        num_joints (int): Total number of joints.
        device (str): The Warp device to create arrays on.
        disabled_joint_ratio (float): The fraction (0.0 to 1.0) of joints
            that should be disabled.
        world_joint_ratio (float): The fraction (0.0 to 1.0) of joints
            that should connect to the world (parent index -1).
    """
    N_b, N_j = num_bodies, num_joints

    # --- Generate Joint Connectivity ---
    parent_indices = np.random.randint(0, N_b, size=N_j, dtype=np.int32)
    child_indices = np.random.randint(0, N_b, size=N_j, dtype=np.int32)
    mask = parent_indices == child_indices
    while np.any(mask):
        child_indices[mask] = np.random.randint(
            0, N_b, size=np.sum(mask), dtype=np.int32
        )
        mask = parent_indices == child_indices

    num_world = int(N_j * world_joint_ratio)
    if num_world > 0:
        world_indices = np.random.choice(N_j, num_world, replace=False)
        parent_indices[world_indices] = -1

    # --- Generate Joint States ---
    joint_active = np.ones(N_j, dtype=bool)
    num_disabled = int(N_j * disabled_joint_ratio)
    if num_disabled > 0:
        disabled_indices = np.random.choice(N_j, num_disabled, replace=False)
        joint_active[disabled_indices] = False

    # --- System and Constraint Dimensions ---
    num_j_constraints = N_j * 5

    # --- Create Interactions Array ---
    interactions_list = []
    for i in range(N_j):
        inter = JointInteraction()
        inter.is_active = joint_active[i]
        inter.parent_idx = parent_indices[i]
        inter.child_idx = child_indices[i]

        is_world = parent_indices[i] < 0

        # axis0
        inter.axis0.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis0.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis0.error = np.random.rand() * 0.2 - 0.1
        inter.axis0.compliance = 0.0

        # axis1
        inter.axis1.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis1.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis1.error = np.random.rand() * 0.2 - 0.1
        inter.axis1.compliance = 0.0

        # axis2
        inter.axis2.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis2.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis2.error = np.random.rand() * 0.2 - 0.1
        inter.axis2.compliance = 0.0

        # axis3
        inter.axis3.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis3.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis3.error = np.random.rand() * 0.2 - 0.1
        inter.axis3.compliance = 0.0

        # axis4
        inter.axis4.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis4.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis4.error = np.random.rand() * 0.2 - 0.1
        inter.axis4.compliance = 0.0

        # axis5
        inter.axis5.J_child = wp.spatial_vector(*(np.random.rand(6) - 0.5))
        inter.axis5.J_parent = (
            wp.spatial_vector()
            if is_world
            else wp.spatial_vector(*(np.random.rand(6) - 0.5))
        )
        inter.axis5.error = np.random.rand() * 0.2 - 0.1
        inter.axis5.compliance = 0.0

        interactions_list.append(inter)

    interactions = wp.array(interactions_list, dtype=JointInteraction, device=device)

    # --- Create Warp arrays from NumPy data ---
    data = {
        "body_qd": wp.from_numpy(
            (np.random.rand(N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "interactions": interactions,
        "lambda_j": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
        "dt": 1.0 / 60.0,
        "joint_stabilization_factor": 0.1,
        "g": wp.zeros(N_b, dtype=wp.spatial_vector, device=device),
        "h_j": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
        "J_j_values": wp.zeros(
            (num_j_constraints, 2), dtype=wp.spatial_vector, device=device
        ),
        "C_j_values": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(
    num_bodies,
    num_joints,
    num_iterations=200,
    disabled_joint_ratio=0.0,
    world_joint_ratio=0.0,
):
    """Measures execution time of the joint kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking: N_b={num_bodies}, N_j={num_joints}, "
        f"Disabled={disabled_joint_ratio:.0%}, World={world_joint_ratio:.0%}, "
        f"Iters={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(
        num_bodies,
        num_joints,
        device,
        disabled_joint_ratio,
        world_joint_ratio,
    )

    kernel_args = [
        data["body_qd"],
        data["lambda_j"],
        data["interactions"],
        data["dt"],
        data["joint_stabilization_factor"],
        data["g"],
        data["h_j"],
        data["J_j_values"],
        data["C_j_values"],
    ]

    # --- Standard Launch ---
    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        kernel=joint_constraint_kernel,
        dim=(5, num_joints),
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=joint_constraint_kernel,
            dim=(5, num_joints),
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- CUDA Graph Launch ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        wp.capture_begin()
        data["g"].zero_()
        wp.launch(
            kernel=joint_constraint_kernel,
            dim=(5, num_joints),
            inputs=kernel_args,
            device=device,
        )
        graph = wp.capture_end()

        # Warm-up run
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
            "\nWarning: No CUDA device found. Performance will be poor and "
            "CUDA graph test will be skipped."
        )

    # --- Baseline: Typical ragdoll-like characters ---
    print("\n>>> Running Baseline Benchmarks (0% Disabled, 5% World)...")
    run_benchmark(num_bodies=200, num_joints=199, world_joint_ratio=0.05)
    run_benchmark(num_bodies=500, num_joints=499, world_joint_ratio=0.05)

    # --- Divergence Test: Simulating scenes where many objects are asleep ---
    print("\n>>> Running Divergence Benchmarks (50% Disabled Joints)...")
    run_benchmark(
        num_bodies=500, num_joints=499, disabled_joint_ratio=0.5, world_joint_ratio=0.05
    )

    # --- World Connection Test: Simulating many objects anchored to the world ---
    print("\n>>> Running World Connection Benchmarks (50% World Joints)...")
    run_benchmark(num_bodies=500, num_joints=499, world_joint_ratio=0.5)

    # --- Combined Scenario ---
    print("\n>>> Running Combined Scenario (50% Disabled, 50% World)...")
    run_benchmark(
        num_bodies=500,
        num_joints=499,
        disabled_joint_ratio=0.5,
        world_joint_ratio=0.5,
    )
