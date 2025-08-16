import time

import numpy as np
import warp as wp
from axion.constraints import joint_constraint_kernel


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
    joint_enabled = np.ones(N_j, dtype=np.int32)
    num_disabled = int(N_j * disabled_joint_ratio)
    if num_disabled > 0:
        disabled_indices = np.random.choice(N_j, num_disabled, replace=False)
        joint_enabled[disabled_indices] = 0

    # --- System and Constraint Dimensions ---
    num_j_constraints = N_j * 5
    dyn_dim = N_b * 6

    # --- Create NumPy arrays for all kernel arguments ---
    np_body_q = np.array(
        [np.concatenate([np.random.randn(3), _rand_quat()]) for _ in range(N_b)],
        dtype=np.float32,
    )
    np_joint_xp = np.array(
        [np.concatenate([np.random.randn(3), _rand_quat()]) for _ in range(N_j)],
        dtype=np.float32,
    )
    np_joint_xc = np.array(
        [np.concatenate([np.random.randn(3), _rand_quat()]) for _ in range(N_j)],
        dtype=np.float32,
    )
    np_joint_axis = (np.random.rand(N_j, 3) - 0.5) * 2.0
    np_joint_axis /= np.linalg.norm(np_joint_axis, axis=1)[:, None]

    # --- Create Warp arrays from NumPy data ---
    # This is an efficient way to move data from the host to the GPU device.
    data = {
        "body_q": wp.from_numpy(np_body_q, dtype=wp.transform, device=device),
        "body_qd": wp.from_numpy(
            (np.random.rand(N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_com": wp.from_numpy(
            np.zeros((N_b, 3), dtype=np.float32), dtype=wp.vec3, device=device
        ),
        "joint_type": wp.array(
            [wp.sim.JOINT_REVOLUTE] * N_j, dtype=wp.int32, device=device
        ),
        "joint_enabled": wp.from_numpy(joint_enabled, device=device),
        "joint_parent": wp.from_numpy(parent_indices, device=device),
        "joint_child": wp.from_numpy(child_indices, device=device),
        "joint_X_p": wp.from_numpy(np_joint_xp, dtype=wp.transform, device=device),
        "joint_X_c": wp.from_numpy(np_joint_xc, dtype=wp.transform, device=device),
        "joint_axis_start": wp.array(np.arange(N_j), dtype=wp.int32, device=device),
        "joint_axis_dim": wp.array([[0, 1]] * N_j, dtype=wp.int32, device=device),
        "joint_axis": wp.from_numpy(
            np_joint_axis.astype(np.float32), dtype=wp.vec3, device=device
        ),
        "joint_linear_compliance": wp.zeros(N_j, dtype=wp.float32, device=device),
        "joint_angular_compliance": wp.zeros(N_j, dtype=wp.float32, device=device),
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
        data["body_q"],
        data["body_qd"],
        data["body_com"],
        data["joint_type"],
        data["joint_enabled"],
        data["joint_parent"],
        data["joint_child"],
        data["joint_X_p"],
        data["joint_X_c"],
        data["joint_axis_start"],
        data["joint_axis_dim"],
        data["joint_axis"],
        data["joint_linear_compliance"],
        data["joint_angular_compliance"],
        data["lambda_j"],
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
