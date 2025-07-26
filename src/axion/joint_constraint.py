"""
Defines the core NVIDIA Warp kernel for processing joint constraints.

This module is a key component of a Non-Smooth Newton (NSN) physics engine,
responsible for enforcing the kinematic constraints imposed by joints.
The current implementation focuses on revolute joints, which restrict relative
motion between two bodies to a single rotational degree of freedom.

For each revolute joint, the kernel computes the residuals and Jacobians for
five constraints:
- Three translational constraints to lock the joint's position.
- Two rotational constraints to align the bodies, allowing rotation only
  around the specified joint axis.

These outputs are used by the main solver to compute corrective impulses that
maintain the joint connections. The computations are designed for parallel
execution on the GPU [nvidia.github.io/warp](https://nvidia.github.io/warp/).
"""
import time

import numpy as np
import warp as wp
from axion.utils import orthogonal_basis

wp.config.lineinfo = True


@wp.kernel
def joint_constraint_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    # --- Joint Definition Inputs ---
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables (from current Newton iterate) ---
    lambda_j_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Offsets for Output Arrays ---
    h_j_offset: wp.int32,
    J_j_offset: wp.int32,
    C_j_offset: wp.int32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    """
    Computes residuals and Jacobians for revolute joint constraints.

    This kernel is launched once per joint. For each active revolute joint, it calculates
    the contributions to the global linear system being solved by the NSN engine.
    It formulates the constraints at the velocity level, including a Baumgarte
    stabilization term to correct positional and rotational drift over time.

    Args:
        body_q: (N_b,) Current transforms (position and orientation) of all bodies.
        body_qd: (N_b,) Current spatial velocities of all bodies.
        body_com: (N_b,) Center of mass offset for each body.
        joint_type: (N_j,) Type of each joint (e.g., revolute, prismatic).
        joint_enabled: (N_j,) Flag indicating if a joint is active (1) or disabled (0).
        joint_parent: (N_j,) Index of the parent body for each joint. -1 for world connection.
        joint_child: (N_j,) Index of the child body for each joint.
        joint_X_p: (N_j,) Transform from the parent body's frame to the joint frame.
        joint_X_c: (N_j,) Transform from the child body's frame to the joint frame.
        joint_axis_start: (N_j,) Starting index in `joint_axis` for this joint's axes.
        joint_axis_dim: (N_j, 2) Dimensions of linear and angular axes.
        joint_axis: (N_a,) Array of all joint axes vectors.
        joint_linear_compliance: (N_j,) Compliance for translational constraints.
        joint_angular_compliance: (N_j,) Compliance for rotational constraints.
        lambda_j_offset: Integer offset for joint impulses in the global `_lambda` array.
        _lambda: (con_dim,) Full vector of constraint impulses from the current Newton iteration.
        dt: Simulation timestep duration.
        joint_stabilization_factor: Baumgarte stabilization coefficient for joints.
        h_j_offset: Start index in the global `h` vector for joint constraint residuals.
        J_j_offset: Start row in the global `J_values` matrix for joint constraint Jacobians.
        C_j_offset: Start index in the global `C_values` vector for joint compliance values.

    Outputs (written via atomic adds or direct indexing):
        g: (N_b * 6,) Accumulates generalized forces from joint impulses (`-J_j^T * λ_j`).
        h: (con_dim,) Stores the constraint residuals. Writes 5 error values per joint.
        J_values: (con_dim, 2) Stores Jacobian blocks. Writes 5 Jacobian rows per joint.
        C_values: (con_dim,) Stores compliance blocks. Writes 5 compliance values per joint.
    """
    tid = wp.tid()
    j_type = joint_type[tid]

    # Early exit for disabled or non-revolute joints
    if (
        joint_enabled[tid] == 0
        or j_type != wp.sim.JOINT_REVOLUTE
        or joint_parent[tid] < 0  # To-do: Handle world joints
    ):
        return

    child_idx = joint_child[tid]
    parent_idx = joint_parent[tid]

    # Kinematics (Child)
    body_q_c = body_q[child_idx]
    X_wj_c = body_q_c * joint_X_c[tid]
    r_c = wp.transform_get_translation(X_wj_c) - wp.transform_point(
        body_q_c, body_com[child_idx]
    )
    q_c_rot = wp.transform_get_rotation(body_q_c)

    # Kinematics (Parent)
    body_q_p = body_q[parent_idx]
    X_wj_p = body_q_p * joint_X_p[tid]
    r_p = wp.transform_get_translation(X_wj_p) - wp.transform_point(
        body_q_p, body_com[parent_idx]
    )
    q_p_rot = wp.transform_get_rotation(body_q_p)

    # Joint Axis in World Frame
    axis = joint_axis[joint_axis_start[tid]]
    axis_p_w = wp.quat_rotate(q_p_rot, axis)

    # Define orthogonal basis in child's local frame
    b1_c, b2_c = orthogonal_basis(axis)
    b1_c_w = wp.quat_rotate(q_c_rot, b1_c)
    b2_c_w = wp.quat_rotate(q_c_rot, b2_c)

    # Positional Constraint Error
    C_pos = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)

    # Rotational Constraint Error
    C_rot_u = wp.dot(axis_p_w, b1_c_w)
    C_rot_v = wp.dot(axis_p_w, b2_c_w)

    # Jacobian Calculation (Positional)
    J_pos_x_c = wp.spatial_vector(0.0, r_c[2], -r_c[1], 1.0, 0.0, 0.0)
    J_pos_y_c = wp.spatial_vector(-r_c[2], 0.0, r_c[0], 0.0, 1.0, 0.0)
    J_pos_z_c = wp.spatial_vector(r_c[1], -r_c[0], 0.0, 0.0, 0.0, 1.0)
    J_pos_x_p = wp.spatial_vector(0.0, -r_p[2], r_p[1], -1.0, 0.0, 0.0)
    J_pos_y_p = wp.spatial_vector(r_p[2], 0.0, -r_p[0], 0.0, -1.0, 0.0)
    J_pos_z_p = wp.spatial_vector(-r_p[1], r_p[0], 0.0, 0.0, 0.0, -1.0)

    # Jacobian Calculation (Rotational)
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)
    zero_vec = wp.vec3()
    J_rot_u_c = wp.spatial_vector(-b1_x_axis, zero_vec)
    J_rot_v_c = wp.spatial_vector(-b2_x_axis, zero_vec)
    J_rot_u_p = wp.spatial_vector(b1_x_axis, zero_vec)
    J_rot_v_p = wp.spatial_vector(b2_x_axis, zero_vec)

    # Velocity Error Calculation
    body_qd_c = body_qd[child_idx]
    body_qd_p = body_qd[parent_idx]
    C_dot_pos_x = wp.dot(J_pos_x_c, body_qd_c) + wp.dot(J_pos_x_p, body_qd_p)
    C_dot_pos_y = wp.dot(J_pos_y_c, body_qd_c) + wp.dot(J_pos_y_p, body_qd_p)
    C_dot_pos_z = wp.dot(J_pos_z_c, body_qd_c) + wp.dot(J_pos_z_p, body_qd_p)
    C_dot_rot_u = wp.dot(J_rot_u_c, body_qd_c) + wp.dot(J_rot_u_p, body_qd_p)
    C_dot_rot_v = wp.dot(J_rot_v_c, body_qd_c) + wp.dot(J_rot_v_p, body_qd_p)

    # --- Update global system components ---
    # 1. Residual Vector h (constraint violation)
    bias_scale = joint_stabilization_factor / dt
    base_h_idx = h_j_offset + tid * 5
    h[base_h_idx + 0] = C_dot_pos_x + bias_scale * C_pos.x
    h[base_h_idx + 1] = C_dot_pos_y + bias_scale * C_pos.y
    h[base_h_idx + 2] = C_dot_pos_z + bias_scale * C_pos.z
    h[base_h_idx + 3] = C_dot_rot_u + bias_scale * C_rot_u
    h[base_h_idx + 4] = C_dot_rot_v + bias_scale * C_rot_v

    # 2. Update g (momentum balance residual: -J^T * lambda)
    base_lambda_idx = lambda_j_offset + tid * 5
    lambda_j_x = _lambda[base_lambda_idx + 0]
    lambda_j_y = _lambda[base_lambda_idx + 1]
    lambda_j_z = _lambda[base_lambda_idx + 2]
    lambda_j_u = _lambda[base_lambda_idx + 3]
    lambda_j_v = _lambda[base_lambda_idx + 4]

    g_c = (
        -J_pos_x_c * lambda_j_x
        - J_pos_y_c * lambda_j_y
        - J_pos_z_c * lambda_j_z
        - J_rot_u_c * lambda_j_u
        - J_rot_v_c * lambda_j_v
    )
    for i in range(wp.static(6)):
        wp.atomic_add(g, child_idx * 6 + i, g_c[i])

    g_p = (
        -J_pos_x_p * lambda_j_x
        - J_pos_y_p * lambda_j_y
        - J_pos_z_p * lambda_j_z
        - J_rot_u_p * lambda_j_u
        - J_rot_v_p * lambda_j_v
    )
    for i in range(wp.static(6)):
        wp.atomic_add(g, parent_idx * 6 + i, g_p[i])

    # 3. Compliance (diagonal block C of the system matrix)
    base_C_idx = C_j_offset + tid * 5
    C_values[base_C_idx + 0] = joint_linear_compliance[tid]
    C_values[base_C_idx + 1] = joint_linear_compliance[tid]
    C_values[base_C_idx + 2] = joint_linear_compliance[tid]
    C_values[base_C_idx + 3] = joint_angular_compliance[tid]
    C_values[base_C_idx + 4] = joint_angular_compliance[tid]

    # 4. Jacobian (off-diagonal block J of the system matrix)
    base_J_idx = J_j_offset + tid * 5
    J_values[base_J_idx + 0, 0] = J_pos_x_p
    J_values[base_J_idx + 0, 1] = J_pos_x_c
    J_values[base_J_idx + 1, 0] = J_pos_y_p
    J_values[base_J_idx + 1, 1] = J_pos_y_c
    J_values[base_J_idx + 2, 0] = J_pos_z_p
    J_values[base_J_idx + 2, 1] = J_pos_z_c
    J_values[base_J_idx + 3, 0] = J_rot_u_p
    J_values[base_J_idx + 3, 1] = J_rot_u_c
    J_values[base_J_idx + 4, 0] = J_rot_v_p
    J_values[base_J_idx + 4, 1] = J_rot_v_c


# =======================================================================
#
#   BENCHMARKING SUITE
#
# =======================================================================


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
        "_lambda": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
        "params": {
            "lambda_j_offset": 0,
            "dt": 1.0 / 60.0,
            "joint_stabilization_factor": 0.1,
            "h_j_offset": 0,
            "J_j_offset": 0,
            "C_j_offset": 0,
        },
        "g": wp.zeros(dyn_dim, dtype=wp.float32, device=device),
        "h": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
        "J_values": wp.zeros(
            (num_j_constraints, 2), dtype=wp.spatial_vector, device=device
        ),
        "C_values": wp.zeros(num_j_constraints, dtype=wp.float32, device=device),
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

    params = data["params"]
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
        params["lambda_j_offset"],
        data["_lambda"],
        params["dt"],
        params["joint_stabilization_factor"],
        params["h_j_offset"],
        params["J_j_offset"],
        params["C_j_offset"],
        data["g"],
        data["h"],
        data["J_values"],
        data["C_values"],
    ]

    # --- Standard Launch ---
    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        kernel=joint_constraint_kernel,
        dim=num_joints,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=joint_constraint_kernel,
            dim=num_joints,
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
            dim=num_joints,
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
    run_benchmark(num_bodies=1000, num_joints=999, world_joint_ratio=0.05)

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
