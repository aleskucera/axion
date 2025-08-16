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
import warp as wp

from .utils import get_random_idx_to_res_buffer
from .utils import orthogonal_basis


@wp.func
def compute_joint_kinematics(
    body_q: wp.transform, joint_X: wp.transform, body_com: wp.vec3
) -> wp.vec3:
    """
    Compute the joint position relative to the body's center of mass.

    Args:
        body_q: Body's current transform (position + rotation)
        joint_X: Joint attachment transform relative to body frame
        body_com: Body's center of mass in body frame

    Returns:
        r: Vector from body COM to joint position in world frame
    """
    X_wj = body_q * joint_X
    joint_pos_world = wp.transform_get_translation(X_wj)
    body_com_world = wp.transform_point(body_q, body_com)
    return joint_pos_world - body_com_world


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
    lambda_j: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.spatial_vector),
    h_j: wp.array(dtype=wp.float32),
    J_j_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_j_values: wp.array(dtype=wp.float32),
):
    constraint_idx, joint_idx = wp.tid()
    j_type = joint_type[joint_idx]

    # Early exit for disabled or non-revolute joints
    if (
        joint_enabled[joint_idx] == 0
        or j_type != wp.sim.JOINT_REVOLUTE
        or joint_parent[joint_idx] < 0
    ):
        return

    child_idx = joint_child[joint_idx]
    parent_idx = joint_parent[joint_idx]

    body_q_c = body_q[child_idx]
    body_q_p = body_q[parent_idx]

    r_c = compute_joint_kinematics(body_q_c, joint_X_c[joint_idx], body_com[child_idx])
    r_p = compute_joint_kinematics(body_q_p, joint_X_p[joint_idx], body_com[parent_idx])

    # Only compute child rotation when needed (for rotational constraints)
    q_c_rot = wp.transform_get_rotation(body_q_c)
    q_p_rot = wp.transform_get_rotation(body_q_p)

    # Positional Constraint Error - shared for constraints 0, 1, 2
    joint_pos_c = wp.transform_get_translation(body_q_c * joint_X_c[joint_idx])
    joint_pos_p = wp.transform_get_translation(body_q_p * joint_X_p[joint_idx])
    c_pos = joint_pos_c - joint_pos_p

    # Velocity Error Calculation
    body_qd_c = body_qd[child_idx]
    body_qd_p = body_qd[parent_idx]

    global_constraint_idx = joint_idx * 5 + constraint_idx
    bias_scale = joint_stabilization_factor / dt

    # Constraint-specific computations
    if constraint_idx == 0:  # X translation
        J_c = wp.spatial_vector(0.0, r_c[2], -r_c[1], 1.0, 0.0, 0.0)
        J_p = wp.spatial_vector(0.0, -r_p[2], r_p[1], -1.0, 0.0, 0.0)
        grad_c = wp.dot(J_c, body_qd_c) + wp.dot(J_p, body_qd_p)
        constraint_error = c_pos.x
        compliance = joint_linear_compliance[joint_idx]

    elif constraint_idx == 1:  # Y translation
        J_c = wp.spatial_vector(-r_c[2], 0.0, r_c[0], 0.0, 1.0, 0.0)
        J_p = wp.spatial_vector(r_p[2], 0.0, -r_p[0], 0.0, -1.0, 0.0)
        grad_c = wp.dot(J_c, body_qd_c) + wp.dot(J_p, body_qd_p)
        constraint_error = c_pos.y
        compliance = joint_linear_compliance[joint_idx]

    elif constraint_idx == 2:  # Z translation
        J_c = wp.spatial_vector(r_c[1], -r_c[0], 0.0, 0.0, 0.0, 1.0)
        J_p = wp.spatial_vector(-r_p[1], r_p[0], 0.0, 0.0, 0.0, -1.0)
        grad_c = wp.dot(J_c, body_qd_c) + wp.dot(J_p, body_qd_p)
        constraint_error = c_pos.z
        compliance = joint_linear_compliance[joint_idx]

    elif constraint_idx == 3:  # U rotation
        axis = joint_axis[joint_axis_start[joint_idx]]
        axis_p_w = wp.quat_rotate(q_p_rot, axis)
        b1_c, b2_c = orthogonal_basis(axis)
        b1_c_w = wp.quat_rotate(q_c_rot, b1_c)

        b1_x_axis = wp.cross(axis_p_w, b1_c_w)
        zero_vec = wp.vec3()

        J_c = wp.spatial_vector(-b1_x_axis, zero_vec)
        J_p = wp.spatial_vector(b1_x_axis, zero_vec)
        grad_c = wp.dot(J_c, body_qd_c) + wp.dot(J_p, body_qd_p)
        constraint_error = wp.dot(axis_p_w, b1_c_w)
        compliance = joint_angular_compliance[joint_idx]

    elif constraint_idx == 4:  # V rotation
        axis = joint_axis[joint_axis_start[joint_idx]]
        axis_p_w = wp.quat_rotate(q_p_rot, axis)
        b1_c, b2_c = orthogonal_basis(axis)
        b2_c_w = wp.quat_rotate(q_c_rot, b2_c)

        b2_x_axis = wp.cross(axis_p_w, b2_c_w)
        zero_vec = wp.vec3()

        J_c = wp.spatial_vector(-b2_x_axis, zero_vec)
        J_p = wp.spatial_vector(b2_x_axis, zero_vec)
        grad_c = wp.dot(J_c, body_qd_c) + wp.dot(J_p, body_qd_p)
        constraint_error = wp.dot(axis_p_w, b2_c_w)
        compliance = joint_angular_compliance[joint_idx]
    else:
        return

    # --- Update global system components ---
    h_j[global_constraint_idx] = grad_c + bias_scale * constraint_error

    lambda_current = lambda_j[global_constraint_idx]
    wp.atomic_add(g, child_idx, -J_c * lambda_current)
    wp.atomic_add(g, parent_idx, -J_p * lambda_current)

    C_j_values[global_constraint_idx] = compliance

    J_j_values[global_constraint_idx, 0] = J_p
    J_j_values[global_constraint_idx, 1] = J_c


@wp.kernel
def linesearch_joint_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
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
    # --- Outputs ---
    res_buffer: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, joint_idx = wp.tid()
    j_type = joint_type[joint_idx]

    # Early exit for disabled or non-revolute joints
    if (
        joint_enabled[joint_idx] == 0
        or j_type != wp.sim.JOINT_REVOLUTE
        or joint_parent[joint_idx] < 0  # To-do: Handle world joints
    ):
        return

    alpha = alphas[alpha_idx]

    child_idx = joint_child[joint_idx]
    parent_idx = joint_parent[joint_idx]

    # Kinematics (Child)
    body_q_c = body_q[child_idx]
    X_wj_c = body_q_c * joint_X_c[joint_idx]
    r_c = wp.transform_get_translation(X_wj_c) - wp.transform_point(
        body_q_c, body_com[child_idx]
    )
    q_c_rot = wp.transform_get_rotation(body_q_c)

    # Kinematics (Parent)
    body_q_p = body_q[parent_idx]
    X_wj_p = body_q_p * joint_X_p[joint_idx]
    r_p = wp.transform_get_translation(X_wj_p) - wp.transform_point(
        body_q_p, body_com[parent_idx]
    )
    q_p_rot = wp.transform_get_rotation(body_q_p)

    # Joint Axis in World Frame
    axis = joint_axis[joint_axis_start[joint_idx]]
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
    body_qd_c = body_qd[child_idx] + alpha * delta_body_qd[child_idx]
    body_qd_p = body_qd[parent_idx] + alpha * delta_body_qd[parent_idx]
    C_dot_pos_x = wp.dot(J_pos_x_c, body_qd_c) + wp.dot(J_pos_x_p, body_qd_p)
    C_dot_pos_y = wp.dot(J_pos_y_c, body_qd_c) + wp.dot(J_pos_y_p, body_qd_p)
    C_dot_pos_z = wp.dot(J_pos_z_c, body_qd_c) + wp.dot(J_pos_z_p, body_qd_p)
    C_dot_rot_u = wp.dot(J_rot_u_c, body_qd_c) + wp.dot(J_rot_u_p, body_qd_p)
    C_dot_rot_v = wp.dot(J_rot_v_c, body_qd_c) + wp.dot(J_rot_v_p, body_qd_p)

    # --- Update global system components ---
    # 1. Update g (momentum balance residual: -J^T * lambda)
    base_lambda_idx = lambda_j_offset + joint_idx * 5
    lambda_j_x = (
        _lambda[base_lambda_idx + 0] + alpha * delta_lambda[base_lambda_idx + 0]
    )
    lambda_j_y = (
        _lambda[base_lambda_idx + 1] + alpha * delta_lambda[base_lambda_idx + 1]
    )
    lambda_j_z = (
        _lambda[base_lambda_idx + 2] + alpha * delta_lambda[base_lambda_idx + 2]
    )
    lambda_j_u = (
        _lambda[base_lambda_idx + 3] + alpha * delta_lambda[base_lambda_idx + 3]
    )
    lambda_j_v = (
        _lambda[base_lambda_idx + 4] + alpha * delta_lambda[base_lambda_idx + 4]
    )

    g_c = (
        -J_pos_x_c * lambda_j_x
        - J_pos_y_c * lambda_j_y
        - J_pos_z_c * lambda_j_z
        - J_rot_u_c * lambda_j_u
        - J_rot_v_c * lambda_j_v
    )
    for i in range(wp.static(6)):
        st_i = wp.static(i)
        res_buffer[alpha_idx, child_idx * 6 + st_i] += g_c[st_i]

    g_p = (
        -J_pos_x_p * lambda_j_x
        - J_pos_y_p * lambda_j_y
        - J_pos_z_p * lambda_j_z
        - J_rot_u_p * lambda_j_u
        - J_rot_v_p * lambda_j_v
    )
    for i in range(wp.static(6)):
        st_i = wp.static(i)
        res_buffer[alpha_idx, parent_idx * 6 + st_i] += g_p[st_i]

    # 2. Residual Vector h (constraint violation)
    bias_scale = joint_stabilization_factor / dt
    h_x = C_dot_pos_x + bias_scale * C_pos.x
    h_y = C_dot_pos_y + bias_scale * C_pos.y
    h_z = C_dot_pos_z + bias_scale * C_pos.z
    h_u = C_dot_rot_u + bias_scale * C_rot_u
    h_v = C_dot_rot_v + bias_scale * C_rot_v
    h_sq_sum = (
        wp.pow(h_x, 2.0)
        + wp.pow(h_y, 2.0)
        + wp.pow(h_z, 2.0)
        + wp.pow(h_u, 2.0)
        + wp.pow(h_v, 2.0)
    )

    buff_idx = get_random_idx_to_res_buffer(alpha_idx + joint_idx)
    res_buffer[alpha_idx, buff_idx] += h_sq_sum
