import warp as wp


@wp.func
def orthogonal_basis(axis: wp.vec3):
    # Choose v as the unit vector along the axis with the smallest absolute component
    if wp.abs(axis.x) <= wp.abs(axis.y) and wp.abs(axis.x) <= wp.abs(axis.z):
        v = wp.vec3(1.0, 0.0, 0.0)
    elif wp.abs(axis.y) <= wp.abs(axis.z):
        v = wp.vec3(0.0, 1.0, 0.0)
    else:
        v = wp.vec3(0.0, 0.0, 1.0)

    # Compute b1 as the normalized cross product of axis and v
    b1 = wp.normalize(wp.cross(axis, v))

    # Compute b2 as the cross product of axis and b1
    b2 = wp.cross(axis, b1)

    return b1, b2


@wp.kernel
def joint_constraint_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
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
    lambda_j_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    h_j_offset: wp.int32,
    J_j_offset: wp.int32,
    C_j_offset: wp.int32,
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    j_type = joint_type[tid]

    if (
        joint_enabled[tid] == 0
        or j_type != wp.sim.JOINT_REVOLUTE
        or joint_parent[tid] < 0
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

    # Define orthogonal basis in child's local frame (assuming axis is same in child's frame at reference)
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

    # Jacobian Calculation (Rotational) with corrected signs
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)  # Correct order for child Jacobian
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)  # Correct order for child Jacobian
    zero_vec = wp.vec3()

    # Child Jacobian is -cross(axis, basis_vec)
    J_rot_u_c = wp.spatial_vector(-b1_x_axis, zero_vec)
    J_rot_v_c = wp.spatial_vector(-b2_x_axis, zero_vec)

    # Parent Jacobian is +cross(axis, basis_vec)
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

    # Residual Vector h
    bias_scale = joint_stabilization_factor / dt
    base_h_idx = h_j_offset + tid * 5
    h[base_h_idx + 0] = C_dot_pos_x + bias_scale * C_pos.x
    h[base_h_idx + 1] = C_dot_pos_y + bias_scale * C_pos.y
    h[base_h_idx + 2] = C_dot_pos_z + bias_scale * C_pos.z
    h[base_h_idx + 3] = C_dot_rot_u + bias_scale * C_rot_u
    h[base_h_idx + 4] = C_dot_rot_v + bias_scale * C_rot_v

    # Update g (momentum balance)
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

    # Compliance (C_values)
    base_C_idx = C_j_offset + tid * 5
    C_values[base_C_idx + 0] = joint_linear_compliance[tid]
    C_values[base_C_idx + 1] = joint_linear_compliance[tid]
    C_values[base_C_idx + 2] = joint_linear_compliance[tid]
    C_values[base_C_idx + 3] = joint_angular_compliance[tid]
    C_values[base_C_idx + 4] = joint_angular_compliance[tid]

    # Jacobian (J_values)
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
