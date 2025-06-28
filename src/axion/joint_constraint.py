import warp as wp


@wp.func
def orthogonal_basis(axis: wp.vec3):
    """
    Returns an orthogonal basis for the given axis.
    The basis is returned as a tuple of two orthogonal vectors.
    """
    if wp.abs(axis.x) < wp.abs(axis.y):
        b1 = wp.vec3(-axis.z, 0.0, axis.x)
    else:
        b1 = wp.vec3(0.0, axis.z, -axis.y)
    b1 = wp.normalize(b1)
    b2 = wp.cross(axis, b1)
    return b1, b2


@wp.kernel
def joint_constraint_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.bool),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    # Velocity impulse variables
    lambda_j_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    # Offsets for output arrays
    h_j_offset: wp.int32,
    J_j_offset: wp.int32,
    C_j_offset: wp.int32,
    # Output arrays
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    j_type = joint_type[tid]

    # Return early if the joint is not enabled
    if joint_enabled[tid] == 0:
        return

    # Currently support only revolute joints
    if j_type != wp.sim.JOINT_TYPE_REVOLUTE:
        return

    # Joint must have a parent
    if joint_parent[tid] < 0:
        return

    child_idx = joint_child[tid]
    parent_idx = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    # I can do this only for revolute joints (only one axis)
    axis = joint_axis[joint_axis_start[tid]]

    # ------- Child -------
    body_q_c = body_q[child_idx]
    body_com_c = body_com[child_idx]
    X_wj_c = body_q_c * X_cj  # X_wc * X_cj
    r_c = X_wj_c.translation - wp.transform_point(body_q_c, body_com_c)

    # Compute the joint basis in the world frame
    q_c = wp.transform_get_rotation(body_q_c)
    b1, b2 = orthogonal_basis(axis)
    b1_w = wp.quat_rotate(q_c, b1)
    b2_w = wp.quat_rotate(q_c, b2)

    # ------- Parent -------
    body_q_p = body_q[parent_idx]
    body_com_p = body_com[parent_idx]
    X_wj_p = body_q_p * X_pj  # X_wp * X_pj
    r_p = X_wj_p.translation - wp.transform_point(body_q_p, body_com_p)

    # Compute the joint axis in the world frame
    q_p = wp.transform_get_rotation(body_q_p)
    axis_w = wp.quat_rotate(q_p, axis)

    # Revolute joint constraints 5 degrees of freedom:
    # 1.(x) Position constraint in x direction: X_wj_c.x - X_wj_p.x = 0
    # 2.(y) Position constraint in y direction: X_wj_c.y - X_wj_p.y = 0
    # 3.(z) Position constraint in z direction: X_wj_c.z - X_wj_p.z = 0
    # 4.(u) Orientation constraint: wp.dot(axis_w, b1_w) = 0
    # 5.(v) Orientation constraint: wp.dot(axis_w, b2_w) = 0

    zero_vec = wp.vec3(0.0, 0.0, 0.0)
    b1_x_axis = wp.cross(b1_w, axis_w)
    b2_x_axis = wp.cross(b2_w, axis_w)

    # --- Compute the Jacobian for the position constraints ---
    # Child
    J_pos_x_c = wp.spatial_vector(0.0, r_c[2], -r_c[1], 1.0, 0.0, 0.0)
    J_pos_y_c = wp.spatial_vector(-r_c[2], 0.0, r_c[0], 0.0, 1.0, 0.0)
    J_pos_z_c = wp.spatial_vector(r_c[1], -r_c[0], 0.0, 0.0, 0.0, 1.0)

    J_rot_u_c = wp.spatial_vector(b1_x_axis, zero_vec)
    J_rot_v_c = wp.spatial_vector(b2_x_axis, zero_vec)

    # Parent
    J_pos_x_p = wp.spatial_vector(0.0, -r_p[2], r_p[1], -1.0, 0.0, 0.0)
    J_pos_y_p = wp.spatial_vector(r_p[2], 0.0, -r_p[0], 0.0, -1.0, 0.0)
    J_pos_z_p = wp.spatial_vector(-r_p[1], r_p[0], 0.0, 0.0, 0.0, -1.0)

    J_rot_u_p = wp.spatial_vector((-1) * b1_x_axis, zero_vec)
    J_rot_v_p = wp.spatial_vector((-1) * b2_x_axis, zero_vec)

    base_lambda_idx = lambda_j_offset + tid * 5
    lambda_j_x = _lambda[base_lambda_idx + 0]
    lambda_j_y = _lambda[base_lambda_idx + 1]
    lambda_j_z = _lambda[base_lambda_idx + 2]
    lambda_j_u = _lambda[base_lambda_idx + 3]
    lambda_j_v = _lambda[base_lambda_idx + 4]

    # --- g --- (momentum balance)

    # Get body indices once
    child_body_idx = joint_child[tid]
    parent_body_idx = joint_parent[tid]

    # Accumulate forces for CHILD body
    g_c = (
        -J_pos_x_c * lambda_j_x
        - J_pos_y_c * lambda_j_y
        - J_pos_z_c * lambda_j_z
        - J_rot_u_c * lambda_j_u
        - J_rot_v_c * lambda_j_v
    )
    for i in range(wp.static(6)):
        wp.atomic_add(g, child_body_idx * 6 + i, g_c[i])

    # Accumulate forces for PARENT body
    g_p = (
        -J_pos_x_p * lambda_j_x
        - J_pos_y_p * lambda_j_y
        - J_pos_z_p * lambda_j_z
        - J_rot_u_p * lambda_j_u
        - J_rot_v_p * lambda_j_v
    )
    for i in range(wp.static(6)):
        wp.atomic_add(g, parent_body_idx * 6 + i, g_p[i])

    # --- h --- (vector of the constraint errors)
    h_x = X_wj_c.translation.x - X_wj_p.translation.x
    h_y = X_wj_c.translation.y - X_wj_p.translation.y
    h_z = X_wj_c.translation.z - X_wj_p.translation.z
    h_u = wp.dot(axis_w, b1_w)
    h_v = wp.dot(axis_w, b2_w)

    # Each joint (tid) is responsible for 5 constraints. Calculate the base index.
    base_h_idx = h_j_offset + tid * 5
    base_C_idx = C_j_offset + tid * 5
    base_J_idx = J_j_offset + tid * 5

    # --- h --- (vector of the constraint errors)
    h[base_h_idx + 0] = h_x
    h[base_h_idx + 1] = h_y
    h[base_h_idx + 2] = h_z
    h[base_h_idx + 3] = h_u
    h[base_h_idx + 4] = h_v

    # --- C --- (compliance block)
    C_values[base_C_idx + 0] = joint_linear_compliance[tid]
    C_values[base_C_idx + 1] = joint_linear_compliance[tid]
    C_values[base_C_idx + 2] = joint_linear_compliance[tid]
    C_values[base_C_idx + 3] = joint_angular_compliance[tid]
    C_values[base_C_idx + 4] = joint_angular_compliance[tid]

    # --- J --- (Jacobian block)
    J_values[base_J_idx + 0, 0] = J_pos_x_c
    J_values[base_J_idx + 1, 0] = J_pos_y_c
    J_values[base_J_idx + 2, 0] = J_pos_z_c
    J_values[base_J_idx + 3, 0] = J_rot_u_c
    J_values[base_J_idx + 4, 0] = J_rot_v_c

    J_values[base_J_idx + 0, 1] = J_pos_x_p
    J_values[base_J_idx + 1, 1] = J_pos_y_p
    J_values[base_J_idx + 2, 1] = J_pos_z_p
    J_values[base_J_idx + 3, 1] = J_rot_u_p
    J_values[base_J_idx + 4, 1] = J_rot_v_p
