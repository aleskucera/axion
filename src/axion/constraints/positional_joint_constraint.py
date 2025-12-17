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


@wp.func
def submit_constraint_contribution(
    # --- Computed Values ---
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: wp.float32,
    _lambda: wp.float32,
    # --- Identifiers ---
    world_idx: wp.int32,
    global_constraint_index: wp.int32,  # The unique ID of this constraint in the whole system
    body_p_idx: wp.int32,
    body_c_idx: wp.int32,
    # --- Constants ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- System Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    # 1. Apply forces to bodies (atomic add required as bodies are shared)
    if body_p_idx >= 0:
        wp.atomic_add(h_d, world_idx, body_p_idx, -J_p * _lambda * dt)

    wp.atomic_add(h_d, world_idx, body_c_idx, -J_c * _lambda * dt)

    # 2. Write the constraint residual (h_j)
    # The solver tries to drive (error + compliance * lambda) / dt -> 0
    h_j[world_idx, global_constraint_index] = (error + compliance * _lambda) / dt

    # 3. Store the Jacobians for the linear solver step
    J_hat_j_values[world_idx, global_constraint_index, 0] = J_p
    J_hat_j_values[world_idx, global_constraint_index, 1] = J_c

    # 4. Store Compliance (inverse stiffness)
    C_j_values[world_idx, global_constraint_index] = compliance / (dt * dt)


@wp.func
def compute_joint_kinematics(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Indices ---
    world_idx: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Local Data ---
    X_p_local: wp.transform,
    X_c_local: wp.transform,
):
    # --- Parent Side ---
    X_body_p = wp.transform_identity()
    com_p_world = wp.vec3(0.0, 0.0, 0.0)

    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p_world = wp.transform_point(X_body_p, body_com[world_idx, p_idx])

    # Joint frame in world space
    X_wp = X_body_p * X_p_local
    pos_p_world = wp.transform_get_translation(X_wp)

    # Lever arm (vector from Center of Mass to Joint Anchor)
    # Be careful with static bodies (p_idx < 0): r_p is effectively just pos_p_world
    r_p = pos_p_world - com_p_world

    # --- Child Side ---
    X_body_c = body_q[world_idx, c_idx]
    com_c_world = wp.transform_point(X_body_c, body_com[world_idx, c_idx])

    X_wc = X_body_c * X_c_local
    pos_c_world = wp.transform_get_translation(X_wc)

    r_c = pos_c_world - com_c_world

    return X_wp, X_wc, r_p, r_c, pos_p_world, pos_c_world


@wp.func
def solve_linear_constraint(
    # --- Context ---
    world_idx: wp.int32,
    global_constraint_index: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Geometry ---
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    axis_idx: wp.int32,  # 0=X, 1=Y, 2=Z
    # --- System ---
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
):
    # 1. Define the Global Axis we are constraining (X, Y, or Z)
    axis_vec = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        axis_vec = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        axis_vec = wp.vec3(0.0, 1.0, 0.0)
    else:
        axis_vec = wp.vec3(0.0, 0.0, 1.0)

    # 2. Compute Jacobian: J = [n, r x n]
    # This represents: Linear push along 'axis_vec' + Torque needed at COM
    J_c = wp.spatial_vector(axis_vec, wp.cross(r_c, axis_vec))

    J_p = wp.spatial_vector()  # Default zero
    if p_idx >= 0:
        J_p = wp.spatial_vector(-axis_vec, -wp.cross(r_p, axis_vec))

    # 3. Compute Error: Simple 1D distance along the axis
    # We want (pos_c - pos_p) to be 0
    delta = pos_c - pos_p
    error = delta[axis_idx]

    # 4. Submit
    current_lambda = body_lambda_j[world_idx, global_constraint_index]

    submit_constraint_contribution(
        J_p,
        J_c,
        error,
        current_lambda,
        world_idx,
        global_constraint_index,
        p_idx,
        c_idx,
        dt,
        compliance,
        h_d,
        h_j,
        J_hat_j_values,
        C_j_values,
    )


@wp.func
def solve_angular_constraint(
    # --- Context ---
    world_idx: wp.int32,
    global_constraint_index: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Geometry ---
    X_wp: wp.transform,
    X_wc: wp.transform,
    axis_local: wp.vec3,  # The hinge axis in local space
    ortho_idx: wp.int32,  # 0 or 1 (First or Second orthogonal basis vector)
    # --- System ---
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
):
    # 1. Transform Key Vectors to World Space
    # Parent Frame is the "Target", Child Frame is "Current"
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # The axis we want to align (The Hinge Axis)
    axis_p_world = wp.quat_rotate(q_p, axis_local)

    # The basis vectors perpendicular to the hinge
    b1_local, b2_local = orthogonal_basis(axis_local)

    # We check if the Child's basis vectors are orthogonal to the Parent's axis
    # If aligned, dot(axis_p, b_child) == 0.
    target_basis_world = wp.vec3()
    if ortho_idx == 0:
        target_basis_world = wp.quat_rotate(q_c, b1_local)
    else:
        target_basis_world = wp.quat_rotate(q_c, b2_local)

    # 2. Compute Error (Dot product should be zero)
    error = wp.dot(axis_p_world, target_basis_world)

    # 3. Compute Jacobian: J = [0, axis_of_rotation]
    # The rotation axis needed to fix this error is perpendicular to both vectors
    rot_axis = wp.cross(axis_p_world, target_basis_world)

    J_c = wp.spatial_vector(wp.vec3(), -rot_axis)

    J_p = wp.spatial_vector()
    if p_idx >= 0:
        J_p = wp.spatial_vector(wp.vec3(), rot_axis)

    # 4. Submit
    current_lambda = body_lambda_j[world_idx, global_constraint_index]

    submit_constraint_contribution(
        J_p,
        J_c,
        error,
        current_lambda,
        world_idx,
        global_constraint_index,
        p_idx,
        c_idx,
        dt,
        compliance,
        h_d,
        h_j,
        J_hat_j_values,
        C_j_values,
    )


@wp.kernel
def positional_joint_constraint_kernel(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    # --- Body Definition ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Joint Definition Inputs ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- Solver State ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    j_type = joint_type[world_idx, joint_idx]

    # 1. Fetch Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    constraint_start_idx = constraint_offsets[world_idx, joint_idx]

    # 2. Compute Kinematics (Done once per joint, registers reused)
    X_wp, X_wc, r_p, r_c, pos_p, pos_c = compute_joint_kinematics(
        body_q,
        body_com,
        world_idx,
        p_idx,
        c_idx,
        joint_X_p[world_idx, joint_idx],
        joint_X_c[world_idx, joint_idx],
    )

    # 3. Apply Constraints based on Joint Type

    # === LINEAR PART ===
    # Both REVOLUTE(1) and BALL(2) constrain X, Y, Z translation
    if j_type == 1 or j_type == 2:
        for local_constraint_idx in range(wp.static(3)):
            solve_linear_constraint(
                world_idx,
                constraint_start_idx + wp.static(local_constraint_idx),
                p_idx,
                c_idx,
                r_p,
                r_c,
                pos_p,
                pos_c,
                wp.static(local_constraint_idx),  # Axis Index (0=X, 1=Y, 2=Z)
                body_lambda_j,
                h_d,
                h_j,
                J_hat_j_values,
                C_j_values,
                dt,
                compliance,
            )

    # === ANGULAR PART ===
    # REVOLUTE(1) constrains rotation around 2 axes
    if j_type == 1:
        # Get the Hinge Axis
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

        # Apply 2 rotational constraints (Start index is +3 after linear)
        # First Orthogonal Vector
        solve_angular_constraint(
            world_idx,
            constraint_start_idx + 3,
            p_idx,
            c_idx,
            X_wp,
            X_wc,
            axis_local,
            0,
            body_lambda_j,
            h_d,
            h_j,
            J_hat_j_values,
            C_j_values,
            dt,
            compliance,
        )
        # Second Orthogonal Vector
        solve_angular_constraint(
            world_idx,
            constraint_start_idx + 4,
            p_idx,
            c_idx,
            X_wp,
            X_wc,
            axis_local,
            1,
            body_lambda_j,
            h_d,
            h_j,
            J_hat_j_values,
            C_j_values,
            dt,
            compliance,
        )


# ---------------------------------------------------------------------------- #
#                           Batch Helper Functions                             #
# ---------------------------------------------------------------------------- #


@wp.func
def submit_batch_residual(
    # --- Computed Values (Local) ---
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: wp.float32,
    _lambda: wp.float32,
    # --- Identifiers ---
    batch_idx: wp.int32,
    world_idx: wp.int32,
    global_constraint_index: wp.int32,
    body_p_idx: wp.int32,
    body_c_idx: wp.int32,
    # --- Constants ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- System Outputs (3D) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
):
    # 1. Apply forces to bodies (Atomic add is required)
    # Force = -J^T * lambda * dt
    if body_p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_p_idx, -J_p * _lambda * dt)

    wp.atomic_add(h_d, batch_idx, world_idx, body_c_idx, -J_c * _lambda * dt)

    # 2. Write the constraint residual (h_j)
    # Residual = (Error + Compliance * lambda) / dt
    h_j[batch_idx, world_idx, global_constraint_index] = (error + compliance * _lambda) / dt


@wp.func
def compute_batch_joint_kinematics(
    # --- State (3D) ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    # --- Model Data (2D) ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Indices ---
    batch_idx: wp.int32,
    world_idx: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Local Data ---
    X_p_local: wp.transform,
    X_c_local: wp.transform,
):
    # --- Parent Side ---
    X_body_p = wp.transform_identity()
    com_p_world = wp.vec3(0.0, 0.0, 0.0)

    if p_idx >= 0:
        # Access 3D body_q
        X_body_p = body_q[batch_idx, world_idx, p_idx]
        com_p_world = wp.transform_point(X_body_p, body_com[world_idx, p_idx])

    X_wp = X_body_p * X_p_local
    pos_p_world = wp.transform_get_translation(X_wp)
    r_p = pos_p_world - com_p_world

    # --- Child Side ---
    # Access 3D body_q
    X_body_c = body_q[batch_idx, world_idx, c_idx]
    com_c_world = wp.transform_point(X_body_c, body_com[world_idx, c_idx])

    X_wc = X_body_c * X_c_local
    pos_c_world = wp.transform_get_translation(X_wc)
    r_c = pos_c_world - com_c_world

    return X_wp, X_wc, r_p, r_c, pos_p_world, pos_c_world


@wp.func
def solve_batch_linear_residual(
    # --- Context ---
    batch_idx: wp.int32,
    world_idx: wp.int32,
    global_constraint_index: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Geometry ---
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    axis_idx: wp.int32,
    # --- System ---
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
    dt: wp.float32,
    compliance: wp.float32,
):
    # 1. Define Axis
    axis_vec = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        axis_vec = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        axis_vec = wp.vec3(0.0, 1.0, 0.0)
    else:
        axis_vec = wp.vec3(0.0, 0.0, 1.0)

    # 2. Compute Jacobian (Locally for this batch instance)
    J_c = wp.spatial_vector(axis_vec, wp.cross(r_c, axis_vec))
    J_p = wp.spatial_vector()
    if p_idx >= 0:
        J_p = wp.spatial_vector(-axis_vec, -wp.cross(r_p, axis_vec))

    # 3. Compute Error
    delta = pos_c - pos_p
    error = delta[axis_idx]

    # 4. Submit
    current_lambda = body_lambda_j[batch_idx, world_idx, global_constraint_index]

    submit_batch_residual(
        J_p,
        J_c,
        error,
        current_lambda,
        batch_idx,
        world_idx,
        global_constraint_index,
        p_idx,
        c_idx,
        dt,
        compliance,
        h_d,
        h_j,
    )


@wp.func
def solve_batch_angular_residual(
    # --- Context ---
    batch_idx: wp.int32,
    world_idx: wp.int32,
    global_constraint_index: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- Geometry ---
    X_wp: wp.transform,
    X_wc: wp.transform,
    axis_local: wp.vec3,
    ortho_idx: wp.int32,
    # --- System ---
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
    dt: wp.float32,
    compliance: wp.float32,
):
    # 1. Geometry
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    axis_p_world = wp.quat_rotate(q_p, axis_local)
    b1_local, b2_local = orthogonal_basis(axis_local)

    target_basis_world = wp.vec3()
    if ortho_idx == 0:
        target_basis_world = wp.quat_rotate(q_c, b1_local)
    else:
        target_basis_world = wp.quat_rotate(q_c, b2_local)

    error = wp.dot(axis_p_world, target_basis_world)
    rot_axis = wp.cross(axis_p_world, target_basis_world)

    J_c = wp.spatial_vector(wp.vec3(), -rot_axis)
    J_p = wp.spatial_vector()
    if p_idx >= 0:
        J_p = wp.spatial_vector(wp.vec3(), rot_axis)

    # 2. Submit
    current_lambda = body_lambda_j[batch_idx, world_idx, global_constraint_index]

    submit_batch_residual(
        J_p,
        J_c,
        error,
        current_lambda,
        batch_idx,
        world_idx,
        global_constraint_index,
        p_idx,
        c_idx,
        dt,
        compliance,
        h_d,
        h_j,
    )


# ---------------------------------------------------------------------------- #
#                           Main Batch Kernel                                  #
# ---------------------------------------------------------------------------- #


@wp.kernel
def batch_positional_joint_residual_kernel(
    # --- State (Batched) ---
    body_q: wp.array(dtype=wp.transform, ndim=3),  # [Batch, World, Body]
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),  # [Batch, World, Constraint]
    # --- Model Data (Shared) ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Joint Definition (Shared) ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- Outputs (Batched) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
    h_j: wp.array(dtype=wp.float32, ndim=3),  # [Batch, World, Constraint]
):
    # Grid: (batch_size, num_worlds, num_joints)
    batch_idx, world_idx, joint_idx = wp.tid()

    j_type = joint_type[world_idx, joint_idx]

    # 1. Fetch Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    constraint_start_idx = constraint_offsets[world_idx, joint_idx]

    # 2. Compute Kinematics for this specific Batch & Joint
    X_wp, X_wc, r_p, r_c, pos_p, pos_c = compute_batch_joint_kinematics(
        body_q,
        body_com,
        batch_idx,
        world_idx,
        p_idx,
        c_idx,
        joint_X_p[world_idx, joint_idx],
        joint_X_c[world_idx, joint_idx],
    )

    # 3. Apply Constraints

    # === LINEAR ===
    if j_type == 1 or j_type == 2:
        for local_constraint_idx in range(wp.static(3)):
            solve_batch_linear_residual(
                batch_idx,
                world_idx,
                constraint_start_idx + wp.static(local_constraint_idx),
                p_idx,
                c_idx,
                r_p,
                r_c,
                pos_p,
                pos_c,
                wp.static(local_constraint_idx),
                body_lambda_j,
                h_d,
                h_j,
                dt,
                compliance,
            )

    # === ANGULAR ===
    if j_type == 1:
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

        solve_batch_angular_residual(
            batch_idx,
            world_idx,
            constraint_start_idx + 3,
            p_idx,
            c_idx,
            X_wp,
            X_wc,
            axis_local,
            0,
            body_lambda_j,
            h_d,
            h_j,
            dt,
            compliance,
        )

        solve_batch_angular_residual(
            batch_idx,
            world_idx,
            constraint_start_idx + 4,
            p_idx,
            c_idx,
            X_wp,
            X_wc,
            axis_local,
            1,
            body_lambda_j,
            h_d,
            h_j,
            dt,
            compliance,
        )
