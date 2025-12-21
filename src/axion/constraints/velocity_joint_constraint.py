import warp as wp
from .joint_kinematics import (
    compute_joint_transforms,
    get_linear_component,
    get_angular_component,
    get_revolute_angular_component,
)


@wp.func
def submit_velocity_component(
    # --- Constraint Data ---
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: wp.float32, # This is position error "C(x)"
    # --- Identifiers ---
    world_idx: wp.int32,
    constraint_idx: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- System State ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    lambda_j = body_lambda_j[world_idx, constraint_idx]

    # Get Velocities
    u_c = body_u[world_idx, c_idx]
    u_p = wp.spatial_vector()
    if p_idx >= 0:
        u_p = body_u[world_idx, p_idx]

    # Compute Relative Velocity along Constraint: J * u
    # J_p is for parent, J_c is for child
    v_j = wp.dot(J_c, u_c) + wp.dot(J_p, u_p)

    # Apply Forces / Impulses
    # Force = -J^T * lambda
    # Impulse = Force * dt
    # h_d accumulates these impulses
    
    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, -dt * J_p * lambda_j)
    wp.atomic_add(h_d, world_idx, c_idx, -dt * J_c * lambda_j)

    # Compute Residual
    # residual = v_rel + (bias) + compliance * lambda
    # bias = (upsilon / dt) * error
    
    bias = (upsilon / dt) * error
    h_j[world_idx, constraint_idx] = v_j + bias + dt * compliance * lambda_j

    # Store for Linear Solver
    J_hat_j_values[world_idx, constraint_idx, 0] = J_p
    J_hat_j_values[world_idx, constraint_idx, 1] = J_c

    C_j_values[world_idx, constraint_idx] = compliance


@wp.kernel
def velocity_joint_constraint_kernel(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    # --- Joint Definition ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Parameters ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    j_type = joint_type[world_idx, joint_idx]
    
    # Check if active
    if joint_enabled[world_idx, joint_idx] == 0:
        return
        
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Kinematics
    # Note: We need transforms for both parent and child
    
    # Child
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx]
    )
    
    # Parent
    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        
    X_w_p, r_p, pos_p = compute_joint_transforms(
        X_body_p,
        com_p,
        joint_X_p[world_idx, joint_idx]
    )

    # -----------------------------------------------------------
    # Solve based on Type
    # -----------------------------------------------------------
    
    # === REVOLUTE (1) or BALL (2) or FIXED (3) ===
    # All have 3 linear constraints
    if j_type == 1 or j_type == 2 or j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_velocity_component(
                J_p, J_c, err,
                world_idx, start_offset + i, p_idx, c_idx,
                body_u, body_lambda_j,
                dt, upsilon, compliance,
                h_d, h_j, J_hat_j_values, C_j_values
            )

    # === REVOLUTE (1) ===
    # 2 Angular Constraints
    if j_type == 1:
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]
        
        # Ortho 1
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 0)
        submit_velocity_component(
            J_p, J_c, err,
            world_idx, start_offset + 3, p_idx, c_idx,
            body_u, body_lambda_j,
            dt, upsilon, compliance,
            h_d, h_j, J_hat_j_values, C_j_values
        )
        
        # Ortho 2
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 1)
        submit_velocity_component(
            J_p, J_c, err,
            world_idx, start_offset + 4, p_idx, c_idx,
            body_u, body_lambda_j,
            dt, upsilon, compliance,
            h_d, h_j, J_hat_j_values, C_j_values
        )

    # === FIXED (3) ===
    # 3 Angular Constraints (Lock Rotation)
    if j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            # Row index starts after linear (3)
            submit_velocity_component(
                J_p, J_c, err,
                world_idx, start_offset + 3 + i, p_idx, c_idx,
                body_u, body_lambda_j,
                dt, upsilon, compliance,
                h_d, h_j, J_hat_j_values, C_j_values
            )


# ---------------------------------------------------------------------------- #
#                               BATCHED VERSION                                #
# ---------------------------------------------------------------------------- #

@wp.func
def submit_batch_velocity_component(
    # --- Constraint Data ---
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: wp.float32,
    # --- Identifiers ---
    batch_idx: wp.int32,
    world_idx: wp.int32,
    constraint_idx: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- System State ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    # --- Params ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
):
    lambda_j = body_lambda_j[batch_idx, world_idx, constraint_idx]

    u_c = body_u[batch_idx, world_idx, c_idx]
    u_p = wp.spatial_vector()
    if p_idx >= 0:
        u_p = body_u[batch_idx, world_idx, p_idx]

    v_j = wp.dot(J_c, u_c) + wp.dot(J_p, u_p)
    
    if p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, p_idx, -dt * J_p * lambda_j)
    wp.atomic_add(h_d, batch_idx, world_idx, c_idx, -dt * J_c * lambda_j)

    bias = (upsilon / dt) * error
    h_j[batch_idx, world_idx, constraint_idx] = v_j + bias + dt * compliance * lambda_j


@wp.kernel
def batch_velocity_joint_residual_kernel(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2), # [World, Body] - Shared
    body_com: wp.array(dtype=wp.vec3, ndim=2),    # [World, Body] - Shared
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    # --- Joint Definition (Shared) ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Parameters ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, joint_idx = wp.tid()

    j_type = joint_type[world_idx, joint_idx]
    
    if joint_enabled[world_idx, joint_idx] == 0:
        return
        
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Kinematics (Shared across batches)
    
    # Child
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx]
    )
    
    # Parent
    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        
    X_w_p, r_p, pos_p = compute_joint_transforms(
        X_body_p,
        com_p,
        joint_X_p[world_idx, joint_idx]
    )

    # === LINEAR (XYZ) ===
    if j_type == 1 or j_type == 2 or j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_batch_velocity_component(
                J_p, J_c, err,
                batch_idx, world_idx, start_offset + i, p_idx, c_idx,
                body_u, body_lambda_j,
                dt, upsilon, compliance,
                h_d, h_j
            )

    # === REVOLUTE (Angular) ===
    if j_type == 1:
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]
        
        # Ortho 1
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 0)
        submit_batch_velocity_component(
            J_p, J_c, err,
            batch_idx, world_idx, start_offset + 3, p_idx, c_idx,
            body_u, body_lambda_j,
            dt, upsilon, compliance,
            h_d, h_j
        )
        # Ortho 2
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 1)
        submit_batch_velocity_component(
            J_p, J_c, err,
            batch_idx, world_idx, start_offset + 4, p_idx, c_idx,
            body_u, body_lambda_j,
            dt, upsilon, compliance,
            h_d, h_j
        )

    # === FIXED (Angular) ===
    if j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            submit_batch_velocity_component(
                J_p, J_c, err,
                batch_idx, world_idx, start_offset + 3 + i, p_idx, c_idx,
                body_u, body_lambda_j,
                dt, upsilon, compliance,
                h_d, h_j
            )