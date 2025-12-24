import warp as wp
import numpy as np
from axion.constraints.joint_kinematics import get_linear_component, get_angular_component, compute_joint_transforms
from axion.constraints.track_curve import track_project, track_get_frame

# Define internal types for Axion Equality Constraints
EQ_TYPE_CONNECT = 0
EQ_TYPE_WELD = 1
EQ_TYPE_JOINT = 2  # Not implemented positionally
EQ_TYPE_TRACK = 4

@wp.func
def submit_equality_constraint_component(
    # --- Constraint Data ---
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: wp.float32,
    # --- Identifiers ---
    world_idx: wp.int32,
    constraint_idx: wp.int32,
    p_idx: wp.int32,
    c_idx: wp.int32,
    # --- System Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_eq: wp.array(dtype=wp.float32, ndim=2),
    J_hat_eq_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_eq_values: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_eq: wp.array(dtype=wp.float32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
):
    current_lambda = body_lambda_eq[world_idx, constraint_idx]

    # 1. Apply forces to bodies (atomic add required as bodies are shared)
    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, -J_p * current_lambda * dt)

    if c_idx >= 0:
        wp.atomic_add(h_d, world_idx, c_idx, -J_c * current_lambda * dt)

    # 2. Write the constraint residual (h_eq)
    # The solver tries to drive (error + compliance * lambda) / dt -> 0
    h_eq[world_idx, constraint_idx] = (error + compliance * current_lambda) / dt

    # 3. Store the Jacobians for the linear solver step
    J_hat_eq_values[world_idx, constraint_idx, 0] = J_p
    J_hat_eq_values[world_idx, constraint_idx, 1] = J_c

    # 4. Store Compliance (inverse stiffness)
    C_eq_values[world_idx, constraint_idx] = compliance / (dt * dt)

@wp.kernel
def positional_equality_constraint_kernel(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_lambda_eq: wp.array(dtype=wp.float32, ndim=2),
    # --- Body Definition ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Equality Constraint Definition Inputs ---
    eq_type: wp.array(dtype=wp.int32, ndim=2),
    eq_body1: wp.array(dtype=wp.int32, ndim=2),
    eq_body2: wp.array(dtype=wp.int32, ndim=2),
    eq_anchor: wp.array(dtype=wp.vec3, ndim=2),
    eq_relpose: wp.array(dtype=wp.transform, ndim=2),
    eq_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- Solver State ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_eq: wp.array(dtype=wp.float32, ndim=2),
    J_hat_eq_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_eq_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, eq_idx = wp.tid()

    if not eq_enabled[world_idx, eq_idx]:
        return
        
    type = eq_type[world_idx, eq_idx]
    
    count = 0
    if type == EQ_TYPE_CONNECT: 
        count = 3
    elif type == EQ_TYPE_WELD: 
        count = 6
    elif type == EQ_TYPE_TRACK:
        count = 5

    start_offset = constraint_offsets[world_idx, eq_idx]
    
    # 1. Fetch Indices
    b1_idx = eq_body1[world_idx, eq_idx]
    b2_idx = eq_body2[world_idx, eq_idx]

    # Set active
    for k in range(count):
        constraint_active_mask[world_idx, start_offset + k] = 1.0

    # 2. Compute Kinematics
    
    # Body 2 (Child)
    X_body_2 = wp.transform_identity()
    com_2 = wp.vec3(0.0)
    if b2_idx >= 0:
        X_body_2 = body_q[world_idx, b2_idx]
        com_2 = body_com[world_idx, b2_idx]
        
    # Body 1 (Parent)
    X_body_1 = wp.transform_identity()
    com_1 = wp.vec3(0.0)
    if b1_idx >= 0:
        X_body_1 = body_q[world_idx, b1_idx]
        com_1 = body_com[world_idx, b1_idx]
        
    X_p_local = wp.transform_identity()
    X_c_local = wp.transform_identity()
    
    if type == EQ_TYPE_CONNECT: 
        X_p_local = wp.transform(eq_anchor[world_idx, eq_idx], wp.quat_identity())
        X_c_local = wp.transform_identity() 
        
    elif type == EQ_TYPE_WELD: 
        X_p_local = eq_relpose[world_idx, eq_idx]
        X_c_local = wp.transform_identity()
        
    # Kinematics
    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_1, com_1, X_p_local)
    X_w_c, r_c, pos_c = compute_joint_transforms(X_body_2, com_2, X_c_local)
    
    # === LINEAR (CONNECT, WELD) ===
    if type == EQ_TYPE_CONNECT or type == EQ_TYPE_WELD:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_equality_constraint_component(
                J_p, J_c, err,
                world_idx, start_offset + i, b1_idx, b2_idx,
                h_d, h_eq, J_hat_eq_values, C_eq_values, body_lambda_eq,
                dt, compliance
            )

    # === ANGULAR (WELD only) ===
    if type == EQ_TYPE_WELD:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            submit_equality_constraint_component(
                J_p, J_c, err,
                world_idx, start_offset + 3 + i, b1_idx, b2_idx,
                h_d, h_eq, J_hat_eq_values, C_eq_values, body_lambda_eq,
                dt, compliance
            )

    # === TRACK ===
    if type == EQ_TYPE_TRACK:
        # Extract params from anchor
        params = eq_anchor[world_idx, eq_idx]
        dist = params[0]
        r1 = params[1]
        r2 = params[2]
        
        # We assume X_w_p represents the TRACK FRAME Origin (World Space)
        # X_w_p was computed using X_p_local = relpose.
        # So relpose stores the Track's transform relative to Body 1.
        X_p_local = eq_relpose[world_idx, eq_idx]
        X_w_p, r_p, pos_p = compute_joint_transforms(X_body_1, com_1, X_p_local)
        
        # Current Child Pos
        p_c_local = wp.transform_point(wp.transform_inverse(X_w_p), pos_c)
        
        u = track_project(wp.vec2(p_c_local[0], p_c_local[1]), r1, r2, dist)
        t_pos_2d, t_tan_2d, t_norm_2d, length = track_get_frame(u, r1, r2, dist)
        
        t_pos_local = wp.vec3(t_pos_2d[0], t_pos_2d[1], 0.0)
        t_tan_local = wp.vec3(t_tan_2d[0], t_tan_2d[1], 0.0)
        t_norm_local = wp.vec3(t_norm_2d[0], t_norm_2d[1], 0.0)
        t_binorm_local = wp.vec3(0.0, 0.0, 1.0)
        
        t_pos_world = wp.transform_point(X_w_p, t_pos_local)
        t_tan_world = wp.transform_vector(X_w_p, t_tan_local)
        t_norm_world = wp.transform_vector(X_w_p, t_norm_local)
        t_binorm_world = wp.transform_vector(X_w_p, t_binorm_local)
        
        # --- Angular (Rows 0, 1, 2) ---
        # --- Angular (Rows 0, 1, 2) ---
        q_target = wp.quat_from_matrix(wp.matrix_from_cols(t_tan_world, t_norm_world, t_binorm_world))
        X_target = wp.transform(t_pos_world, q_target)
        
        for i in range(3):
            J_p, J_c, err = get_angular_component(X_target, X_w_c, i)
            submit_equality_constraint_component(
                J_p, J_c, err,
                world_idx, start_offset + i, b1_idx, b2_idx,
                h_d, h_eq, J_hat_eq_values, C_eq_values, body_lambda_eq,
                dt, compliance
            )
            
        # --- Linear (Rows 3, 4) ---
        delta = pos_c - t_pos_world
        com_1_world = wp.transform_point(X_body_1, com_1)
        com_2_world = wp.transform_point(X_body_2, com_2)
        
        # Row 3: Normal
        axis = t_norm_world
        err = wp.dot(delta, axis)
        
        J_c_lin = axis
        J_c_ang = wp.cross(pos_c - com_2_world, axis)
        J_c_sv = wp.spatial_vector(J_c_lin, J_c_ang)
        
        J_p_lin = -axis
        J_p_ang = wp.cross(t_pos_world - com_1_world, -axis)
        J_p_sv = wp.spatial_vector(J_p_lin, J_p_ang)
        
        submit_equality_constraint_component(
             J_p_sv, J_c_sv, err,
             world_idx, start_offset + 3, b1_idx, b2_idx,
             h_d, h_eq, J_hat_eq_values, C_eq_values, body_lambda_eq,
             dt, compliance
        )
        
        # Row 4: Binormal
        axis = t_binorm_world
        err = wp.dot(delta, axis)
        
        J_c_lin = axis
        J_c_ang = wp.cross(pos_c - com_2_world, axis)
        J_c_sv = wp.spatial_vector(J_c_lin, J_c_ang)
        
        J_p_lin = -axis
        J_p_ang = wp.cross(t_pos_world - com_1_world, -axis)
        J_p_sv = wp.spatial_vector(J_p_lin, J_p_ang)
        
        submit_equality_constraint_component(
             J_p_sv, J_c_sv, err,
             world_idx, start_offset + 4, b1_idx, b2_idx,
             h_d, h_eq, J_hat_eq_values, C_eq_values, body_lambda_eq,
             dt, compliance
        )

def compute_equality_constraint_offsets_batched(eq_types: wp.array):
    """
    eq_types: numpy array of shape (num_worlds, num_constraints)
    """
    eq_types_np = eq_types.numpy()
    
    counts = np.zeros_like(eq_types_np, dtype=np.int32)
    counts[eq_types_np == EQ_TYPE_CONNECT] = 3
    counts[eq_types_np == EQ_TYPE_WELD] = 6
    counts[eq_types_np == EQ_TYPE_TRACK] = 5
    
    total_constraints = counts.sum(axis=1)
    
    offsets = np.zeros_like(counts)
    offsets[:, 1:] = np.cumsum(counts[:, :-1], axis=1)
    
    return wp.array(offsets, dtype=wp.int32, device=eq_types.device), total_constraints[0]

@wp.kernel
def fill_equality_constraint_body_idx_kernel(
    eq_type: wp.array(dtype=wp.int32, ndim=2),
    eq_body1: wp.array(dtype=wp.int32, ndim=2),
    eq_body2: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # Output
    eq_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, eq_idx = wp.tid()
    
    if world_idx >= eq_type.shape[0] or eq_idx >= eq_type.shape[1]:
        return

    type = eq_type[world_idx, eq_idx]
    b1 = eq_body1[world_idx, eq_idx]
    b2 = eq_body2[world_idx, eq_idx]
    start_offset = constraint_offsets[world_idx, eq_idx]

    count = 0
    if type == EQ_TYPE_CONNECT: count = 3
    elif type == EQ_TYPE_WELD: count = 6
    elif type == EQ_TYPE_TRACK: count = 5
    
    for k in range(count):
        offset = start_offset + k
        if offset < eq_constraint_body_idx.shape[1]:
            eq_constraint_body_idx[world_idx, offset, 0] = b1
            eq_constraint_body_idx[world_idx, offset, 1] = b2

@wp.kernel
def fill_equality_constraint_active_mask_kernel(
    eq_type: wp.array(dtype=wp.int32, ndim=2),
    eq_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # Output
    eq_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, eq_idx = wp.tid()
    if world_idx >= eq_type.shape[0] or eq_idx >= eq_type.shape[1]:
        return

    type = eq_type[world_idx, eq_idx]
    is_enabled = eq_enabled[world_idx, eq_idx]
    start_offset = constraint_offsets[world_idx, eq_idx]

    count = 0
    if type == EQ_TYPE_CONNECT: count = 3
    elif type == EQ_TYPE_WELD: count = 6
    elif type == EQ_TYPE_TRACK: count = 5
    
    val = 1.0 if is_enabled else 0.0
    for k in range(count):
        offset = start_offset + k
        if offset < eq_constraint_active_mask.shape[1]:
            eq_constraint_active_mask[world_idx, offset] = val

@wp.func
def submit_batch_equality_constraint_component(
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
    # --- System Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_eq: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_eq: wp.array(dtype=wp.float32, ndim=3),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
):
    current_lambda = body_lambda_eq[batch_idx, world_idx, constraint_idx]

    if p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, p_idx, -J_p * current_lambda * dt)

    if c_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, c_idx, -J_c * current_lambda * dt)

    h_eq[batch_idx, world_idx, constraint_idx] = (error + compliance * current_lambda) / dt

@wp.kernel
def batch_positional_equality_constraint_residual_kernel(
    # --- State (Batched) ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_lambda_eq: wp.array(dtype=wp.float32, ndim=3),
    # --- Model Data (Shared) ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # --- Equality Constraint Definition ---
    eq_type: wp.array(dtype=wp.int32, ndim=2),
    eq_body1: wp.array(dtype=wp.int32, ndim=2),
    eq_body2: wp.array(dtype=wp.int32, ndim=2),
    eq_anchor: wp.array(dtype=wp.vec3, ndim=2),
    eq_relpose: wp.array(dtype=wp.transform, ndim=2),
    eq_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
    # --- Outputs (Batched) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_eq: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, eq_idx = wp.tid()

    if not eq_enabled[world_idx, eq_idx]:
        return
        
    type = eq_type[world_idx, eq_idx]
    start_offset = constraint_offsets[world_idx, eq_idx]
    
    b1_idx = eq_body1[world_idx, eq_idx]
    b2_idx = eq_body2[world_idx, eq_idx]

    # Kinematics
    X_body_2 = wp.transform_identity()
    com_2 = wp.vec3(0.0)
    if b2_idx >= 0:
        X_body_2 = body_q[batch_idx, world_idx, b2_idx]
        com_2 = body_com[world_idx, b2_idx]
        
    X_body_1 = wp.transform_identity()
    com_1 = wp.vec3(0.0)
    if b1_idx >= 0:
        X_body_1 = body_q[batch_idx, world_idx, b1_idx]
        com_1 = body_com[world_idx, b1_idx]
        
    X_p_local = wp.transform_identity()
    X_c_local = wp.transform_identity()
    
    if type == EQ_TYPE_CONNECT: 
        X_p_local = wp.transform(eq_anchor[world_idx, eq_idx], wp.quat_identity())
        
    elif type == EQ_TYPE_WELD: 
        X_p_local = eq_relpose[world_idx, eq_idx]

    elif type == EQ_TYPE_TRACK:
        X_p_local = eq_relpose[world_idx, eq_idx]
        
    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_1, com_1, X_p_local)
    X_w_c, r_c, pos_c = compute_joint_transforms(X_body_2, com_2, X_c_local)
    
    # === LINEAR (CONNECT, WELD) ===
    if type == EQ_TYPE_CONNECT or type == EQ_TYPE_WELD:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_batch_equality_constraint_component(
                J_p, J_c, err,
                batch_idx, world_idx, start_offset + i, b1_idx, b2_idx,
                h_d, h_eq, body_lambda_eq,
                dt, compliance
            )

    # === ANGULAR (WELD) ===
    if type == EQ_TYPE_WELD:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            submit_batch_equality_constraint_component(
                J_p, J_c, err,
                batch_idx, world_idx, start_offset + 3 + i, b1_idx, b2_idx,
                h_d, h_eq, body_lambda_eq,
                dt, compliance
            )

    # === TRACK ===
    if type == EQ_TYPE_TRACK:
        params = eq_anchor[world_idx, eq_idx]
        dist = params[0]
        r1 = params[1]
        r2 = params[2]
        
        # Current Child Pos
        p_c_local = wp.transform_point(wp.transform_inverse(X_w_p), pos_c)
        u = track_project(wp.vec2(p_c_local[0], p_c_local[1]), r1, r2, dist)
        t_pos_2d, t_tan_2d, t_norm_2d, length = track_get_frame(u, r1, r2, dist)
        
        t_pos_local = wp.vec3(t_pos_2d[0], t_pos_2d[1], 0.0)
        t_tan_local = wp.vec3(t_tan_2d[0], t_tan_2d[1], 0.0)
        t_norm_local = wp.vec3(t_norm_2d[0], t_norm_2d[1], 0.0)
        t_binorm_local = wp.vec3(0.0, 0.0, 1.0)
        
        t_pos_world = wp.transform_point(X_w_p, t_pos_local)
        t_tan_world = wp.transform_vector(X_w_p, t_tan_local)
        t_norm_world = wp.transform_vector(X_w_p, t_norm_local)
        t_binorm_world = wp.transform_vector(X_w_p, t_binorm_local)
        
        # --- Angular ---
        # --- Angular (Rows 0, 1, 2) ---
        q_target = wp.quat_from_matrix(wp.matrix_from_cols(t_tan_world, t_norm_world, t_binorm_world))
        X_target = wp.transform(t_pos_world, q_target)
        
        for i in range(3):
            J_p, J_c, err = get_angular_component(X_target, X_w_c, i)
            submit_batch_equality_constraint_component(
                J_p, J_c, err,
                batch_idx, world_idx, start_offset + i, b1_idx, b2_idx,
                h_d, h_eq, body_lambda_eq,
                dt, compliance
            )
            
        # --- Linear ---
        delta = pos_c - t_pos_world
        com_1_world = wp.transform_point(X_body_1, com_1)
        com_2_world = wp.transform_point(X_body_2, com_2)
        
        # Normal
        axis = t_norm_world
        err = wp.dot(delta, axis)
        J_c_lin = axis
        J_c_ang = wp.cross(pos_c - com_2_world, axis)
        J_c_sv = wp.spatial_vector(J_c_lin, J_c_ang)
        J_p_lin = -axis
        J_p_ang = wp.cross(t_pos_world - com_1_world, -axis)
        J_p_sv = wp.spatial_vector(J_p_lin, J_p_ang)
        
        submit_batch_equality_constraint_component(
             J_p_sv, J_c_sv, err,
             batch_idx, world_idx, start_offset + 3, b1_idx, b2_idx,
             h_d, h_eq, body_lambda_eq,
             dt, compliance
        )
        
        # Binormal
        axis = t_binorm_world
        err = wp.dot(delta, axis)
        J_c_lin = axis
        J_c_ang = wp.cross(pos_c - com_2_world, axis)
        J_c_sv = wp.spatial_vector(J_c_lin, J_c_ang)
        J_p_lin = -axis
        J_p_ang = wp.cross(t_pos_world - com_1_world, -axis)
        J_p_sv = wp.spatial_vector(J_p_lin, J_p_ang)
        
        submit_batch_equality_constraint_component(
             J_p_sv, J_c_sv, err,
             batch_idx, world_idx, start_offset + 4, b1_idx, b2_idx,
             h_d, h_eq, body_lambda_eq,
             dt, compliance
        )