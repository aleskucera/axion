import warp as wp
from axion.constraints.joint_kinematics import compute_joint_transforms
from axion.core.control_utils import JointMode


@wp.func
def compute_revolute_q_qd(
    X_w_p: wp.transform,
    X_w_c: wp.transform,
    axis_local: wp.vec3,
    body_u_p: wp.spatial_vector,
    body_u_c: wp.spatial_vector,
):
    # 1. Compute q (angle)
    q_p = wp.transform_get_rotation(X_w_p)
    q_c = wp.transform_get_rotation(X_w_c)
    
    q_p_inv = wp.quat_inverse(q_p)
    q_rel = wp.mul(q_p_inv, q_c)
    
    vec_part = wp.vec3(q_rel[0], q_rel[1], q_rel[2])
    sin_half_theta = wp.dot(vec_part, axis_local)
    cos_half_theta = q_rel[3]
    
    q = 2.0 * wp.atan2(sin_half_theta, cos_half_theta)
    
    # 2. Compute qd (velocity)
    # relative angular velocity in world space
    # NOTE: spatial_bottom is angular velocity in (linear, angular) convention
    w_p = wp.spatial_bottom(body_u_p)
    w_c = wp.spatial_bottom(body_u_c)
    
    w_rel_world = w_c - w_p
    
    # Project onto world axis
    axis_world = wp.quat_rotate(q_p, axis_local)
    qd = wp.dot(w_rel_world, axis_world)
    
    return q, qd


@wp.func
def get_control_component(
    X_w_p: wp.transform,
    axis_local: wp.vec3,
    q: wp.float32,
    qd: wp.float32,
    target: wp.float32,
    mode: wp.int32,
    dt: wp.float32,
):
    """
    Computes Jacobian and Error for control constraint.
    """
    J_p = wp.spatial_vector()
    J_c = wp.spatial_vector()
    error = 0.0

    # 1. Compute Jacobian (Same as joint axis)
    # Hinge axis in world space
    q_p = wp.transform_get_rotation(X_w_p)
    axis_w = wp.quat_rotate(q_p, axis_local)
    
    # J_c = [0, axis], J_p = [0, -axis]
    J_c = wp.spatial_vector(wp.vec3(0.0), axis_w)
    J_p = wp.spatial_vector(wp.vec3(0.0), -axis_w)

    # 2. Compute Error (Velocity level)
    # We want h_c such that b = (JHinvg - h_c) / dt is Acceleration.
    # So h_c should be Velocity.
    if mode == JointMode.TARGET_POSITION:
        # C(q) = q - target = 0
        # h_c = (q - target) / dt
        error = (q - target) / dt
        
    elif mode == JointMode.TARGET_VELOCITY:
        # C(u) = qd - target = 0
        # h_c = qd - target
        error = qd - target

    return J_p, J_c, error


@wp.kernel
def control_constraint_kernel(
    # --- State ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2), # Needed for qd
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=2),
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
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    
    # --- Control Inputs ---
    joint_target: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    
    # --- Params ---
    dt: wp.float32,
    
    # --- Solver State Output ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_ctrl: wp.array(dtype=wp.float32, ndim=2),
    J_hat_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    
    # Only support Revolute(1)
    if j_type != 1:
        return
        
    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    
    # Check mode
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0: # NONE
        return

    # Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]
    
    # Activate
    constraint_active_mask[world_idx, ctrl_offset] = 1.0
    
    # Kinematics
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx]
    )
    
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
    
    axis_local = joint_axis[world_idx, qd_start_idx]
    
    # Compute current_q and current_qd
    body_u_c = body_u[world_idx, c_idx]
    body_u_p = wp.spatial_vector()
    if p_idx >= 0:
        body_u_p = body_u[world_idx, p_idx]
    
    current_q, current_qd = compute_revolute_q_qd(
        X_w_p, X_w_c, axis_local, body_u_p, body_u_c
    )
    
    target = joint_target[world_idx, qd_start_idx]
    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]
    
    # Calculate Compliance (alpha)
    # alpha should have units 1/Mass
    alpha = 0.0
    
    if mode == JointMode.TARGET_POSITION:
        # alpha = 1 / (dt^2 * Kp + dt * Kd)
        denom = (dt * dt * ke + dt * kd)
        if denom > 1e-6:
            alpha = 1.0 / denom
            
    elif mode == JointMode.TARGET_VELOCITY:
        # alpha = 1 / (dt * Kv)
        # We use 'ke' as the velocity gain Kv
        denom = (dt * ke)
        if denom > 1e-6:
            alpha = 1.0 / denom
    
    J_p, J_c, error_vel = get_control_component(
        X_w_p, axis_local, current_q, current_qd, target, mode, dt
    )
    
    # Submit constraint
    current_lambda = body_lambda_ctrl[world_idx, ctrl_offset]
    
    # 1. Apply forces
    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, -J_p * current_lambda * dt)
    wp.atomic_add(h_d, world_idx, c_idx, -J_c * current_lambda * dt)
    
    # 2. Residual
    # h_c = velocity_error + alpha * current_lambda * dt
    # alpha * lambda * dt -> (1/Mass) * Force * Time = Velocity. Consistent.
    h_ctrl[world_idx, ctrl_offset] = error_vel + alpha * current_lambda * dt
    
    # 3. Jacobians
    J_hat_values[world_idx, ctrl_offset, 0] = J_p
    J_hat_values[world_idx, ctrl_offset, 1] = J_c
    
    # 4. Compliance
    C_values[world_idx, ctrl_offset] = alpha


@wp.kernel
def batch_control_constraint_residual_kernel(
    # --- State (Batched) ---
    body_q: wp.array(dtype=wp.transform, ndim=3), # [Batch, World, Body]
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=3), # [Batch, World, Constraint]
    
    # --- Model Data (Shared) ---
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    
    # --- Control Inputs (Shared) ---
    joint_target: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    
    # --- Params ---
    dt: wp.float32,
    
    # --- Output (Batched) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_ctrl: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    
    # Only support Revolute(1)
    if j_type != 1:
        return
        
    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    
    # Check mode
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0: # NONE
        return

    # Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]
    
    # Kinematics
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[batch_idx, world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx]
    )
    
    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[batch_idx, world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        
    X_w_p, r_p, pos_p = compute_joint_transforms(
        X_body_p,
        com_p,
        joint_X_p[world_idx, joint_idx]
    )
    
    axis_local = joint_axis[world_idx, qd_start_idx]
    
    # Compute current_q and current_qd
    body_u_c = body_u[batch_idx, world_idx, c_idx]
    body_u_p = wp.spatial_vector()
    if p_idx >= 0:
        body_u_p = body_u[batch_idx, world_idx, p_idx]
    
    current_q, current_qd = compute_revolute_q_qd(
        X_w_p, X_w_c, axis_local, body_u_p, body_u_c
    )
    
    target = joint_target[world_idx, qd_start_idx]
    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]
    
    # Compliance
    alpha = 0.0
    if mode == JointMode.TARGET_POSITION:
        denom = (dt * dt * ke + dt * kd)
        if denom > 1e-6:
            alpha = 1.0 / denom
    elif mode == JointMode.TARGET_VELOCITY:
        denom = (dt * ke)
        if denom > 1e-6:
            alpha = 1.0 / denom
    
    J_p, J_c, error_vel = get_control_component(
        X_w_p, axis_local, current_q, current_qd, target, mode, dt
    )
    
    # Submit constraint
    current_lambda = body_lambda_ctrl[batch_idx, world_idx, ctrl_offset]
    
    # 1. Apply forces
    if p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, p_idx, -J_p * current_lambda * dt)
    wp.atomic_add(h_d, batch_idx, world_idx, c_idx, -J_c * current_lambda * dt)
    
    # 2. Residual
    h_ctrl[batch_idx, world_idx, ctrl_offset] = error_vel + alpha * current_lambda * dt


@wp.kernel
def count_control_constraints_kernel(
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    counts: wp.array(dtype=wp.int32, ndim=2),
):
    world_idx, joint_idx = wp.tid()
    
    j_type = joint_type[world_idx, joint_idx]
    qd_start = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start]
    
    count = 0
    # Only Revolute(1) supported for now
    if j_type == 1 and mode != 0:
        count = 1
        
    counts[world_idx, joint_idx] = count


def compute_control_constraint_offsets_batched(
    joint_type: wp.array,
    joint_dof_mode: wp.array,
    joint_qd_start: wp.array,
):
    num_worlds = joint_type.shape[0]
    num_joints = joint_type.shape[1]
    
    counts = wp.zeros((num_worlds, num_joints), dtype=wp.int32, device=joint_type.device)
    
    wp.launch(
        kernel=count_control_constraints_kernel,
        dim=(num_worlds, num_joints),
        inputs=[joint_type, joint_dof_mode, joint_qd_start],
        outputs=[counts],
        device=joint_type.device,
    )
    
    counts_np = counts.numpy()
    import numpy as np
    offsets_np = np.zeros_like(counts_np)
    
    row_counts = counts_np[0]
    row_offsets = np.cumsum(np.concatenate(([0], row_counts[:-1])))
    total = np.sum(row_counts)
    
    offsets_np[:] = row_offsets
    
    offsets = wp.from_numpy(offsets_np, dtype=wp.int32, device=joint_type.device)
    
    return offsets, int(total)


@wp.kernel
def fill_control_constraint_body_idx_kernel(
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    control_offsets: wp.array(dtype=wp.int32, ndim=2),
    # Output
    constraint_body_idx_ctrl: wp.array(dtype=wp.int32, ndim=3), 
):
    world_idx, joint_idx = wp.tid()
    
    j_type = joint_type[world_idx, joint_idx]
    qd_start = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start]
    
    count = 0
    # Only Revolute(1) supported for now
    if j_type == 1 and mode != 0:
        count = 1
        
    if count == 0:
        return
        
    offset = control_offsets[world_idx, joint_idx]
    
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    
    constraint_body_idx_ctrl[world_idx, offset, 0] = p_idx
    constraint_body_idx_ctrl[world_idx, offset, 1] = c_idx