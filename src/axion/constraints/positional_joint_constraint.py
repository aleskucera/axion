import warp as wp

from .joint_kinematics import compute_joint_transforms
from .joint_kinematics import get_angular_component
from .joint_kinematics import get_linear_component
from .joint_kinematics import get_revolute_angular_component


@wp.func
def submit_position_component(
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
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
):
    current_lambda = body_lambda_j[world_idx, constraint_idx]

    # 1. Apply forces to bodies (atomic add required as bodies are shared)
    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, -J_p * current_lambda * dt)

    wp.atomic_add(h_d, world_idx, c_idx, -J_c * current_lambda * dt)

    # 2. Write the constraint residual (h_j)
    # The solver tries to drive (error + compliance * lambda) / dt -> 0
    h_j[world_idx, constraint_idx] = (error + compliance * current_lambda) / dt

    # 3. Store the Jacobians for the linear solver step
    J_hat_j_values[world_idx, constraint_idx, 0] = J_p
    J_hat_j_values[world_idx, constraint_idx, 1] = J_c

    # 4. Store Compliance (inverse stiffness)
    C_j_values[world_idx, constraint_idx] = compliance / (dt * dt)


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
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_compliance_override: wp.array(dtype=wp.float32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    global_compliance: wp.float32,
    # --- Solver State ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    j_type = joint_type[world_idx, joint_idx]

    comp = joint_compliance_override[world_idx, joint_idx]
    if comp < 0.0:
        comp = global_compliance

    count = 0
    if j_type == 1:  # REVOLUTE
        count = 5
    elif j_type == 2:  # BALL
        count = 3
    elif j_type == 3:  # FIXED
        count = 6

    start_offset = constraint_offsets[world_idx, joint_idx]

    # 1. Fetch Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]

    if joint_enabled[world_idx, joint_idx] == 0 or c_idx < 0:
        for k in range(count):
            constraint_active_mask[world_idx, start_offset + k] = 0.0
            body_lambda_j[world_idx, start_offset + k] = 0.0
            # Also zero out other outputs to be safe?
            # h_j[...] = 0.0, etc. is good practice but maybe not strictly required if solver ignores inactive rows
        return

    # Set active
    for k in range(count):
        constraint_active_mask[world_idx, start_offset + k] = 1.0

    # 2. Compute Kinematics
    # Child
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx], body_com[world_idx, c_idx], joint_X_c[world_idx, joint_idx]
    )

    # Parent
    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]

    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    # 3. Apply Constraints based on Joint Type

    # === LINEAR PART (XYZ) ===
    # REVOLUTE(1), BALL(2), FIXED(3)
    if j_type == 1 or j_type == 2 or j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_position_component(
                J_p,
                J_c,
                err,
                world_idx,
                start_offset + i,
                p_idx,
                c_idx,
                h_d,
                h_j,
                J_hat_j_values,
                C_j_values,
                body_lambda_j,
                dt,
                comp,
            )

    # === REVOLUTE ANGULAR (1) ===
    if j_type == 1:
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

        # Ortho 1
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 0)
        submit_position_component(
            J_p,
            J_c,
            err,
            world_idx,
            start_offset + 3,
            p_idx,
            c_idx,
            h_d,
            h_j,
            J_hat_j_values,
            C_j_values,
            body_lambda_j,
            dt,
            comp,
        )
        # Ortho 2
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 1)
        submit_position_component(
            J_p,
            J_c,
            err,
            world_idx,
            start_offset + 4,
            p_idx,
            c_idx,
            h_d,
            h_j,
            J_hat_j_values,
            C_j_values,
            body_lambda_j,
            dt,
            comp,
        )

    # === FIXED ANGULAR (3) ===
    if j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            submit_position_component(
                J_p,
                J_c,
                err,
                world_idx,
                start_offset + 3 + i,
                p_idx,
                c_idx,
                h_d,
                h_j,
                J_hat_j_values,
                C_j_values,
                body_lambda_j,
                dt,
                comp,
            )


# ---------------------------------------------------------------------------- #
#                           Batch Helper Functions                             #
# ---------------------------------------------------------------------------- #


@wp.func
def submit_batch_position_component(
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
    h_j: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
):
    current_lambda = body_lambda_j[batch_idx, world_idx, constraint_idx]

    if p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, p_idx, -J_p * current_lambda * dt)

    wp.atomic_add(h_d, batch_idx, world_idx, c_idx, -J_c * current_lambda * dt)

    h_j[batch_idx, world_idx, constraint_idx] = (error + compliance * current_lambda) / dt


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
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
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

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    # 1. Fetch Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    start_offset = constraint_offsets[world_idx, joint_idx]

    # Child
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[batch_idx, world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx],
    )

    # Parent
    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[batch_idx, world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]

    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    # 3. Apply Constraints

    # === LINEAR (XYZ) ===
    if j_type == 1 or j_type == 2 or j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
            submit_batch_position_component(
                J_p,
                J_c,
                err,
                batch_idx,
                world_idx,
                start_offset + i,
                p_idx,
                c_idx,
                h_d,
                h_j,
                body_lambda_j,
                dt,
                compliance,
            )

    # === REVOLUTE (Angular) ===
    if j_type == 1:
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

        # Ortho 1
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 0)
        submit_batch_position_component(
            J_p,
            J_c,
            err,
            batch_idx,
            world_idx,
            start_offset + 3,
            p_idx,
            c_idx,
            h_d,
            h_j,
            body_lambda_j,
            dt,
            compliance,
        )
        # Ortho 2
        J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 1)
        submit_batch_position_component(
            J_p,
            J_c,
            err,
            batch_idx,
            world_idx,
            start_offset + 4,
            p_idx,
            c_idx,
            h_d,
            h_j,
            body_lambda_j,
            dt,
            compliance,
        )

    # === FIXED (Angular) ===
    if j_type == 3:
        for i in range(wp.static(3)):
            J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
            submit_batch_position_component(
                J_p,
                J_c,
                err,
                batch_idx,
                world_idx,
                start_offset + 3 + i,
                p_idx,
                c_idx,
                h_d,
                h_j,
                body_lambda_j,
                dt,
                compliance,
            )


@wp.kernel
def fused_batch_positional_joint_residual_kernel(
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
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Params ---
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # --- Outputs (Batched) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
    h_j: wp.array(dtype=wp.float32, ndim=3),  # [Batch, World, Constraint]
):
    # Grid: (num_worlds, num_joints)
    world_idx, joint_idx = wp.tid()

    if joint_idx >= joint_type.shape[1]:
        return

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    
    # 1. Fetch Indices
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    start_offset = constraint_offsets[world_idx, joint_idx]
    
    # Static Data Loading
    joint_X_c_val = joint_X_c[world_idx, joint_idx]
    com_c = body_com[world_idx, c_idx]
    
    joint_X_p_val = joint_X_p[world_idx, joint_idx]
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        com_p = body_com[world_idx, p_idx]
        
    axis_idx = joint_qd_start[world_idx, joint_idx]
    axis_local = joint_axis[world_idx, axis_idx]

    for b in range(num_batches):
        # Child
        X_w_c, r_c, pos_c = compute_joint_transforms(
            body_q[b, world_idx, c_idx],
            com_c,
            joint_X_c_val,
        )

        # Parent
        X_body_p = wp.transform_identity()
        if p_idx >= 0:
            X_body_p = body_q[b, world_idx, p_idx]

        X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p_val)

        # 3. Apply Constraints

        # === LINEAR (XYZ) ===
        if j_type == 1 or j_type == 2 or j_type == 3:
            for i in range(wp.static(3)):
                J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, i)
                
                # INLINE submit
                c_idx_local = start_offset + i
                lam = body_lambda_j[b, world_idx, c_idx_local]
                if p_idx >= 0:
                    wp.atomic_add(h_d, b, world_idx, p_idx, -J_p * lam * dt)
                wp.atomic_add(h_d, b, world_idx, c_idx, -J_c * lam * dt)
                h_j[b, world_idx, c_idx_local] = (err + compliance * lam) / dt

        # === REVOLUTE (Angular) ===
        if j_type == 1:
            # Ortho 1
            J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 0)
            
            c_idx_local = start_offset + 3
            lam = body_lambda_j[b, world_idx, c_idx_local]
            if p_idx >= 0:
                wp.atomic_add(h_d, b, world_idx, p_idx, -J_p * lam * dt)
            wp.atomic_add(h_d, b, world_idx, c_idx, -J_c * lam * dt)
            h_j[b, world_idx, c_idx_local] = (err + compliance * lam) / dt
            
            # Ortho 2
            J_p, J_c, err = get_revolute_angular_component(X_w_p, X_w_c, axis_local, 1)

            c_idx_local = start_offset + 4
            lam = body_lambda_j[b, world_idx, c_idx_local]
            if p_idx >= 0:
                wp.atomic_add(h_d, b, world_idx, p_idx, -J_p * lam * dt)
            wp.atomic_add(h_d, b, world_idx, c_idx, -J_c * lam * dt)
            h_j[b, world_idx, c_idx_local] = (err + compliance * lam) / dt

        # === FIXED (Angular) ===
        if j_type == 3:
            for i in range(wp.static(3)):
                J_p, J_c, err = get_angular_component(X_w_p, X_w_c, i)
                
                c_idx_local = start_offset + 3 + i
                lam = body_lambda_j[b, world_idx, c_idx_local]
                if p_idx >= 0:
                    wp.atomic_add(h_d, b, world_idx, p_idx, -J_p * lam * dt)
                wp.atomic_add(h_d, b, world_idx, c_idx, -J_c * lam * dt)
                h_j[b, world_idx, c_idx_local] = (err + compliance * lam) / dt

