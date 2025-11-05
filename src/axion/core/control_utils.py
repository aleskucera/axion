import warp as wp
from newton import Control
from newton import JointMode
from newton import JointType
from newton import Model
from newton import State


@wp.func
def joint_force(
    q: float,
    qd: float,
    q_prev: float,
    qd_prev: float,
    act: float,
    target_ke: float,
    target_kd: float,
    target_ki: float,
    integral_error: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
    mode: wp.int32,
    is_custom_attribute_used: wp.array(dtype=bool),
) -> float:
    """
    Control law regulator for a single joint degree of freedom.
    Calculates the desired force/torque based on the control mode,
    target values, and joint limits.
    """

    limit_f = 0.0
    damping_f = 0.0
    target_f = 0.0

    if mode == JointMode.TARGET_POSITION:
        target_f = target_ke * (act - q) - target_kd * qd
    elif mode == JointMode.TARGET_VELOCITY:
        target_f = target_ke * (act - qd)
    elif mode == JointMode.NONE:
        target_f = act

    # Compute limit forces, damping only active when limit is violated
    # Note: Target forces are overridden when limits are active to prioritize the limit spring.
    if limit_ke > 0.0:
        if q < limit_lower:
            limit_f = limit_ke * (limit_lower - q)
            if qd < 0.0:  # Only apply damping against the direction of violation
                damping_f = -limit_kd * qd
            if mode != JointMode.NONE: # if mode == JointMode.TARGET_VELOCITY
                target_f = 0.0  # Override target force when limit is violated
        elif q > limit_upper:
            limit_f = limit_ke * (limit_upper - q)
            if qd > 0.0:  # Only apply damping against the direction of violation
                damping_f = -limit_kd * qd
            if mode != JointMode.NONE: # if mode == JointMode.TARGET_VELOCITY
                target_f = 0.0  # Override target force when limit is violated

    return limit_f + damping_f + target_f


@wp.kernel
def apply_joint_control_kernel(
    # --- State Inputs ---
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_prev: wp.array(dtype=float),     # custom
    joint_qd_prev: wp.array(dtype=float),    # custom
    # --- Model Inputs ---
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    # --- Control/Actuation Inputs ---
    joint_target: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    joint_integral_err: wp.array(dtype=float),
    joint_dof_mode: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_target_ki: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    is_custom_attribute_used: wp.array(dtype=bool),
    # --- Output ---
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()  # Each thread processes one joint

    j_type = joint_type[tid]
    if j_type == JointType.FIXED:
        return

    # === Step 1: Calculate Kinematics (Moment Arms) ===
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    # Parent kinematics
    pose_p = wp.transform_identity()
    com_p = wp.vec3()
    if id_p >= 0:
        pose_p = body_q[id_p]
        com_p = body_com[id_p]

    X_wp = pose_p * joint_X_p[tid]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # Child kinematics (this uses the corrected logic)
    pose_c = body_q[id_c]
    com_c = body_com[id_c]
    X_wc = pose_c * joint_X_c[tid]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # === Step 2: Calculate Forces/Torques for each DoF ===
    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    f_total = wp.vec3()
    t_total = wp.vec3()

    # --- Process Linear Degrees of Freedom ---
    # Unrolled loop for Warp compiler efficiency
    if lin_axis_count > 0:
        dof_idx = qd_start + 0
        f = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        f_total += axis_world * (joint_f[dof_idx] + f)

    if lin_axis_count > 1:
        dof_idx = qd_start + 1
        f = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        f_total += axis_world * (joint_f[dof_idx] + f)

    if lin_axis_count > 2:
        dof_idx = qd_start + 2
        f = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        f_total += axis_world * (joint_f[dof_idx] + f)

    # --- Process Angular Degrees of Freedom ---
    # Angular DoF indices start after all linear DoFs
    ang_dof_start_offset = qd_start + lin_axis_count
    if ang_axis_count > 0:
        dof_idx = ang_dof_start_offset + 0
        t = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        t_total += axis_world * (joint_f[dof_idx] + t)

    if ang_axis_count > 1:
        dof_idx = ang_dof_start_offset + 1
        t = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        t_total += axis_world * (joint_f[dof_idx] + t)

    if ang_axis_count > 2:
        dof_idx = ang_dof_start_offset + 2
        t = joint_force(
            joint_q[dof_idx],
            joint_qd[dof_idx],
            joint_q_prev[dof_idx],
            joint_qd_prev[dof_idx],
            joint_target[dof_idx],
            joint_target_ke[dof_idx],
            joint_target_kd[dof_idx],
            joint_target_ki[dof_idx],
            joint_integral_err[dof_idx],
            joint_limit_lower[dof_idx],
            joint_limit_upper[dof_idx],
            joint_limit_ke[dof_idx],
            joint_limit_kd[dof_idx],
            joint_dof_mode[dof_idx],
            is_custom_attribute_used,
        )
        axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
        t_total += axis_world * (joint_f[dof_idx] + t)

    # === Step 3: Apply Spatial Forces to Bodies ===
    # Apply force/torque to child body
    wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total)))

    # Apply equal and opposite force/torque to parent body
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))


def apply_control(
    model: Model,
    state_in: State,
    state_out: State,
    dt: float,
    control: Control | None = None,
):
    """
    Launches the kernel to calculate and apply joint control forces.
    This function reads joint targets from the control object and applies
    the corresponding forces to the bodies in state_in.
    """
    if control is None:
        control = model.control(clone_variables=False)

    if model.body_count and model.joint_count:

        # Create additional argument for apply_joint_control_kernel signaling which custom attributes does the current Newton Model have
        num_of_custom_attributes = 4
        is_custom_attribute_used = num_of_custom_attributes*[False]
        # mapping: 0 -> joint_q_prev, 1 ->joint_qd_prev, 2->joint_target_ki, 3->joint_integral_err
        is_custom_attribute_used[0] = hasattr(model, "joint_q_prev")
        is_custom_attribute_used[1] = hasattr(model, "joint_qd_prev")
        is_custom_attribute_used[2] = hasattr(model, "joint_target_ki")
        is_custom_attribute_used[3] = hasattr(control, "joint_integral_err")
        is_custom_attribute_used = wp.array(is_custom_attribute_used, dtype=bool, device=model.device)
       
        # Use getattr with fallback to zero-arrays for the custom model attributes
        joint_q_prev = getattr(model, "joint_q_prev", wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device))
        joint_qd_prev = getattr(model, "joint_qd_prev", wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device))
        joint_target_ki = getattr(model, "joint_target_ki", wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device))
        joint_integral_err = getattr(control, "joint_integral_err", wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device))
        
        wp.launch(
            kernel=apply_joint_control_kernel,
            dim=model.joint_count,
            inputs=[
                # State
                state_in.body_q,
                model.joint_q,
                model.joint_qd,
                joint_q_prev,
                joint_qd_prev,
                # Model
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                # Control
                control.joint_target,
                control.joint_f,
                joint_integral_err,
                model.joint_dof_mode,
                model.joint_target_ke,
                model.joint_target_kd,
                joint_target_ki,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                is_custom_attribute_used,
            ],
            outputs=[state_in.body_f],
            device=model.device,
        )
