import warp as wp
from newton import Control
from newton import JointMode
from newton import JointType
from newton import Model
from newton import State


@wp.func
def calculate_errors(
    act: float,
    q: float,
    qd: float,
    err_prev: float,
    mode: wp.int32,
    dt: wp.float32
):
    """
    Calculate the corresponding errors and or error differences.
    This is meant be run before increasing the integral error using wp.atomic_add()
    in the kernel.
    """

    if mode == JointMode.TARGET_POSITION:
        err = (act - q)
        err_d = (err - err_prev) / dt if dt > 0 else 0.0
        delta_err_i = err * dt
    elif mode == JointMode.TARGET_VELOCITY:
        err = (act - qd)
        err_d = (err - err_prev) / dt if dt > 0 else 0.0
        delta_err_i = err * dt
    elif mode == JointMode.NONE:
        err, delta_err_i, err_d = 0.0, 0.0, 0.0     

    return err, delta_err_i, err_d

@wp.func
def joint_force(
    q: float,
    qd: float,
    act: float,
    target_ke: float,
    target_kd: float,
    target_ki: float,
    err: float,
    err_i: float,
    err_d: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
    mode: wp.int32,
) -> float:
    """
    Control law regulator for a single joint degree of freedom.
    Calculates the desired force/torque based on the control mode,
    target values, and joint limits.
    """

    limit_f = 0.0
    damping_f = 0.0
    target_f = 0.0

    if mode == JointMode.TARGET_POSITION or mode == JointMode.TARGET_VELOCITY:
        target_f = target_ke * err + target_ki * err_i + target_kd * err_d   # the derivative term was completely different before, -target_kd * qd 
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
    joint_err_prev: wp.array(dtype=float),  # custom
    joint_err_i: wp.array(dtype=float),  # custom
    joint_dof_mode: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_target_ki: wp.array(dtype=float), # custom
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    dt: wp.float32,
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

    # --- Process Linear and Angular Degrees of Freedom ---
    ang_dof_start_offset = qd_start + lin_axis_count
    
    for i in range(wp.static(6)):   # 6 = number of max DOF
        st_i = wp.static(i)

        # linear DOFs:
        if st_i < 3:
            if lin_axis_count > st_i:
                dof_idx = qd_start + st_i

                err, delta_err_i, err_d = calculate_errors(
                                            joint_target[dof_idx], 
                                            joint_q[dof_idx],
                                            joint_qd[dof_idx],
                                            joint_err_prev[dof_idx],
                                            joint_dof_mode[dof_idx],
                                            dt)
                
                # add to the overall integral error:
                wp.atomic_add(joint_err_i, dof_idx, delta_err_i)
                # CONCERN: what about windup effect? Maybe add anti-windup 

                f = joint_force(
                    joint_q[dof_idx],
                    joint_qd[dof_idx],
                    joint_target[dof_idx],
                    joint_target_ke[dof_idx],
                    joint_target_kd[dof_idx],
                    joint_target_ki[dof_idx],
                    err,
                    joint_err_i[dof_idx],
                    err_d,
                    joint_limit_lower[dof_idx],
                    joint_limit_upper[dof_idx],
                    joint_limit_ke[dof_idx],
                    joint_limit_kd[dof_idx],
                    joint_dof_mode[dof_idx],
                )
                axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
                f_total += axis_world * (joint_f[dof_idx] + f)

                # rewrite the last value of err_prev now:
                wp.printf("--- Joint tid: %d, DofId: %d, Err: %0.2f, Err_prev: %0.2f, Err_i: %0.2f, Err_d: %0.2f \n", tid, dof_idx, err, joint_err_prev[dof_idx], joint_err_i[dof_idx], err_d)
                joint_err_prev[dof_idx] =  err

        # angular DOFs:
        else:
            if ang_axis_count > st_i - 3:
                dof_idx = ang_dof_start_offset + (st_i - 3)

                err, delta_err_i, err_d = calculate_errors(
                                            joint_target[dof_idx], 
                                            joint_q[dof_idx],
                                            joint_qd[dof_idx],
                                            joint_err_prev[dof_idx],
                                            joint_dof_mode[dof_idx],
                                            dt)
                
                # add to the overall integral error:
                wp.atomic_add(joint_err_i, dof_idx, delta_err_i)
                # CONCERN: what about windup effect? Maybe add anti-windup 

                t = joint_force(
                    joint_q[dof_idx],
                    joint_qd[dof_idx],
                    joint_target[dof_idx],
                    joint_target_ke[dof_idx],
                    joint_target_kd[dof_idx],
                    joint_target_ki[dof_idx],
                    err,
                    joint_err_i[dof_idx],
                    err_d,
                    joint_limit_lower[dof_idx],
                    joint_limit_upper[dof_idx],
                    joint_limit_ke[dof_idx],
                    joint_limit_kd[dof_idx],
                    joint_dof_mode[dof_idx],
                )
                axis_world = wp.transform_vector(X_wp, joint_axis[dof_idx])
                t_total += axis_world * (joint_f[dof_idx] + t)

                # rewrite the last value of err_prev now:
                wp.printf("--- Joint tid: %d, DofId: %d, Err: %0.2f, Err_prev: %0.2f, Err_i: %0.2f, Err_d: %0.2f \n", tid, dof_idx, err, joint_err_prev[dof_idx], joint_err_i[dof_idx], err_d)
                joint_err_prev[dof_idx] =  err

    # === Step 3: Apply Spatial Forces to Bodies ===
    # Apply force/torque to child body
    wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total)))

    # Apply equal and opposite force/torque to parent body
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))

    wp.printf("\n")


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
        
        wp.launch(
            kernel=apply_joint_control_kernel,
            dim=model.joint_count,
            inputs=[
                # State
                state_in.body_q,
                model.joint_q,
                model.joint_qd,
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
                control.joint_err_prev,
                control.joint_err_i,
                model.joint_dof_mode,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_target_ki,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                wp.float32(dt),
            ],
            outputs=[state_in.body_f],
            device=model.device,
        )
