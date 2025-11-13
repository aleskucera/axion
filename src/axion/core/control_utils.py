from enum import IntEnum

import warp as wp
from newton import Control
from newton import JointType
from newton import Model
from newton import State


class JointMode(IntEnum):
    """
    Specifies the control mode for a joint's actuation.

    Joint modes determine how a joint is actuated or controlled during simulation.
    """

    NONE = 0
    """No implicit control is applied to the joint, but the joint can be controlled by applying forces."""

    TARGET_POSITION = 1
    """The joint is controlled to reach a target position."""

    TARGET_VELOCITY = 2
    """The joint is controlled to reach a target velocity."""


@wp.func
def calculate_error(
    q: wp.float32,
    qd: wp.float32,
    target: wp.float32,
    mode: wp.int32,
):
    if mode == JointMode.TARGET_POSITION:
        return target - q
    elif mode == JointMode.TARGET_VELOCITY:
        return target - qd
    elif mode == JointMode.NONE:
        return 0.0


@wp.kernel
def apply_joint_control_kernel(
    # --- State Inputs ---
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    # --- Model Inputs ---
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    # --- Control/Actuation Inputs ---
    joint_target: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    joint_err_prev: wp.array(dtype=wp.float32),
    joint_err_i: wp.array(dtype=wp.float32),
    joint_target_ke: wp.array(dtype=wp.float32),
    joint_target_kd: wp.array(dtype=wp.float32),
    joint_target_ki: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    joint_limit_ke: wp.array(dtype=wp.float32),
    joint_limit_kd: wp.array(dtype=wp.float32),
    dt: wp.float32,
    # --- Output ---
    body_f: wp.array(dtype=wp.spatial_vector),
):
    j_idx = wp.tid()
    joint_count = joint_type.shape[0]

    if j_idx >= joint_count or joint_enabled[j_idx] == 0:
        return

    j_type = joint_type[j_idx]

    # ======================== FIXED/D6/DISTANCE JOINT ========================
    # (supports no control)
    if j_type == JointType.FIXED or j_type == JointType.D6 or j_type == JointType.DISTANCE:
        return

    child = joint_child[j_idx]
    parent = joint_parent[j_idx]

    q_start = joint_q_start[j_idx]
    qd_start = joint_qd_start[j_idx]

    # ======================== FREE JOINT ========================
    # (supports only joint_f)
    if j_type == JointType.FREE:
        wrench = wp.spatial_vector(
            joint_f[qd_start + 0],
            joint_f[qd_start + 1],
            joint_f[qd_start + 2],
            joint_f[qd_start + 3],
            joint_f[qd_start + 4],
            joint_f[qd_start + 5],
        )

        wp.atomic_add(body_f, child, wrench)
        return

    # Parent kinematics
    X_p_w = wp.transform_identity()
    com_p = wp.vec3()
    if parent >= 0:
        X_p_w = body_q[parent]
        com_p = body_com[parent]

    X_pj_w = X_p_w * joint_X_p[j_idx]
    r_p_w = wp.transform_get_translation(X_pj_w) - wp.transform_point(X_p_w, com_p)

    # Child kinematics (this uses the corrected logic)
    X_c_w = body_q[child]
    com_c = body_com[child]
    X_cj_w = X_c_w * joint_X_c[j_idx]
    r_c_w = wp.transform_get_translation(X_cj_w) - wp.transform_point(X_c_w, com_c)

    # Initialize force and torque
    f_total = wp.vec3()
    t_total = wp.vec3()

    # ======================== BALL JOINT ========================
    # (supports only joint_f)
    if j_type == JointType.BALL:
        t_total = wp.vec3(
            joint_f[qd_start + 0],
            joint_f[qd_start + 1],
            joint_f[qd_start + 2],
        )
    # ======================== REVOLUTE/PRISMATIC JOINT ========================
    # (supports both joint_f and joint_target)
    elif j_type == JointType.REVOLUTE or j_type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        axis_w = wp.transform_vector(X_pj_w, axis)

        # ----------- Direct (raw) force -----------
        raw_f = joint_f[qd_start]

        # ----------- Control force -----------
        q = joint_q[q_start]
        qd = joint_qd[qd_start]
        target = joint_target[qd_start]

        mode = joint_dof_mode[qd_start]

        err = calculate_error(q, qd, target, mode)
        err_i = joint_err_i[qd_start]
        err_d = (err - joint_err_prev[qd_start]) / dt

        ke = joint_target_ke[qd_start]
        ki = joint_target_ki[qd_start]
        kd = joint_target_kd[qd_start]

        control_f = ke * err + ki * err_i + kd * err_d

        # TODO: Add anti-windup, limits etc.

        if j_type == JointType.REVOLUTE:
            t_total = axis_w * (raw_f + control_f)
        else:
            f_total = axis_w * (raw_f + control_f)

        joint_err_prev[qd_start] = err
        wp.atomic_add(joint_err_i, qd_start, err * dt)

    if parent >= 0:
        wp.atomic_sub(
            body_f, parent, wp.spatial_vector(f_total, t_total + wp.cross(r_p_w, f_total))
        )

    wp.atomic_add(body_f, child, wp.spatial_vector(f_total, t_total + wp.cross(r_c_w, f_total)))

    # wp.printf("Joint idx %d: (%f, %f, %f) \n", j_idx, t_total[0], t_total[1], t_total[2])


def apply_control(
    model: Model,
    state_in: State,
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
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_dof_mode,
                model.joint_axis,
                # Control
                control.joint_target,
                control.joint_f,
                control.joint_err_prev,
                control.joint_err_i,
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
