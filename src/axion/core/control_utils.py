import warp as wp
from warp.sim import Control
from warp.sim import Model
from warp.sim import State


@wp.kernel
def apply_joint_actions_kernel(
    body_q: wp.array(dtype=wp.transform),  # [body_count]
    body_com: wp.array(dtype=wp.vec3),  # [body_count]
    joint_q: wp.array(dtype=float),  # [joint_coord_count]
    joint_qd: wp.array(dtype=float),  # [joint_coord_count]
    joint_ke: wp.array(dtype=float),  # [joint_axis_count]
    joint_kd: wp.array(dtype=float),  # [joint_axis_count]
    joint_type: wp.array(dtype=int),  # [joint_count]
    joint_enabled: wp.array(dtype=int),  # [joint_count]
    joint_parent: wp.array(dtype=int),  # [joint_count]
    joint_child: wp.array(dtype=int),  # [joint_count]
    joint_X_p: wp.array(dtype=wp.transform),  # [joint_count]
    joint_X_c: wp.array(dtype=wp.transform),  # [joint_count]
    joint_axis_start: wp.array(dtype=int),  # [joint_count]
    joint_axis_dim: wp.array(dtype=int, ndim=2),  # [joint_axis_count, 2]
    joint_axis: wp.array(dtype=wp.vec3),  # [joint_axis_count]
    joint_axis_mode: wp.array(dtype=int),  # [joint_axis_count]
    joint_act: wp.array(dtype=float),  # [joint_axis_count]
    # --- Outputs ---
    body_f: wp.array(dtype=wp.spatial_vector),  # [body_count]
):
    tid = wp.tid()
    j_type = joint_type[tid]
    if joint_enabled[tid] == 0 or j_type != wp.sim.JOINT_REVOLUTE:
        return

    # rigid body indices of the child and parent
    child_idx = joint_child[tid]
    parent_idx = joint_parent[tid]

    is_parent_dynamic = parent_idx >= 0

    # Kinematics (Parent)
    X_wj_p = joint_X_p[tid]
    if is_parent_dynamic:
        body_q_p = body_q[parent_idx]
        X_wj_p = body_q_p * joint_X_p[tid]

    # Joint properties (only for revolute joints)
    axis_start = joint_axis_start[tid]
    mode = joint_axis_mode[axis_start]
    axis = joint_axis[axis_start]
    act = joint_act[axis_start]

    # handle angular constraints
    a_w_p = wp.transform_vector(X_wj_p, axis)
    torque = wp.vec3()
    if mode == wp.sim.JOINT_MODE_FORCE:
        torque = act * a_w_p
    elif mode == wp.sim.JOINT_MODE_TARGET_VELOCITY:
        control_val = joint_ke[axis_start] * (act - joint_qd[axis_start])
        torque = control_val * a_w_p

    # write forces
    if is_parent_dynamic:
        wp.atomic_sub(body_f, parent_idx, wp.spatial_vector(torque, wp.vec3()))
    wp.atomic_add(body_f, child_idx, wp.spatial_vector(torque, wp.vec3()))


def apply_control(
    model: Model,
    state_in: State,
    state_out: State,
    dt: float,
    control: Control | None = None,
):
    # The main simulation loop logic remains unchanged
    wp.sim.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
    if control is not None:
        wp.launch(
            kernel=apply_joint_actions_kernel,
            dim=(model.joint_count,),
            inputs=[
                state_in.body_q,
                model.body_com,
                state_in.joint_q,
                state_in.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis_start,
                model.joint_axis_dim,
                model.joint_axis,
                model.joint_axis_mode,
                control.joint_act,
            ],
            outputs=[state_in.body_f],
        )
