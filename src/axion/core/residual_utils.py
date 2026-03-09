import warp as wp
from axion.constraints import contact_residual_kernel
from axion.constraints import control_residual_kernel
from axion.constraints import friction_residual_kernel
from axion.constraints import joint_residual_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.math import compute_spatial_momentum
from axion.math import compute_world_inertia

from .contacts import AxionContacts
from .types import JointMode


def compute_residual(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.ext_force,
            model.body_mass,
            model.body_inertia,
            data.dt,
            model.g_accel,
        ],
        outputs=[data.res.d_spatial],
        device=data.device,
    )

    wp.launch(
        kernel=joint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.constr_force.j,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_constraint_offsets,
            model.joint_compliance,
            data.dt,
            config.joint_compliance,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.j,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=control_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.constr_force.ctrl,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_dof_mode,
            model.control_constraint_offsets,
            data.joint_target_pos,
            data.joint_target_vel,
            model.joint_target_ke,
            model.joint_target_kd,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.ctrl,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=contact_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.body_pose_prev,
            data.constr_force.n,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.shape_body,
            contacts.contact_count,
            contacts.contact_shape0,
            contacts.contact_shape1,
            contacts.contact_point0,
            contacts.contact_point1,
            contacts.contact_thickness0,
            contacts.contact_thickness1,
            contacts.contact_normal,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.n,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=friction_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.body_vel,
            data.body_pose_prev,
            data.constr_force.f,
            data.constr_force_prev_iter.f,
            data.constr_force_prev_iter.n,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.shape_body,
            model.shape_material_mu,
            contacts.contact_count,
            contacts.contact_shape0,
            contacts.contact_shape1,
            contacts.contact_point0,
            contacts.contact_point1,
            contacts.contact_thickness0,
            contacts.contact_thickness1,
            contacts.contact_normal,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.f,
        ],
        device=data.device,
    )

    data.res.sync_to_float()


@wp.kernel
def body_vel_prev_grad_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    adjoint_vector: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Output ---
    body_vel_prev_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    if body_idx >= body_pose.shape[1]:
        return

    pose = body_pose[world_idx, body_idx]
    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]

    # Compute World Inertia
    I_world = compute_world_inertia(pose, I_body)

    w = adjoint_vector[world_idx, body_idx]

    body_vel_prev_grad[world_idx, body_idx] += compute_spatial_momentum(-m, -I_world, w)


@wp.kernel
def control_target_grad_kernel(
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    w_ctrl: wp.array(dtype=wp.float32, ndim=2),  # Adjoint vector for control constraints
    dt: wp.float32,
    # Outputs:
    target_pos_grad: wp.array(dtype=wp.float32, ndim=2),
    target_vel_grad: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]

    # Extract the adjoint variable w_lambda corresponding to this constraint row
    w_lambda = w_ctrl[world_idx, ctrl_offset]

    if mode == JointMode.TARGET_POSITION:
        # gradient = w_lambda * (-1.0 / dt)
        wp.atomic_add(target_pos_grad, world_idx, qd_start_idx, w_lambda * (-1.0 / dt))

    elif mode == JointMode.TARGET_VELOCITY:
        # gradient = w_lambda * (-1.0)
        wp.atomic_add(target_vel_grad, world_idx, qd_start_idx, w_lambda * (-1.0))


def compute_residual_gradient(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):

    wp.launch(
        kernel=body_vel_prev_grad_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            model.body_mass,
            model.body_inertia,
            data.w.d_spatial,
        ],
        outputs=[data.body_vel_prev.grad],
        device=data.device,
    )

    wp.launch(
        kernel=control_target_grad_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            model.joint_type,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_dof_mode,
            model.control_constraint_offsets,
            data.w.c.ctrl,
            data.dt,
        ],
        outputs=[
            data.joint_target_pos.grad,
            data.joint_target_vel.grad,
        ],
        device=data.device,
    )
