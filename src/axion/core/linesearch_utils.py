import warp as wp
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_positional_contact_residual_kernel
from axion.constraints import batch_positional_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import batch_velocity_contact_residual_kernel
from axion.constraints import batch_velocity_joint_residual_kernel
from axion.constraints import fused_batch_friction_residual_kernel
from axion.constraints import fused_batch_positional_contact_residual_kernel
from axion.constraints import fused_batch_positional_joint_residual_kernel
from axion.constraints import fused_batch_unconstrained_dynamics_kernel
from axion.constraints import fused_batch_velocity_contact_residual_kernel
from axion.constraints import fused_batch_velocity_joint_residual_kernel
from axion.constraints.control_constraint import batch_control_constraint_residual_kernel
from axion.constraints.control_constraint import fused_batch_control_constraint_residual_kernel

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


@wp.kernel
def compute_linesearch_batch_body_lambda_kernel(
    body_lambda: wp.array(dtype=wp.float32, ndim=2),
    dbody_lambda: wp.array(dtype=wp.float32, ndim=2),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_batch_body_lambda: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, con_idx = wp.tid()

    lambda_ = body_lambda[world_idx, con_idx]
    dlambda = dbody_lambda[world_idx, con_idx]
    alpha = linesearch_steps[batch_idx]

    linesearch_batch_body_lambda[batch_idx, world_idx, con_idx] = lambda_ + alpha * dlambda


@wp.kernel
def compute_body_lambda_without_linesearch_kernel(
    dbody_lambda: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    body_lambda: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, con_idx = wp.tid()

    lambda_ = body_lambda[world_idx, con_idx]
    dlambda = dbody_lambda[world_idx, con_idx]

    body_lambda[world_idx, con_idx] = lambda_ + dlambda


@wp.kernel
def compute_linesearch_batch_body_u_kernel(
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    dbody_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()

    u = body_u[world_idx, body_idx]
    du = dbody_u[world_idx, body_idx]
    alpha = linesearch_steps[batch_idx]

    linesearch_batch_body_u[batch_idx, world_idx, body_idx] = u + alpha * du


@wp.kernel
def compute_linesearch_batch_body_q_kernel(
    batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_q_prev: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    dt: wp.float32,
    # Outputs
    batch_body_q: wp.array(dtype=wp.transform, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()

    v = wp.spatial_top(batch_body_u[batch_idx, world_idx, body_idx])
    w = wp.spatial_bottom(batch_body_u[batch_idx, world_idx, body_idx])

    x_prev = wp.transform_get_translation(body_q_prev[world_idx, body_idx])
    r_prev = wp.transform_get_rotation(body_q_prev[world_idx, body_idx])

    com = body_com[world_idx, body_idx]
    x_com = x_prev + wp.quat_rotate(r_prev, com)

    x = x_com + v * dt
    r = wp.normalize(r_prev + wp.quat(w, 0.0) * r_prev * 0.5 * dt)

    batch_body_q[batch_idx, world_idx, body_idx] = wp.transform(x - wp.quat_rotate(r, com), r)


@wp.kernel
def compute_body_u_without_linesearch_kernel(
    dbody_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Outputs
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()

    u = body_u[world_idx, body_idx]
    du = dbody_u[world_idx, body_idx]

    body_u[world_idx, body_idx] = u + du


@wp.kernel
def copy_batch_sample_body_u_kernel(
    batch_array: wp.array(dtype=wp.spatial_vector, ndim=3),
    batch_idx_array: wp.array(dtype=wp.int32),
    # Outputs
    output_array: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    batch_idx = batch_idx_array[world_idx]
    output_array[world_idx, body_idx] = batch_array[batch_idx, world_idx, body_idx]


@wp.kernel
def copy_batch_sample_body_q_kernel(
    batch_array: wp.array(dtype=wp.transform, ndim=3),
    batch_idx_array: wp.array(dtype=wp.int32),
    # Outputs
    output_array: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()
    batch_idx = batch_idx_array[world_idx]
    output_array[world_idx, body_idx] = batch_array[batch_idx, world_idx, body_idx]


@wp.kernel
def copy_batch_sample_body_lambda_kernel(
    batch_array: wp.array(dtype=wp.float32, ndim=3),
    batch_idx_array: wp.array(dtype=wp.int32),
    # Outputs
    output_array: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, con_idx = wp.tid()
    batch_idx = batch_idx_array[world_idx]
    output_array[world_idx, con_idx] = batch_array[batch_idx, world_idx, con_idx]


@wp.kernel
def find_minimal_residual_index_kernel(
    batch_h_norm_sq: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    minimal_index: wp.array(dtype=wp.int32),
):
    world_idx = wp.tid()

    min_idx = wp.int32(0)
    min_value = batch_h_norm_sq[0, world_idx]

    # Iterate over batch dimension (dim 0)
    for i in range(1, batch_h_norm_sq.shape[0]):
        value = batch_h_norm_sq[i, world_idx]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value
    minimal_index[world_idx] = min_idx


@wp.kernel
def update_body_q_kernel(
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_q_prev: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    dt: wp.float32,
    body_q: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()

    v = wp.spatial_top(body_u[world_idx, body_idx])
    w = wp.spatial_bottom(body_u[world_idx, body_idx])

    x_prev = wp.transform_get_translation(body_q_prev[world_idx, body_idx])
    r_prev = wp.transform_get_rotation(body_q_prev[world_idx, body_idx])

    com = body_com[world_idx, body_idx]
    x_com = x_prev + wp.quat_rotate(r_prev, com)

    x = x_com + v * dt
    r = wp.normalize(r_prev + wp.quat(w, 0.0) * r_prev * 0.5 * dt)

    body_q[world_idx, body_idx] = wp.transform(x - wp.quat_rotate(r, com), r)


def update_body_q(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=update_body_q_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.body_u,
            data.body_q_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.body_q],
        device=device,
    )


def compute_linesearch_batch_variables(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    B = data.linesearch.batch_body_u.shape[0]

    # 1. Compute candidates for U
    wp.launch(
        kernel=compute_linesearch_batch_body_u_kernel,
        dim=(B, dims.N_w, dims.N_b),
        inputs=[
            data.body_u,
            data.dbody_u,
            data.linesearch.steps,
        ],
        outputs=[data.linesearch.batch_body_u],
        device=device,
    )

    # 2. Compute candidates for Q (integration)
    wp.launch(
        kernel=compute_linesearch_batch_body_q_kernel,
        dim=(B, dims.N_w, dims.N_b),
        inputs=[
            data.linesearch.batch_body_u,
            data.body_q_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.linesearch.batch_body_q],
        device=device,
    )

    # 3. Compute candidates for Lambda
    wp.launch(
        kernel=compute_linesearch_batch_body_lambda_kernel,
        dim=(B, dims.N_w, dims.N_c),
        inputs=[
            data.body_lambda.full,
            data.dbody_lambda.full,
            data.linesearch.steps,
        ],
        outputs=[data.linesearch.batch_body_lambda.full],
        device=device,
    )


def update_variables_without_linesearch(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=compute_body_u_without_linesearch_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.dbody_u,
        ],
        outputs=[data.body_u],
        device=device,
    )

    # Also update Q
    update_body_q(model, data, config, dims)

    wp.launch(
        kernel=compute_body_lambda_without_linesearch_kernel,
        dim=(dims.N_w, dims.N_c),
        inputs=[
            data.dbody_lambda.full,
        ],
        outputs=[data.body_lambda.full],
        device=device,
    )


def compute_linesearch_batch_h(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    data.linesearch.batch_h.full.zero_()
    data.linesearch.batch_h_norm_sq.zero_()

    B = data.linesearch.batch_body_u.shape[0]

    # Evaluate residual for unconstrained dynamics
    wp.launch(
        kernel=fused_batch_unconstrained_dynamics_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.linesearch.batch_body_q,
            data.linesearch.batch_body_u,
            data.body_u_prev,
            data.body_f,
            model.body_mass,
            model.body_inertia,
            data.dt,
            data.g_accel,
            B,
        ],
        outputs=[data.linesearch.batch_h.d_spatial],
        device=device,
    )

    # Evaluate residual for joint constraints
    if config.joint_constraint_level == "pos":
        wp.launch(
            kernel=fused_batch_positional_joint_residual_kernel,
            dim=(dims.N_w, dims.joint_count),
            inputs=[
                data.linesearch.batch_body_q,
                data.linesearch.batch_body_lambda.j,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                model.joint_enabled,
                data.joint_constraint_offsets,
                data.dt,
                config.joint_compliance,
                B,
            ],
            outputs=[
                data.linesearch.batch_h.d_spatial,
                data.linesearch.batch_h.c.j,
            ],
            device=device,
        )
    elif config.joint_constraint_level == "vel":
        wp.launch(
            kernel=fused_batch_velocity_joint_residual_kernel,
            dim=(dims.N_w, dims.joint_count),
            inputs=[
                data.body_q,
                model.body_com,
                data.linesearch.batch_body_u,
                data.linesearch.batch_body_lambda.j,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                model.joint_enabled,
                data.joint_constraint_offsets,
                data.dt,
                config.joint_stabilization_factor,
                config.joint_compliance,
                B,
            ],
            outputs=[
                data.linesearch.batch_h.d_spatial,
                data.linesearch.batch_h.c.j,
            ],
            device=device,
        )
    else:
        raise ValueError("Joint constraint level can be only 'pos' or 'vel'.")

    # Evaluate residual for control constraints
    wp.launch(
        kernel=fused_batch_control_constraint_residual_kernel,
        dim=(dims.N_w, dims.joint_count),
        inputs=[
            data.linesearch.batch_body_q,
            data.linesearch.batch_body_u,
            data.linesearch.batch_body_lambda.ctrl,
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
            data.control_constraint_offsets,
            data.joint_target,
            model.joint_target_ke,
            model.joint_target_kd,
            data.dt,
            B,
        ],
        outputs=[
            data.linesearch.batch_h.d_spatial,
            data.linesearch.batch_h.c.ctrl,
        ],
        device=device,
    )

    # Evaluate residual for normal contact constraints
    if config.contact_constraint_level == "pos":
        wp.launch(
            kernel=fused_batch_positional_contact_residual_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.linesearch.batch_body_q,
                data.linesearch.batch_body_u,
                data.body_u_prev,
                data.linesearch.batch_body_lambda.n,
                data.contact_interaction,
                data.world_M_inv,
                data.dt,
                config.contact_compliance,
                B,
            ],
            outputs=[
                data.linesearch.batch_h.d_spatial,
                data.linesearch.batch_h.c.n,
            ],
            device=device,
        )
    elif config.contact_constraint_level == "vel":
        wp.launch(
            kernel=fused_batch_velocity_contact_residual_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.linesearch.batch_body_u,
                data.body_u_prev,
                data.linesearch.batch_body_lambda.n,
                data.contact_interaction,
                data.dt,
                config.contact_stabilization_factor,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
                B,
            ],
            outputs=[
                data.linesearch.batch_h.d_spatial,
                data.linesearch.batch_h.c.n,
            ],
            device=device,
        )
    else:
        raise ValueError("Contact constraint level can be only 'pos' or 'vel'.")

    # Evaluate residual for friction constraints
    wp.launch(
        kernel=fused_batch_friction_residual_kernel,
        dim=(dims.N_w, dims.N_n),
        inputs=[
            data.linesearch.batch_body_u,
            data.linesearch.batch_body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            data.world_M_inv,
            data.dt,
            config.friction_compliance,
            B,
        ],
        outputs=[
            data.linesearch.batch_h.d_spatial,
            data.linesearch.batch_h.c.f,
        ],
        device=device,
    )


def select_minimal_residual_variables(
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    B = data.linesearch.batch_h.full.shape[0]

    # Compute norm squared for each batch using TiledSqNorm
    data.linesearch.tiled_sq_norm.compute(
        data.linesearch.batch_h.full, data.linesearch.batch_h_norm_sq
    )

    # Find the index with minimal residual norm (per world)
    wp.launch(
        kernel=find_minimal_residual_index_kernel,
        dim=dims.N_w,
        inputs=[data.linesearch.batch_h_norm_sq],
        outputs=[data.linesearch.minimal_index],
        device=device,
    )

    # Copy the minimal residual state variables back
    wp.launch(
        kernel=copy_batch_sample_body_u_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.linesearch.batch_body_u,
            data.linesearch.minimal_index,
        ],
        outputs=[data.body_u],
        device=device,
    )

    wp.launch(
        kernel=copy_batch_sample_body_q_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.linesearch.batch_body_q,
            data.linesearch.minimal_index,
        ],
        outputs=[data.body_q],
        device=device,
    )

    wp.launch(
        kernel=copy_batch_sample_body_lambda_kernel,
        dim=(dims.N_w, dims.N_c),
        inputs=[
            data.linesearch.batch_body_lambda.full,
            data.linesearch.minimal_index,
        ],
        outputs=[data.body_lambda.full],
        device=device,
    )


def perform_linesearch(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    if config.enable_linesearch:
        compute_linesearch_batch_variables(model, data, config, dims)
        compute_linesearch_batch_h(model, data, config, dims)
        select_minimal_residual_variables(data, config, dims)
    else:
        update_variables_without_linesearch(model, data, config, dims)
