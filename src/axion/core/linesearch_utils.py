import warp as wp
from axion.constraints import batch_contact_residual_kernel
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel

from .batched_model import BatchedModel
from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def compute_linesearch_batch_body_lambda_kernel(
    body_lambda: wp.array(dtype=wp.float32),
    dbody_lambda: wp.array(dtype=wp.float32),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_batch_body_lambda: wp.array(dtype=wp.float32, ndim=2),
):
    batch_idx, con_idx = wp.tid()

    lambda_ = body_lambda[con_idx]
    dlambda = dbody_lambda[con_idx]
    alpha = linesearch_steps[batch_idx]

    linesearch_batch_body_lambda[batch_idx, con_idx] = lambda_ + alpha * dlambda


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
    body_u: wp.array(dtype=wp.spatial_vector),
    dbody_u: wp.array(dtype=wp.spatial_vector),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    batch_idx, body_idx = wp.tid()

    u = body_u[body_idx]
    du = dbody_u[body_idx]
    alpha = linesearch_steps[batch_idx]

    linesearch_batch_body_u[batch_idx, body_idx] = u + alpha * du


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
def copy_batch_row_body_u_kernel(
    batch_array: wp.array(dtype=wp.spatial_vector, ndim=2),
    batch_idx_array: wp.array(dtype=wp.int32),
    # Outputs
    output_array: wp.array(dtype=wp.spatial_vector),
):
    idx = wp.tid()
    batch_idx = batch_idx_array[0]
    output_array[idx] = batch_array[batch_idx, idx]


@wp.kernel
def copy_batch_row_body_lambda_kernel(
    batch_array: wp.array(dtype=wp.float32, ndim=2),
    batch_idx_array: wp.array(dtype=wp.int32),
    # Outputs
    output_array: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    batch_idx = batch_idx_array[0]
    output_array[idx] = batch_array[batch_idx, idx]


@wp.kernel
def compute_batch_h_norm_squared_kernel(
    linesearch_batch_h: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    batch_h_norm_sq: wp.array(dtype=wp.float32),
):
    batch_idx = wp.tid()

    norm_sq = wp.float32(0.0)
    for i in range(linesearch_batch_h.shape[1]):
        val = linesearch_batch_h[batch_idx, i]
        norm_sq += val * val

    batch_h_norm_sq[batch_idx] = norm_sq


@wp.kernel
def find_minimal_residual_index_kernel(
    batch_h_norm_sq: wp.array(dtype=wp.float32),
    # Outputs
    minimal_index: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid > 0:
        return

    min_idx = wp.int32(0)
    min_value = batch_h_norm_sq[0]

    for i in range(1, batch_h_norm_sq.shape[0]):
        value = batch_h_norm_sq[i]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value
    minimal_index[0] = min_idx


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
    model: BatchedModel,
    data: EngineArrays,
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
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    B = data.linesearch_batch_body_u.shape[0]

    wp.launch(
        kernel=compute_linesearch_batch_body_u_kernel,
        dim=(B, dims.N_b),
        inputs=[
            data.body_u,
            data.dbody_u,
            data.linesearch_steps,
        ],
        outputs=[data.linesearch_batch_body_u],
        device=device,
    )

    wp.launch(
        kernel=compute_linesearch_batch_body_lambda_kernel,
        dim=(B, dims.N_c),
        inputs=[
            data.body_lambda.full,
            data.dbody_lambda.full,
            data.linesearch_steps,
        ],
        outputs=[data.linesearch_batch_body_lambda.full],
        device=device,
    )


def update_variables_without_linesearch(
    data: EngineArrays,
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
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    data.linesearch_batch_h.full.zero_()
    data.linesearch_batch_h_norm_sq.zero_()

    B = data.linesearch_batch_body_u.shape[0]

    # Evaluate residual for unconstrained dynamics
    wp.launch(
        kernel=batch_unconstrained_dynamics_kernel,
        dim=(B, dims.N_b),
        inputs=[
            data.linesearch_batch_body_u,
            data.body_u_prev,
            data.body_f,
            data.body_M,
            data.dt,
            data.g_accel,
        ],
        outputs=[data.linesearch_batch_h.d_spatial],
        device=device,
    )

    # Evaluate residual for joint constraints
    wp.launch(
        kernel=batch_joint_residual_kernel,
        dim=(B, dims.N_j),
        inputs=[
            data.linesearch_batch_body_u,
            data.linesearch_batch_body_lambda.j,
            data.joint_constraint_data,
            data.dt,
            config.joint_stabilization_factor,
            config.joint_compliance,
        ],
        outputs=[
            data.linesearch_batch_h.d_spatial,
            data.linesearch_batch_h.c.j,
        ],
        device=device,
    )

    # Evaluate residual for normal contact constraints
    wp.launch(
        kernel=batch_contact_residual_kernel,
        dim=(B, dims.N_n),
        inputs=[
            data.linesearch_batch_body_u,
            data.body_u_prev,
            data.linesearch_batch_body_lambda.n,
            data.contact_interaction,
            data.body_M_inv,
            data.dt,
            config.contact_stabilization_factor,
            config.contact_fb_alpha,
            config.contact_fb_beta,
            config.contact_compliance,
        ],
        outputs=[
            data.linesearch_batch_h.d_spatial,
            data.linesearch_batch_h.c.n,
        ],
        device=device,
    )
    # Evaluate residual for friction constraints
    wp.launch(
        kernel=batch_friction_residual_kernel,
        dim=(B, dims.N_n),
        inputs=[
            data.linesearch_batch_body_u,
            data.linesearch_batch_body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.linesearch_batch_h.d_spatial,
            data.linesearch_batch_h.c.f,
        ],
        device=device,
    )


def select_minimal_residual_variables(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    B = data.linesearch_batch_h.full.shape[0]

    # Compute norm squared for each batch using Warp kernels
    wp.launch(
        kernel=compute_batch_h_norm_squared_kernel,
        dim=B,
        inputs=[data.linesearch_batch_h.full],
        outputs=[data.linesearch_batch_h_norm_sq],
        device=device,
    )

    # Find the index with minimal residual norm
    wp.launch(
        kernel=find_minimal_residual_index_kernel,
        dim=1,
        inputs=[data.linesearch_batch_h_norm_sq],
        outputs=[data.linesearch_minimal_index],
        device=device,
    )

    # Copy the minimal residual state variables back
    wp.launch(
        kernel=copy_batch_row_body_u_kernel,
        dim=dims.N_b,
        inputs=[
            data.linesearch_batch_body_u,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_u],
        device=device,
    )

    wp.launch(
        kernel=copy_batch_row_body_lambda_kernel,
        dim=dims.N_c,
        inputs=[
            data.linesearch_batch_body_lambda.full,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_lambda.full],
        device=device,
    )


def perform_linesearch(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    if config.enable_linesearch:
        compute_linesearch_batch_variables(data, config, dims)
        compute_linesearch_batch_h(data, config, dims)
        select_minimal_residual_variables(data, config, dims)
    else:
        update_variables_without_linesearch(data, config, dims)
