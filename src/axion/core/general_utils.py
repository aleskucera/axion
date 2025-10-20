import warp as wp
from newton import Model

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_variables_kernel(
    alpha: wp.array(dtype=wp.float32),
    dbody_u: wp.array(dtype=wp.spatial_vector),
    dbody_lambda: wp.array(dtype=wp.float32),
    # Outputs
    body_u: wp.array(dtype=wp.spatial_vector),
    body_lambda: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    body_qd_dim = body_u.shape[0]
    lambda_dim = body_lambda.shape[0]

    if tid < body_qd_dim:
        idx = tid
        body_u[idx] += alpha[0] * dbody_u[idx]
    elif tid < body_qd_dim + lambda_dim:
        idx = tid - body_qd_dim
        body_lambda[idx] += alpha[0] * dbody_lambda[idx]
    else:
        return


@wp.kernel
def update_lambda_kernel(
    alpha: wp.array(dtype=wp.float32),
    dbody_lambda: wp.array(dtype=wp.float32),
    # Outputs
    body_lambda: wp.array(dtype=wp.float32),
):
    con_idx = wp.tid()

    body_lambda[con_idx] += alpha[0] * dbody_lambda[con_idx]


@wp.kernel
def update_body_qd_kernel(
    alpha: wp.array(dtype=wp.float32),
    dbody_u: wp.array(dtype=wp.spatial_vector),
    # Outputs
    body_u: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()

    body_u[body_idx] += alpha[0] * dbody_u[body_idx]


@wp.kernel
def update_body_q_kernel(
    body_u: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    body_q: wp.array(dtype=wp.transform),
):
    body_idx = wp.tid()

    v = wp.spatial_top(body_u[body_idx])
    w = wp.spatial_bottom(body_u[body_idx])

    x_prev = wp.transform_get_translation(body_q_prev[body_idx])
    r_prev = wp.transform_get_rotation(body_q_prev[body_idx])

    com = body_com[body_idx]
    x_com = x_prev + wp.quat_rotate(r_prev, com)

    x = x_com + v * dt
    r = wp.normalize(r_prev + wp.quat(w, 0.0) * r_prev * 0.5 * dt)

    body_q[body_idx] = wp.transform(x - wp.quat_rotate(r, com), r)


def update_variables(
    model: Model,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
    dt: float,
):
    device = data.device

    wp.launch(
        kernel=update_lambda_kernel,
        dim=dims.N_c,
        inputs=[
            data.alpha,
            data.dbody_lambda,
        ],
        outputs=[data.body_lambda],
        device=device,
    )

    wp.launch(
        kernel=update_body_qd_kernel,
        dim=dims.N_b,
        inputs=[
            data.alpha,
            data.dbody_u_v,
        ],
        outputs=[data.body_u],
        device=device,
    )


def update_body_q(
    model: Model,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
    dt: float,
):
    device = data.device

    wp.launch(
        kernel=update_body_q_kernel,
        dim=dims.N_b,
        inputs=[
            data.body_u,
            data.body_q_prev,
            model.body_com,
            dt,
        ],
        outputs=[data.body_q],
        device=device,
    )
