import warp as wp
from warp.sim import Model

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_variables_kernel(
    alpha: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda: wp.array(dtype=wp.float32),
    # Outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
    _lambda: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    body_qd_dim = body_qd.shape[0]
    lambda_dim = _lambda.shape[0]

    if tid < body_qd_dim:
        idx = tid
        body_qd[idx] += alpha[0] * delta_body_qd[idx]
    elif tid < body_qd_dim + lambda_dim:
        idx = tid - body_qd_dim
        _lambda[idx] += alpha[0] * delta_lambda[idx]
    else:
        return


@wp.kernel
def update_lambda_kernel(
    alpha: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    # Outputs
    _lambda: wp.array(dtype=wp.float32),
):
    con_idx = wp.tid()

    _lambda[con_idx] += alpha[0] * delta_lambda[con_idx]


@wp.kernel
def update_body_qd_kernel(
    alpha: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    # Outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()

    body_qd[body_idx] += alpha[0] * delta_body_qd[body_idx]


@wp.kernel
def update_body_q_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    body_q: wp.array(dtype=wp.transform),
):
    body_idx = wp.tid()

    w = wp.spatial_top(body_qd[body_idx])
    v = wp.spatial_bottom(body_qd[body_idx])

    x_prev = wp.transform_get_translation(body_q[body_idx])
    r_prev = wp.transform_get_rotation(body_q[body_idx])

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
        dim=dims.con_dim,
        inputs=[
            data.alpha,
            data.delta_lambda,
        ],
        outputs=[data._lambda],
        device=device,
    )

    wp.launch(
        kernel=update_body_qd_kernel,
        dim=dims.N_b,
        inputs=[
            data.alpha,
            data.delta_body_qd_v,
        ],
        outputs=[data.body_qd],
        device=device,
    )

    wp.launch(
        kernel=update_body_q_kernel,
        dim=dims.N_b,
        inputs=[
            data.body_qd,
            data.body_q_prev,
            model.body_com,
            dt,
        ],
        outputs=[data.body_q],
        device=device,
    )
