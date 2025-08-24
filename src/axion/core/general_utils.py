import warp as wp

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


def update_variables(data: EngineArrays, config: EngineConfig, dims: EngineDimensions):
    device = data.device

    wp.launch(
        kernel=update_variables_kernel,
        dim=dims.N_b + dims.con_dim,
        inputs=[
            data.alpha,
            data.delta_body_qd_v,
            data.delta_lambda,
        ],
        outputs=[data.body_qd, data._lambda],
        device=device,
    )
