import warp as wp
from axion.constraints import linesearch_contact_residuals_kernel
from axion.constraints import linesearch_dynamics_residuals_kernel
from axion.constraints import linesearch_friction_residuals_kernel
from axion.constraints import linesearch_joint_residuals_kernel

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_sq_norm(
    res_alpha: wp.array(dtype=wp.float32, ndim=2),
    res_alpha_norm_sq: wp.array(dtype=wp.float32),
):
    alpha_idx = wp.tid()

    norm_sq = float(0.0)
    for i in range(res_alpha.shape[1]):
        norm_sq += wp.pow(res_alpha[alpha_idx, i], 2.0)

    res_alpha_norm_sq[alpha_idx] = norm_sq


@wp.kernel
def update_alpha(
    alphas: wp.array(dtype=wp.float32),
    res_alpha_norm_sq: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid > 0:
        return

    best_idx = wp.uint32(0)
    min_value = wp.float32(3.4e38)  # Largest finite float32 value

    for i in range(res_alpha_norm_sq.shape[0]):
        value = res_alpha_norm_sq[i]
        if value < min_value:
            best_idx = wp.uint32(i)
            min_value = value

    alpha[0] = alphas[best_idx]


def perform_linesearch(data: EngineArrays, config: EngineConfig, dims: EngineDimensions, dt: float):
    device = data.device
    data.g_alpha.zero_()

    # =================== UPDATE RESIDUALS ===================
    wp.launch(
        kernel=linesearch_dynamics_residuals_kernel,
        dim=(dims.N_alpha, dims.N_b),
        inputs=[
            data.alphas,
            data.delta_body_qd_v,
            # ---
            data.body_qd,
            data.body_qd_prev,
            data.body_f,
            data.gen_mass,
            dt,
            data.g_accel,
        ],
        outputs=[data.g_alpha_v],
        device=device,
    )

    wp.launch(
        kernel=linesearch_joint_residuals_kernel,
        dim=(dims.N_alpha, 5, dims.N_j),
        inputs=[
            data.alphas,
            data.delta_body_qd_v,
            data.delta_lambda_j,
            # ---
            data.body_qd,
            data.lambda_j,
            data.joint_interaction,
            # Parameters
            dt,
            config.joint_stabilization_factor,
        ],
        outputs=[data.g_alpha_v, data.h_alpha_j],
        device=device,
    )

    wp.launch(
        kernel=linesearch_contact_residuals_kernel,
        dim=(dims.N_alpha, dims.N_c),
        inputs=[
            data.alphas,
            data.delta_body_qd_v,
            data.delta_lambda_n,
            # ---
            data.body_qd,
            data.body_qd_prev,
            data.lambda_n,
            data.contact_interaction,
            # Parameters
            dt,
            config.contact_stabilization_factor,
            config.contact_fb_alpha,
            config.contact_fb_beta,
            config.contact_compliance,
        ],
        outputs=[data.g_alpha_v, data.h_alpha_n],
        device=device,
    )

    wp.launch(
        kernel=linesearch_friction_residuals_kernel,
        dim=(dims.N_alpha, dims.N_c),
        inputs=[
            data.alphas,
            data.delta_body_qd_v,
            data.delta_lambda_f,
            data.delta_lambda_n,
            # ---
            data.body_qd,
            data.lambda_f,
            data.lambda_n,
            data.contact_interaction,
            # Parameters
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[data.g_alpha_v, data.h_alpha_f],
        device=device,
    )

    # =================== FIND BEST STEP SIZE (ALPHA) ===================
    wp.launch(
        kernel=update_sq_norm,
        dim=dims.N_alpha,
        inputs=[data.res_alpha],
        outputs=[data.res_alpha_norm_sq],
        device=device,
    )

    wp.launch(
        kernel=update_alpha,
        dim=1,
        inputs=[data.alphas, data.res_alpha_norm_sq],
        outputs=[data.alpha],
        device=device,
    )
