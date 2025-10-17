import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.constraints import friction_constraint_kernel
from axion.constraints import joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum
from newton import Model

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_system_rhs_kernel(
    Hinv: wp.array(dtype=SpatialInertia),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    g: wp.array(dtype=wp.spatial_vector),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    # Calculate (J_i * H^-1 * g)
    a_contrib = wp.dot(J_ia, to_spatial_momentum(Hinv[body_a], g[body_a]))
    b_contrib = wp.dot(J_ib, to_spatial_momentum(Hinv[body_b], g[body_b]))
    JHinvg = a_contrib + b_contrib

    # b = J * H^-1 * g - h
    b[constraint_idx] = JHinvg - h[constraint_idx]


@wp.kernel
def compute_JT_delta_lambda_kernel(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    delta_lambda: wp.array(dtype=wp.float32),
    # Output array
    JT_delta_lambda: wp.array(dtype=wp.spatial_vector),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]
    dl = delta_lambda[constraint_idx]

    if body_a >= 0:
        JT_delta_lambda[body_a] += dl * J_ia

    if body_b >= 0:
        JT_delta_lambda[body_b] += dl * J_ib


@wp.kernel
def compute_delta_body_qd_kernel(
    gen_inv_mass: wp.array(dtype=SpatialInertia),
    JT_delta_lambda: wp.array(dtype=wp.spatial_vector),
    g: wp.array(dtype=wp.spatial_vector),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()

    if body_idx >= gen_inv_mass.shape[0]:
        return

    delta_body_qd[body_idx] = to_spatial_momentum(
        gen_inv_mass[body_idx], (JT_delta_lambda[body_idx] - g[body_idx])
    )


def compute_linear_system(
    model: Model, data: EngineArrays, config: EngineConfig, dims: EngineDimensions, dt: float
):
    device = data.device

    data.g.zero_()
    data.h.zero_()
    data.J_values.zero_()
    data.C_values.zero_()
    data.JT_delta_lambda.zero_()

    data.b.zero_()

    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=dims.N_b,
        inputs=[
            data.body_qd,
            data.body_qd_prev,
            data.body_f,
            data.gen_mass,
            dt,
            data.g_accel,
        ],
        outputs=[data.g_v],
        device=device,
    )

    wp.launch(
        kernel=joint_constraint_kernel,
        dim=(5, dims.N_j),
        inputs=[
            data.body_qd,
            data.lambda_j,
            data.joint_interaction,
            dt,
            config.joint_stabilization_factor,
        ],
        outputs=[
            data.g_v,
            data.h_j,
            data.J_j_values,
            data.C_j_values,
        ],
        device=device,
    )

    wp.launch(
        kernel=contact_constraint_kernel,
        dim=dims.N_c,
        inputs=[
            data.body_qd,
            data.body_qd_prev,
            data.lambda_n,
            data.contact_interaction,
            data.gen_inv_mass,
            dt,
            config.contact_stabilization_factor,
            config.contact_fb_alpha,
            config.contact_fb_beta,
            config.contact_compliance,
        ],
        outputs=[
            data.lambda_n_scale,
            data.g_v,
            data.h_n,
            data.J_n_values,
            data.C_n_values,
        ],
        device=device,
    )

    wp.launch(
        kernel=friction_constraint_kernel,
        dim=dims.N_c,
        inputs=[
            data.body_qd,
            data.lambda_f,
            data.lambda_f_prev,
            data.lambda_n_prev,
            data.lambda_n_scale_prev,
            data.contact_interaction,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.g_v,
            data.h_f,
            data.J_f_values,
            data.C_f_values,
        ],
        device=device,
    )

    wp.launch(
        kernel=update_system_rhs_kernel,
        dim=(dims.con_dim,),
        inputs=[
            data.gen_inv_mass,
            data.constraint_body_idx,
            data.J_values,
            data.g_v,
            data.h,
        ],
        outputs=[data.b],
        device=device,
    )


def compute_delta_body_qd_from_delta_lambda(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=compute_JT_delta_lambda_kernel,
        dim=dims.con_dim,
        inputs=[
            data.constraint_body_idx,
            data.J_values,
            data.delta_lambda,
        ],
        outputs=[data.JT_delta_lambda],
        device=device,
    )

    wp.launch(
        kernel=compute_delta_body_qd_kernel,
        dim=dims.dyn_dim,
        inputs=[
            data.gen_inv_mass,
            data.JT_delta_lambda,
            data.g_v,
        ],
        outputs=[data.delta_body_qd_v],
        device=device,
    )
