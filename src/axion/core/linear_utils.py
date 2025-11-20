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
    body_M_inv: wp.array(dtype=SpatialInertia),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector),
    h_c: wp.array(dtype=wp.float32),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Output array
    b: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    M_inv_1 = body_M_inv[body_1]
    M_inv_2 = body_M_inv[body_2]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]

    # Calculate (J_i * M^-1 * h_d)
    a_contrib = wp.dot(J_1, to_spatial_momentum(M_inv_1, h_d[body_1]))
    b_contrib = wp.dot(J_2, to_spatial_momentum(M_inv_2, h_d[body_2]))
    JHinvg = a_contrib + b_contrib

    # b = J * M^-1 * h_d - h_c
    b[constraint_idx] = JHinvg - h_c[constraint_idx]


@wp.kernel
def compute_JT_dbody_lambda_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    dbody_lambda: wp.array(dtype=wp.float32),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    # Output array
    JT_dbody_lambda: wp.array(dtype=wp.spatial_vector),
):
    constraint_idx = wp.tid()

    body_1 = constraint_body_idx[constraint_idx, 0]
    body_2 = constraint_body_idx[constraint_idx, 1]

    J_1 = J_values[constraint_idx, 0]
    J_2 = J_values[constraint_idx, 1]
    dlambda = dbody_lambda[constraint_idx]

    if body_1 >= 0:
        JT_dbody_lambda[body_1] += dlambda * J_1

    if body_2 >= 0:
        JT_dbody_lambda[body_2] += dlambda * J_2


@wp.kernel
def compute_dbody_u_kernel(
    body_M_inv: wp.array(dtype=SpatialInertia),
    JT_dbody_lambda: wp.array(dtype=wp.spatial_vector),
    h_d: wp.array(dtype=wp.spatial_vector),
    # Output array
    dbody_u: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()

    if body_idx >= body_M_inv.shape[0]:
        return

    dbody_u[body_idx] = to_spatial_momentum(
        body_M_inv[body_idx], (JT_dbody_lambda[body_idx] - h_d[body_idx])
    )


def compute_linear_system(
    model: Model,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
    dt: float,
):
    device = data.device

    data.h.zero_()
    data.J_values.zero_()
    data.C_values.zero_()
    data.JT_delta_lambda.zero_()
    data.b.zero_()

    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=dims.N_b,
        inputs=[
            data.body_u,
            data.body_u_prev,
            data.body_f,
            data.world_M,
            dt,
            data.g_accel,
        ],
        outputs=[data.h.d_spatial],
        device=device,
    )

    wp.launch(
        kernel=joint_constraint_kernel,
        dim=dims.N_j,
        inputs=[
            data.body_u,
            data.body_lambda.j,
            data.joint_constraint_data,
            dt,
            config.joint_stabilization_factor,
            config.joint_compliance,
        ],
        outputs=[
            data.h.d_spatial,
            data.h.c.j,
            data.J_values.j,
            data.C_values.j,
        ],
        device=device,
    )

    wp.launch(
        kernel=contact_constraint_kernel,
        dim=dims.N_n,
        inputs=[
            data.body_u,
            data.body_u_prev,
            data.body_lambda.n,
            data.contact_interaction,
            data.world_M_inv,
            dt,
            config.contact_stabilization_factor,
            config.contact_fb_alpha,
            config.contact_fb_beta,
            config.contact_compliance,
        ],
        outputs=[
            data.h.d_spatial,
            data.h.c.n,
            data.J_values.n,
            data.C_values.n,
            data.s_n,
        ],
        device=device,
    )

    wp.launch(
        kernel=friction_constraint_kernel,
        dim=dims.N_n,
        inputs=[
            data.body_u,
            data.body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.h.d_spatial,
            data.h.c.f,
            data.J_values.f,
            data.C_values.f,
        ],
        device=device,
    )

    wp.launch(
        kernel=update_system_rhs_kernel,
        dim=dims.N_c,
        inputs=[
            data.world_M_inv,
            data.J_values.full,
            data.h.d_spatial,
            data.h.c.full,
            data.constraint_body_idx.full,
        ],
        outputs=[data.b],
        device=device,
    )


def compute_dbody_qd_from_dbody_lambda(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=compute_JT_dbody_lambda_kernel,
        dim=dims.N_c,
        inputs=[
            data.J_values.full,
            data.dbody_lambda.full,
            data.constraint_body_idx.full,
        ],
        outputs=[data.JT_delta_lambda],
        device=device,
    )

    wp.launch(
        kernel=compute_dbody_u_kernel,
        dim=dims.N_b,
        inputs=[
            data.world_M_inv,
            data.JT_delta_lambda,
            data.h.d_spatial,
        ],
        outputs=[data.dbody_u_spatial],
        device=device,
    )
