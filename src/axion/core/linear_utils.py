import warp as wp
from axion.constraints import friction_constraint_kernel
from axion.constraints import positional_contact_constraint_kernel
from axion.constraints import positional_joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.constraints import velocity_contact_constraint_kernel
from axion.constraints import velocity_joint_constraint_kernel
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


@wp.kernel
def update_system_rhs_kernel(
    body_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_c: wp.array(dtype=wp.float32, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    # Output array
    b: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        b[world_idx, constraint_idx] = 0.0
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    JHinvg = 0.0
    if body_1 >= 0:
        M_inv_1 = body_M_inv[world_idx, body_1]
        J_1 = J_values[world_idx, constraint_idx, 0]
        JHinvg += wp.dot(J_1, to_spatial_momentum(M_inv_1, h_d[world_idx, body_1]))

    if body_2 >= 0:
        M_inv_2 = body_M_inv[world_idx, body_2]
        J_2 = J_values[world_idx, constraint_idx, 1]
        JHinvg += wp.dot(J_2, to_spatial_momentum(M_inv_2, h_d[world_idx, body_2]))

    # b = (J * M^-1 * h_d - h_c) / dt
    b[world_idx, constraint_idx] = (JHinvg - h_c[world_idx, constraint_idx]) / dt


@wp.kernel
def compute_JT_dbody_lambda_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    dbody_lambda: wp.array(dtype=wp.float32, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    # Output array
    JT_dbody_lambda: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]
    dlambda = dbody_lambda[world_idx, constraint_idx]

    if body_1 >= 0:
        wp.atomic_add(JT_dbody_lambda, world_idx, body_1, dlambda * J_1)

    if body_2 >= 0:
        wp.atomic_add(JT_dbody_lambda, world_idx, body_2, dlambda * J_2)


@wp.kernel
def compute_dbody_u_kernel(
    body_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    JT_dbody_lambda: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    dt: wp.float32,
    # Output array
    dbody_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()

    if body_idx >= body_M_inv.shape[1]:
        return

    dbody_u[world_idx, body_idx] = to_spatial_momentum(
        body_M_inv[world_idx, body_idx],
        (JT_dbody_lambda[world_idx, body_idx] * dt - h_d[world_idx, body_idx]),
    )


def compute_linear_system(
    model: AxionModel,
    data: EngineData,
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
        dim=(dims.N_w, dims.N_b),
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

    if config.joint_constraint_level == "pos":
        wp.launch(
            kernel=positional_joint_constraint_kernel,
            dim=(dims.N_w, dims.joint_count),
            inputs=[
                data.body_q,
                data.body_lambda.j,
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
                model.joint_compliance,
                data.dt,
                config.joint_compliance,
            ],
            outputs=[
                data.constraint_active_mask.j,
                data.h.d_spatial,
                data.h.c.j,
                data.J_values.j,
                data.C_values.j,
            ],
            device=device,
        )

    elif config.joint_constraint_level == "vel":
        wp.launch(
            kernel=velocity_joint_constraint_kernel,
            dim=(dims.N_w, dims.joint_count),
            inputs=[
                data.body_q,
                model.body_com,
                data.body_u,
                data.body_lambda.j,
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
            ],
            outputs=[
                data.constraint_active_mask.j,
                data.h.d_spatial,
                data.h.c.j,
                data.J_values.j,
                data.C_values.j,
            ],
            device=device,
        )
    else:
        raise ValueError(
            f"Joint constraint level can be only 'pos' or 'vel' ({config.joint_constraint_level})."
        )

    if config.contact_constraint_level == "pos":
        wp.launch(
            kernel=positional_contact_constraint_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.body_q,
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
                data.constraint_active_mask.n,
                data.h.d_spatial,
                data.h.c.n,
                data.J_values.n,
                data.C_values.n,
                data.s_n,
            ],
            device=device,
        )
    elif config.contact_constraint_level == "vel":
        wp.launch(
            kernel=velocity_contact_constraint_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.body_u,
                data.body_u_prev,
                data.body_lambda.n,
                data.contact_interaction,
                dt,
                config.contact_stabilization_factor,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.constraint_active_mask.n,
                data.h.d_spatial,
                data.h.c.n,
                data.J_values.n,
                data.C_values.n,
                data.s_n,
            ],
            device=device,
        )
    else:
        raise ValueError(
            f"Contact constraint level can be only 'pos' or 'vel' ({config.contact_constraint_level})."
        )

    wp.launch(
        kernel=friction_constraint_kernel,
        dim=(dims.N_w, dims.N_n),
        inputs=[
            data.body_u,
            data.body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            data.dt,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.constraint_active_mask.f,
            data.h.d_spatial,
            data.h.c.f,
            data.J_values.f,
            data.C_values.f,
        ],
        device=device,
    )

    wp.launch(
        kernel=update_system_rhs_kernel,
        dim=(dims.N_w, dims.N_c),
        inputs=[
            data.world_M_inv,
            data.J_values.full,
            data.h.d_spatial,
            data.h.c.full,
            data.constraint_body_idx.full,
            data.constraint_active_mask.full,
            data.dt,
        ],
        outputs=[data.b],
        device=device,
    )


def compute_dbody_qd_from_dbody_lambda(
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=compute_JT_dbody_lambda_kernel,
        dim=(dims.N_w, dims.N_c),
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
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.world_M_inv,
            data.JT_delta_lambda,
            data.h.d_spatial,
            data.dt,
        ],
        outputs=[data.dbody_u],
        device=device,
    )
