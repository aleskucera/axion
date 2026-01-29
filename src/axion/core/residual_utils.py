import warp as wp
from axion.constraints.control_constraint import control_constraint_residual_kernel
from axion.constraints.dynamics_constraint import unconstrained_dynamics_kernel
from axion.constraints.friction_constraint import friction_residual_kernel
from axion.constraints.positional_contact_constraint import positional_contact_residual_kernel
from axion.constraints.positional_joint_constraint import positional_joint_residual_kernel
from axion.constraints.velocity_contact_constraint import velocity_contact_residual_kernel
from axion.constraints.velocity_joint_constraint import velocity_joint_residual_kernel
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel


def compute_residual(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
    dt: float,
):
    """
    Computes the full system residual vector (R) for the current state.

    This function populates:
      - data.h.d_spatial: The dynamics residual (including constraint forces).
      - data.h.c.*: The constraint violation residuals (Joints, Control, Contacts, Friction).
    """
    device = data.device

    # Zero out all residual buffers before accumulation
    data.h.zero_()

    # -------------------------------------------------------------------------
    # 1. Unconstrained Dynamics
    # -------------------------------------------------------------------------
    # Computes: h_d = M(u - u_prev) - (f_ext + f_g + f_gyro) * dt
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.body_q,
            data.body_u,
            data.body_u_prev,
            data.body_f,
            model.body_mass,
            model.body_inertia,
            dt,
            data.g_accel,
        ],
        outputs=[data.h.d_spatial],
        device=device,
    )

    # -------------------------------------------------------------------------
    # 2. Joint Constraints
    # -------------------------------------------------------------------------
    # Adds -J^T * lambda * dt to h_d AND computes constraint violation h_c.j
    if config.joint_constraint_level == "pos":
        wp.launch(
            kernel=positional_joint_residual_kernel,
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
                dt,
                config.joint_compliance,
            ],
            outputs=[
                data.h.d_spatial,
                data.h.c.j,
            ],
            device=device,
        )
    elif config.joint_constraint_level == "vel":
        wp.launch(
            kernel=velocity_joint_residual_kernel,
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
                dt,
                config.joint_stabilization_factor,
                config.joint_compliance,
            ],
            outputs=[
                # data.constraint_active_mask.j,  <-- REMOVED to match new kernel signature
                data.h.d_spatial,
                data.h.c.j,
            ],
            device=device,
        )

    # -------------------------------------------------------------------------
    # 3. Control Constraints
    # -------------------------------------------------------------------------
    wp.launch(
        kernel=control_constraint_residual_kernel,
        dim=(dims.N_w, dims.joint_count),
        inputs=[
            data.body_q,
            data.body_u,
            data.body_lambda.ctrl,
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
            dt,
        ],
        outputs=[
            data.h.d_spatial,
            data.h.c.ctrl,
        ],
        device=device,
    )

    # -------------------------------------------------------------------------
    # 4. Contact Constraints
    # -------------------------------------------------------------------------
    if config.contact_constraint_level == "pos":
        wp.launch(
            kernel=positional_contact_residual_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.body_q,
                data.body_u,
                data.body_u_prev,
                data.body_lambda.n,
                # Expanded contact_interaction struct:
                data.contact_body_a,
                data.contact_body_b,
                data.contact_point_a,
                data.contact_point_b,
                data.contact_thickness_a,
                data.contact_thickness_b,
                data.contact_dist,
                data.contact_basis_n_a,
                data.contact_basis_n_b,
                # End contact data
                model.body_inv_mass,
                model.body_inv_inertia,
                dt,
                config.contact_compliance,
            ],
            outputs=[
                data.h.d_spatial,
                data.h.c.n,
            ],
            device=device,
        )
    elif config.contact_constraint_level == "vel":
        # Note: velocity_contact_residual_kernel has extra alpha/beta args.
        # We assume defaults (1.0) here if they are not present in EngineConfig.
        fb_alpha = getattr(config, "contact_fb_alpha", 1.0)
        fb_beta = getattr(config, "contact_fb_beta", 1.0)

        wp.launch(
            kernel=velocity_contact_residual_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.body_u,
                data.body_u_prev,
                data.body_lambda.n,
                # Expanded contact_interaction struct:
                data.contact_body_a,
                data.contact_body_b,
                data.contact_dist,
                data.contact_restitution_coeff,
                data.contact_basis_n_a,
                data.contact_basis_n_b,
                # End contact data
                dt,
                config.contact_stabilization_factor,
                fb_alpha,
                fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.h.d_spatial,
                data.h.c.n,
            ],
            device=device,
        )

    # -------------------------------------------------------------------------
    # 5. Friction Constraints
    # -------------------------------------------------------------------------
    wp.launch(
        kernel=friction_residual_kernel,
        dim=(dims.N_w, dims.N_n),
        inputs=[
            data.body_q,
            data.body_u,
            data.body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            # Expanded contact_interaction struct:
            data.contact_body_a,
            data.contact_body_b,
            data.contact_friction_coeff,
            data.contact_basis_t1_a,
            data.contact_basis_t2_a,
            data.contact_basis_t1_b,
            data.contact_basis_t2_b,
            # End contact data
            model.body_inv_mass,
            model.body_inv_inertia,
            dt,
            config.friction_compliance,
        ],
        outputs=[
            data.h.d_spatial,
            data.h.c.f,
        ],
        device=device,
    )
