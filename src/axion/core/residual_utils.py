import warp as wp
from axion.constraints.control_constraint import control_constraint_residual_kernel
from axion.constraints.dynamics_constraint import unconstrained_dynamics_kernel
from axion.constraints.friction_constraint import friction_residual_kernel
from axion.constraints.positional_contact_constraint import positional_contact_residual_kernel
from axion.constraints.positional_joint_constraint import positional_joint_residual_kernel
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel


def compute_residual(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    """
    Computes the full system residual vector (R) for the current state.

    This function populates:
      - data.h.d_spatial: The dynamics residual (including constraint forces).
      - data.h.c.*: The constraint violation residuals (Joints, Control, Contacts, Friction).
    """
    # Zero out all residual buffers before accumulation
    # data.res.zero_()

    # -------------------------------------------------------------------------
    # 1. Unconstrained Dynamics
    # -------------------------------------------------------------------------
    # Computes: h_d = M(u - u_prev) - (f_ext + f_g + f_gyro) * dt
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.ext_force,
            model.body_mass,
            model.body_inertia,
            data.dt,
            model.g_accel,
        ],
        outputs=[data.res.d_spatial],
        device=data.device,
    )

    # -------------------------------------------------------------------------
    # 2. Joint Constraints
    # -------------------------------------------------------------------------
    # Adds -J^T * lambda * dt to h_d AND computes constraint violation h_c.j
    wp.launch(
        kernel=positional_joint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.constr_force.j,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_constraint_offsets,
            model.joint_compliance,
            data.dt,
            config.joint_compliance,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.j,
        ],
        device=data.device,
    )

    # -------------------------------------------------------------------------
    # 3. Control Constraints
    # -------------------------------------------------------------------------
    wp.launch(
        kernel=control_constraint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.constr_force.ctrl,
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
            model.control_constraint_offsets,
            data.joint_target_pos,
            data.joint_target_vel,
            model.joint_target_ke,
            model.joint_target_kd,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.ctrl,
        ],
        device=data.device,
    )

    # -------------------------------------------------------------------------
    # 4. Contact Constraints
    # -------------------------------------------------------------------------
    wp.launch(
        kernel=positional_contact_residual_kernel,
        dim=(dims.world_count, dims.contact_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.constr_force.n,
            data.contact_body_a,
            data.contact_body_b,
            data.contact_point_a,
            data.contact_point_b,
            data.contact_thickness_a,
            data.contact_thickness_b,
            data.contact_dist,
            data.contact_basis_n_a,
            data.contact_basis_n_b,
            data.constraint_active_mask.n,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.n,
        ],
        device=data.device,
    )

    # -------------------------------------------------------------------------
    # 5. Friction Constraints
    # -------------------------------------------------------------------------
    wp.launch(
        kernel=friction_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.constr_force.f,
            data.constr_force_prev_iter.f,
            data.constr_force_prev_iter.n,
            data.contact_body_a,
            data.contact_body_b,
            data.contact_friction_coeff,
            data.contact_basis_t1_a,
            data.contact_basis_t2_a,
            data.contact_basis_t1_b,
            data.contact_basis_t2_b,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.dt,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.f,
        ],
        device=data.device,
    )

    data.h.sync_to_float()
