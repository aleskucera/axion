from typing import Any

import warp as wp
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_positional_contact_residual_kernel
from axion.constraints import batch_positional_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import batch_velocity_contact_residual_kernel
from axion.constraints import batch_velocity_joint_residual_kernel
from axion.constraints import fused_batch_friction_residual_kernel
from axion.constraints import fused_batch_positional_contact_residual_kernel
from axion.constraints import fused_batch_positional_joint_residual_kernel
from axion.constraints import fused_batch_unconstrained_dynamics_kernel
from axion.constraints import fused_batch_velocity_contact_residual_kernel
from axion.constraints import fused_batch_velocity_joint_residual_kernel
from axion.constraints.control_constraint import batch_control_constraint_residual_kernel
from axion.constraints.control_constraint import fused_batch_control_constraint_residual_kernel
from axion.math import integrate_batched_body_pose_kernel
from axion.math import integrate_body_pose_kernel

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel
from .residual_utils import compute_residual


@wp.kernel
def linesearch_spread_kernel(
    x: wp.array(dtype=Any, ndim=2),
    dx: wp.array(dtype=Any, ndim=2),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_x: wp.array(dtype=Any, ndim=3),
):
    batch_idx, world_idx, x_idx = wp.tid()

    spread_x = x[world_idx, x_idx] + linesearch_steps[batch_idx] * dx[world_idx, x_idx]
    linesearch_x[batch_idx, world_idx, x_idx] = spread_x


@wp.kernel
def copy_best_sample_kernel(
    linesearch_x: wp.array(dtype=Any, ndim=3),
    best_idx: wp.array(dtype=wp.int32),
    # Outputs
    x: wp.array(dtype=Any, ndim=2),
):
    world_idx, x_idx = wp.tid()
    x[world_idx, x_idx] = linesearch_x[best_idx[world_idx], world_idx, x_idx]


@wp.kernel
def update_without_linesearch(
    dx: wp.array(dtype=Any, ndim=2),
    # Outputs
    x: wp.array(dtype=Any, ndim=2),
):
    world_idx, x_idx = wp.tid()

    x[world_idx, x_idx] = x[world_idx, x_idx] + dx[world_idx, x_idx]


@wp.kernel
def find_minimal_residual_index_kernel(
    batch_h_norm_sq: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    minimal_index: wp.array(dtype=wp.int32),
):
    world_idx = wp.tid()

    min_idx = wp.int32(0)
    min_value = batch_h_norm_sq[0, world_idx]

    # Iterate over batch dimension (dim 0)
    for i in range(1, batch_h_norm_sq.shape[0]):
        value = batch_h_norm_sq[i, world_idx]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value
    minimal_index[world_idx] = min_idx


def compute_linesearch_batch_variables(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    # 1. Compute candidates for body_velocities
    wp.launch(
        kernel=linesearch_spread_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.body_count,
        ),
        inputs=[
            data.body_vel,
            data.dbody_vel,
            data.linesearch_step_size,
        ],
        outputs=[data.linesearch_body_vel],
        device=data.device,
    )

    # 2. Compute candidates for body poses (integration)
    wp.launch(
        kernel=integrate_batched_body_pose_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.body_count,
        ),
        inputs=[
            data.linesearch_body_vel,
            data.body_pose_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.linesearch_body_pose],
        device=data.device,
    )

    # 3. Compute candidates for constraint forces
    wp.launch(
        kernel=linesearch_spread_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.num_constraints,
        ),
        inputs=[
            data.constr_force.full,
            data.dconstr_force.full,
            data.linesearch_step_size,
        ],
        outputs=[data.linesearch_constr_force.full],
        device=data.device,
    )


def compute_linesearch_batch_h(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    # Evaluate residual for unconstrained dynamics
    wp.launch(
        kernel=fused_batch_unconstrained_dynamics_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.body_vel_prev,
            data.ext_force,
            model.body_mass,
            model.body_inertia,
            data.dt,
            model.g_accel,
            dims.linesearch_step_count,
        ],
        outputs=[data.linesearch_res.d_spatial],
        device=data.device,
    )

    wp.launch(
        kernel=fused_batch_positional_joint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_constr_force.j,
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
            data.dt,
            config.joint_compliance,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.j,
        ],
        device=data.device,
    )

    # Evaluate residual for control constraints
    wp.launch(
        kernel=fused_batch_control_constraint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.linesearch_constr_force.ctrl,
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
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.ctrl,
        ],
        device=data.device,
    )

    # Evaluate residual for normal contact constraints
    wp.launch(
        kernel=fused_batch_positional_contact_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.body_vel_prev,
            data.linesearch_constr_force.n,
            data.contact_body_a,
            data.contact_body_b,
            data.contact_point_a,
            data.contact_point_b,
            data.contact_thickness_a,
            data.contact_thickness_b,
            data.contact_dist,
            data.contact_basis_n_a,
            data.contact_basis_n_b,
            data.constr_active_mask.n,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.dt,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.n,
        ],
        device=data.device,
    )

    # Evaluate residual for friction constraints
    wp.launch(
        kernel=fused_batch_friction_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.linesearch_constr_force.f,
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
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.f,
        ],
        device=data.device,
    )

    # FIX: Find out why is this bad?
    data.linesearch_res.sync_to_float()


def select_minimal_residual_variables(
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    # Compute norm squared for each batch using TiledSqNorm
    data.linesearch_tiled_res_sq_norm.compute(data.linesearch_res.full, data.linesearch_res_norm_sq)

    # Find the index with minimal residual norm (per world)
    wp.launch(
        kernel=find_minimal_residual_index_kernel,
        dim=dims.num_worlds,
        inputs=[data.linesearch_res_norm_sq],
        outputs=[data.linesearch_minimal_index],
        device=data.device,
    )

    # Copy the minimal residual state variables back
    wp.launch(
        kernel=copy_best_sample_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_vel,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_vel],
        device=data.device,
    )

    wp.launch(
        kernel=copy_best_sample_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_pose],
        device=data.device,
    )

    wp.launch(
        kernel=copy_best_sample_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data.linesearch_constr_force.full,
            data.linesearch_minimal_index,
        ],
        outputs=[data.constr_force.full],
        device=data.device,
    )

    wp.launch(
        kernel=copy_best_sample_kernel,
        dim=(dims.num_worlds, dims.N_u + dims.num_constraints),
        inputs=[
            data.linesearch_res.full,
            data.linesearch_minimal_index,
        ],
        outputs=[data.res.full],
        device=data.device,
    )


def update_variables_without_linesearch(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=update_without_linesearch,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[data.dbody_vel],
        outputs=[data.body_vel],
        device=data.device,
    )

    wp.launch(
        kernel=integrate_body_pose_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_vel,
            data.body_pose_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.body_pose],
        device=data.device,
    )

    wp.launch(
        kernel=update_without_linesearch,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[data.dconstr_force.full],
        outputs=[data.constr_force.full],
        device=data.device,
    )


@wp.kernel
def simple_sq_norm_kernel(
    val: wp.array(dtype=wp.float32, ndim=2), out_norm: wp.array(dtype=wp.float32, ndim=1)
):
    world_idx, i = wp.tid()
    # Simple atomic add - perfectly safe, no alignment requirements
    val_sq = val[world_idx, i] * val[world_idx, i]
    wp.atomic_add(out_norm, world_idx, val_sq)


def perform_linesearch(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    if config.enable_linesearch:
        compute_linesearch_batch_variables(model, data, config, dims)
        compute_linesearch_batch_h(model, data, config, dims)
        select_minimal_residual_variables(data, config, dims)
    else:
        update_variables_without_linesearch(model, data, config, dims)
    compute_residual(model, data, config, dims)

    # FIX: This causes following error
    # Warp CUDA error 700: an illegal memory access was encountered
    # (in function wp_free_device_async, /builds/omniverse/warp/warp/native/warp.cu:812)
    data.tiled_sq_norm.compute(data.res.full, data.res_norm_sq)
    # data.res_norm_sq.zero_()
    # wp.launch(
    #     kernel=simple_sq_norm_kernel,
    #     dim=(dims.num_worlds, dims.num_constraints),
    #     inputs=[
    #         data._res,
    #     ],
    #     outputs=[
    #         data.res_norm_sq,
    #     ],
    #     device=data.device,
    # )
