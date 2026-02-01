from __future__ import annotations

import warp as wp

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions


@wp.kernel
def copy_h_to_history(
    h: wp.array(dtype=wp.float32, ndim=2),
    h_history: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, h_idx = wp.tid()
    h_history[world_idx, h_idx] = h[world_idx, h_idx]


@wp.kernel
def copy_body_q_to_history(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_q_history: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()
    body_q_history[world_idx, body_idx] = body_q[world_idx, body_idx]


@wp.kernel
def copy_body_u_to_history(
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_history: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    body_u_history[world_idx, body_idx] = body_u[world_idx, body_idx]


@wp.kernel
def copy_body_lambda_to_history(
    body_lambda: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_history: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()
    body_lambda_history[world_idx, constraint_idx] = body_lambda[world_idx, constraint_idx]


def copy_state_to_history(
    newton_iteration: int,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=copy_h_to_history,
        dim=(data.dims.N_w, data.dims.N_u + data.dims.N_c),
        inputs=[data._h],
        outputs=[data.newton_history._h_history[newton_iteration, :, :]],
        device=data.device,
    )

    wp.launch(
        kernel=copy_body_q_to_history,
        dim=(data.dims.N_w, data.dims.N_b),
        inputs=[data.body_q],
        outputs=[data.newton_history.body_q_history[newton_iteration, :, :]],
        device=data.device,
    )

    wp.launch(
        kernel=copy_body_u_to_history,
        dim=(data.dims.N_w, data.dims.N_b),
        inputs=[data.body_u],
        outputs=[data.newton_history.body_u_history[newton_iteration, :, :]],
        device=data.device,
    )

    wp.launch(
        kernel=copy_body_lambda_to_history,
        dim=(data.dims.N_w, data.dims.N_c),
        inputs=[data._body_lambda],
        outputs=[data.newton_history._body_lambda_history[newton_iteration, :, :]],
        device=data.device,
    )


def copy_state_to_trajectory(
    step_idx: int,
    data: EngineData,
):
    if data.trajectory is None:
        return

    # Ensure step_idx is within bounds
    # (Caller should handle logic, but safe to check or let it error)
    
    wp.launch(
        kernel=copy_body_q_to_history,  # Reuse kernel (same signature)
        dim=(data.dims.N_w, data.dims.N_b),
        inputs=[data.body_q],
        outputs=[data.trajectory.body_q_traj[step_idx]],
        device=data.device,
    )

    wp.launch(
        kernel=copy_body_u_to_history,  # Reuse kernel (same signature)
        dim=(data.dims.N_w, data.dims.N_b),
        inputs=[data.body_u],
        outputs=[data.trajectory.body_u_traj[step_idx]],
        device=data.device,
    )