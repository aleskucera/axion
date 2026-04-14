"""Shared residual-loss helpers for warm-start style training."""

import torch

from axion.learning.torch_residual_ad import AxionResidualAD


def validate_warm_start_shapes(dims, body_vel: torch.Tensor, constr_force: torch.Tensor) -> None:
    """Validate residual-loss input tensors against engine dimensions."""
    if body_vel.ndim != 2:
        raise RuntimeError(
            f"body_vel must be rank-2 (num_worlds, N_u), got shape={tuple(body_vel.shape)}."
        )
    if constr_force.ndim != 2:
        raise RuntimeError(
            "constr_force must be rank-2 (num_worlds, num_constraints), "
            f"got shape={tuple(constr_force.shape)}."
        )
    if body_vel.shape[0] != dims.num_worlds:
        raise RuntimeError(
            "body_vel world dimension mismatch: "
            f"{body_vel.shape[0]} vs dims.num_worlds={dims.num_worlds}."
        )
    if constr_force.shape[0] != dims.num_worlds:
        raise RuntimeError(
            "constr_force world dimension mismatch: "
            f"{constr_force.shape[0]} vs dims.num_worlds={dims.num_worlds}."
        )
    if body_vel.shape[1] != dims.N_u:
        raise RuntimeError(
            f"body_vel feature mismatch: {body_vel.shape[1]} vs dims.N_u={dims.N_u}."
        )
    if constr_force.shape[1] != dims.num_constraints:
        raise RuntimeError(
            "constr_force feature mismatch: "
            f"{constr_force.shape[1]} vs dims.num_constraints={dims.num_constraints}."
        )


def compute_residual_loss(engine, body_vel: torch.Tensor, constr_force: torch.Tensor) -> torch.Tensor:
    """Compute ||residual||^2 with exact gradients via Warp autodiff."""
    validate_warm_start_shapes(engine.dims, body_vel, constr_force)
    residual = AxionResidualAD.apply(
        engine.axion_model,
        engine.axion_contacts,
        engine.data,
        engine.config,
        engine.dims,
        body_vel,
        constr_force,
    )
    return torch.sum(residual**2)
