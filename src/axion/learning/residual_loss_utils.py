"""Shared residual-loss helpers for warm-start style training."""

import torch

from axion.learning.torch_residual_ad import AxionResidualAD


def _safe_mean_square(sum_sq: torch.Tensor, count: int) -> torch.Tensor:
    """Return mean-square for a block, guarded against empty blocks."""
    if count <= 0:
        return torch.zeros_like(sum_sq)
    return sum_sq / float(count)


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
    residual_loss, _, _ = compute_residual_diagnostics(engine, body_vel, constr_force)
    return residual_loss


def compute_residual_diagnostics(
    engine,
    body_vel: torch.Tensor,
    constr_force: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Compute ||residual||^2 and diagnostics by residual block.

    Returns:
        total_sq: squared residual sum over all blocks.
        block_sum_sq: squared residual sums for d/j/ctrl/n/f blocks.
        block_mse: per-block mean-square values to reduce size bias.
    """
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
    dims = engine.dims

    # Residual layout:
    # - d: dynamics block (system momentum balance residuals)
    # - c: all constraint residuals concatenated [j, ctrl, n, f]
    residual_d = residual[:, :dims.N_u]
    residual_c = residual[:, dims.N_u:]

    # Constraint sub-block suffixes:
    # - j: joint constraints
    # - ctrl: control constraints
    # - n: normal contact constraints
    # - f: friction constraints
    residual_j = residual_c[:, dims.slice_j]
    residual_ctrl = residual_c[:, dims.slice_ctrl]
    residual_n = residual_c[:, dims.slice_n]
    residual_f = residual_c[:, dims.slice_f]

    block_sum_sq = {
        "residual_sq_d": torch.sum(residual_d**2),
        "residual_sq_j": torch.sum(residual_j**2),
        "residual_sq_ctrl": torch.sum(residual_ctrl**2),
        "residual_sq_n": torch.sum(residual_n**2),
        "residual_sq_f": torch.sum(residual_f**2),
    }
    total_sq = (
        block_sum_sq["residual_sq_d"]
        + block_sum_sq["residual_sq_j"]
        + block_sum_sq["residual_sq_ctrl"]
        + block_sum_sq["residual_sq_n"]
        + block_sum_sq["residual_sq_f"]
    )

    block_mse = {
        "residual_mse_d": _safe_mean_square(block_sum_sq["residual_sq_d"], dims.N_u),
        "residual_mse_j": _safe_mean_square(block_sum_sq["residual_sq_j"], dims.N_j),
        "residual_mse_ctrl": _safe_mean_square(block_sum_sq["residual_sq_ctrl"], dims.N_ctrl),
        "residual_mse_n": _safe_mean_square(block_sum_sq["residual_sq_n"], dims.N_n),
        "residual_mse_f": _safe_mean_square(block_sum_sq["residual_sq_f"], dims.N_f),
    }

    return total_sq, block_sum_sq, block_mse