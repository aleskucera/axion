"""Shared residual-loss helpers for warm-start style training."""

import torch

from axion.learning.torch_residual_ad import AxionResidualAD

RESIDUAL_BLOCK_NAMES = ("d", "j", "ctrl", "n", "f")


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


def update_residual_block_running_var(
    running_var: dict[str, torch.Tensor],
    block_mse: dict[str, torch.Tensor],
    momentum: float,
    exclude_blocks: set[str] | None = None,
) -> None:
    """EMA update for residual block running variances."""
    if exclude_blocks is None:
        exclude_blocks = set()

    for block in RESIDUAL_BLOCK_NAMES:
        if block in exclude_blocks:
            continue
        key = f"residual_mse_{block}"
        if key not in block_mse or block not in running_var:
            continue
        value = block_mse[key].detach().to(dtype=torch.float32)
        value_sq = torch.square(value)
        running_var[block].mul_(momentum).add_((1.0 - momentum) * value_sq)


def compute_weighted_residual_loss_from_blocks(
    block_sum_sq: dict[str, torch.Tensor],
    running_var: dict[str, torch.Tensor],
    eps: float,
    exclude_blocks: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted residual sum and per-block weights using 1/sqrt(var + eps)."""
    if exclude_blocks is None:
        exclude_blocks = set()

    total = torch.zeros_like(next(iter(block_sum_sq.values())))
    weights: dict[str, torch.Tensor] = {}

    for block in RESIDUAL_BLOCK_NAMES:
        sq_key = f"residual_sq_{block}"
        if sq_key not in block_sum_sq:
            continue
        block_sq = block_sum_sq[sq_key]
        if block in exclude_blocks or block not in running_var:
            weight = torch.ones_like(block_sq)
        else:
            var = running_var[block].to(device=block_sq.device, dtype=block_sq.dtype)
            weight = 1.0 / torch.sqrt(var + eps)
        weights[f"residual_weight_{block}"] = weight
        total = total + weight * block_sq

    return total, weights
