# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Pendulum multipliers layout shared by dataset generation, neural logging, and plots."""

from __future__ import annotations

import numpy as np
import torch

# Fixed layout for Pendulum + tilted contact plane (joint_dof_mode NONE vs TARGET_POSITION
# only changes control block width). See EngineDimensions: joint | control | contact.
PENDULUM_FULL_LAMBDA_DIM = 24
PENDULUM_LAMBDA_N_J = 10
PENDULUM_LAMBDA_N_CTRL = 2


def _fill_pendulum_canonical_from_engine_torch(dest: torch.Tensor, src: torch.Tensor) -> None:
    """Write src into dest[..., :24] using canonical joint|control|contact ordering; zeros dest first."""
    dest.zero_()
    ld = int(src.shape[-1])
    nj, nc = PENDULUM_LAMBDA_N_J, PENDULUM_LAMBDA_N_CTRL
    full = PENDULUM_FULL_LAMBDA_DIM
    passive_with_contact = full - nc
    active_without_contact = nj + nc

    if ld == full:
        dest.copy_(src)
        return
    if ld == passive_with_contact:
        # joint | contact (control missing)
        dest[..., :nj].copy_(src[..., :nj])
        dest[..., nj + nc :].copy_(src[..., nj:])
        return
    if ld == active_without_contact:
        # joint | control (contact missing)
        dest[..., :nj].copy_(src[..., :nj])
        dest[..., nj : nj + nc].copy_(src[..., nj:active_without_contact])
        return
    if ld == nj:
        # joint only (control + contact missing)
        dest[..., :nj].copy_(src[..., :nj])
        return

    raise ValueError(
        f"Unexpected Pendulum engine λ width {ld}; supported widths are {full}, "
        f"{passive_with_contact}, {active_without_contact}, or {nj}."
    )


def _fill_pendulum_canonical_from_engine_numpy(dest: np.ndarray, src: np.ndarray) -> None:
    dest.fill(0)
    ld = int(src.shape[-1])
    nj, nc = PENDULUM_LAMBDA_N_J, PENDULUM_LAMBDA_N_CTRL
    full = PENDULUM_FULL_LAMBDA_DIM
    passive_with_contact = full - nc
    active_without_contact = nj + nc

    if ld == full:
        dest[...] = src
        return
    if ld == passive_with_contact:
        dest[..., :nj] = src[..., :nj]
        dest[..., nj + nc :] = src[..., nj:]
        return
    if ld == active_without_contact:
        dest[..., :nj] = src[..., :nj]
        dest[..., nj : nj + nc] = src[..., nj:active_without_contact]
        return
    if ld == nj:
        dest[..., :nj] = src[..., :nj]
        return

    raise ValueError(
        f"Unexpected Pendulum engine λ width {ld}; supported widths are {full}, "
        f"{passive_with_contact}, {active_without_contact}, or {nj}."
    )


def expand_pendulum_engine_lambdas_torch(src: torch.Tensor, target_last_dim: int) -> torch.Tensor:
    """Map raw engine λ to canonical width ``target_last_dim`` (typically 24)."""
    ld = int(src.shape[-1])
    if ld == target_last_dim:
        return src
    if ld > target_last_dim:
        raise ValueError(
            "Cannot narrow Pendulum constraint λ for logging: "
            f"engine width {ld} > target {target_last_dim}."
        )
    if target_last_dim != PENDULUM_FULL_LAMBDA_DIM:
        raise ValueError(
            f"Expanding Pendulum λ from width {ld} is only supported for target "
            f"w={PENDULUM_FULL_LAMBDA_DIM}; got {target_last_dim}."
        )
    out = torch.zeros(
        *src.shape[:-1],
        PENDULUM_FULL_LAMBDA_DIM,
        dtype=src.dtype,
        device=src.device,
    )
    _fill_pendulum_canonical_from_engine_torch(out, src)
    return out


def expand_pendulum_engine_lambdas_numpy(src: np.ndarray, target_last_dim: int) -> np.ndarray:
    """NumPy counterpart of ``expand_pendulum_engine_lambdas_torch``."""
    ld = int(src.shape[-1])
    if ld == target_last_dim:
        return src
    if ld > target_last_dim:
        raise ValueError(
            "Cannot narrow Pendulum constraint λ for logging: "
            f"engine width {ld} > target {target_last_dim}."
        )
    if target_last_dim != PENDULUM_FULL_LAMBDA_DIM:
        raise ValueError(
            f"Expanding Pendulum λ from width {ld} is only supported for target "
            f"w={PENDULUM_FULL_LAMBDA_DIM}; got {target_last_dim}."
        )
    out = np.zeros((*src.shape[:-1], PENDULUM_FULL_LAMBDA_DIM), dtype=src.dtype)
    _fill_pendulum_canonical_from_engine_numpy(out, src)
    return out
