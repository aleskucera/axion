# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Numeric duplicate of TARGET_POSITION position error in
# axion.constraints.control_constraint.compute_control_properties (lines 131–145 only):
# raw_error = q - target, then shortest-path wrap to [-pi, pi] when is_angular.
# Does not apply / dt or gains (those follow in the Warp code).

from __future__ import annotations

import torch

_PI = 3.141592653589793
_TWO_PI = 6.283185307179586


def wrap_revolute_position_error_torch(raw_error: torch.Tensor) -> torch.Tensor:
    raw_error_shifted = raw_error + _PI
    mod_error = raw_error_shifted - _TWO_PI * torch.floor(raw_error_shifted / _TWO_PI)
    return mod_error - _PI


def joint_position_target_errors_torch(
    q: torch.Tensor,
    target: torch.Tensor,
    *,
    is_angular: bool | torch.Tensor = True,
) -> torch.Tensor:
    """Batch shape (*, nq) for q and target; same broadcasting rules as q - target."""
    raw_error = q - target
    if isinstance(is_angular, bool):
        if is_angular:
            return wrap_revolute_position_error_torch(raw_error)
        return raw_error
    wrapped = wrap_revolute_position_error_torch(raw_error)
    return torch.where(is_angular, wrapped, raw_error)
