# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal environment wrapper for dataset generation only.
Wraps the Warp simulation and exposes the API required by the trajectory sampler.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

import torch
import shutil
import warp as wp

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(base_dir)

from axion.neural_solver.utils import warp_utils
from axion.neural_solver.utils.python_utils import print_ok
from axion.neural_solver.envs.axionEnv import AxionEnv
from axion.neural_solver.neural_model_utils_providers.transformer_neural_utils_provider import (
    TransformerNeuralModelUtilsProvider,
)


class AxionEnvToTrajectorySamplerAdapter:
    """
    Minimal simulation wrapper for trajectory sampling and dataset generation.
    Exposes only: reset, step, step_with_joint_act, states, root_body_q,
    abstract_contacts, model, and the properties used by TrajectorySampler and
    WarpSimDataGenerator. Uses a TransformerNeuralModelUtilsProvider for angle
    wrapping (wrap2PI), state history management, and model input assembly.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        warp_env_cfg=None,
        utils_provider_cfg=None,
        neural_model=None,
        default_env_mode: str = "ground-truth",
        device: str = "cuda:0",
        render: bool = False,
        custom_articulation_builder=None,
        **kwargs,
    ):
        # Handle dict-like arguments similar to the original NeuralEnvironment.
        if utils_provider_cfg is None:
            utils_provider_cfg = {}
        if warp_env_cfg is None:
            warp_env_cfg = {}
        if custom_articulation_builder is not None:
            warp_env_cfg = {**warp_env_cfg, "custom_articulation_builder": custom_articulation_builder}

        # Use AxionEnv as backend, preserving (most of) the public API contract.
        self.env = AxionEnv(
            env_name = env_name,
            num_worlds= num_envs,
            device = device,
            requires_grad= False # Check if true
        )

        self.utils_provider = TransformerNeuralModelUtilsProvider(
            model=self.env.model,
            neural_model=neural_model,
            cfg=utils_provider_cfg,
            num_states_history=utils_provider_cfg.get("num_states_history", 1),
            device=device,
        )

        # Default mode for step/reset API; we currently always step the ground-truth AxionEnv.
        assert default_env_mode in ("ground-truth", "neural")
        self.env_mode = default_env_mode

        # State buffers 
        self.states = torch.zeros(
            (self.num_envs, self.state_dim),
            device=self.torch_device,
        )
        self.joint_acts = torch.zeros(
            (self.num_envs, self.joint_act_dim),
            device=self.torch_device,
        )
        self.root_body_q = wp.to_torch(self.sim_states.body_q)[
            0 :: self.bodies_per_env, :
        ].view(self.num_envs, 7).to(self.torch_device)

        # Video export (used by trajectory sampler when export_video=True)
        self.export_video = False
        self.video_export_filename = None
        self.video_tmp_folder = None
        self.video_frame_cnt = 0

    # ---- Properties used by trajectory sampler and simulation sampler ----

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def dof_q_per_env(self):
        return self.env.dof_q_per_world

    @property
    def dof_qd_per_env(self):
        return self.env.dof_qd_per_world

    @property
    def state_dim(self):
        return self.env.dof_q_per_world + self.env.dof_qd_per_world

    @property
    def bodies_per_env(self):
        return self.env.bodies_per_world

    @property
    def joint_act_dim(self):
        return self.env.joint_act_dim

    @property
    def action_dim(self):
        return self.env.control_dim

    @property
    def action_limits(self):
        return self.env.control_limits

    @property
    def joint_types(self):
        return self.env.joint_types

    @property
    def device(self):
        return self.env.device

    @property
    def torch_device(self):
        return wp.device_to_torch(self.env.device)

    @property
    def robot_name(self):
        return self.env.robot_name

    @property
    def abstract_contacts(self):
        return self.env.abstract_contacts

    @property
    def sim_states(self):
        return self.env.state

    @property
    def model(self):
        return self.env.model

    @property
    def eval_collisions(self):
        return self.env.eval_collisions

    @property
    def num_contacts_per_env(self):
        return self.env.abstract_contacts.num_contacts_per_env

    @property
    def frame_dt(self):
        return self.env.frame_dt

    # ---- Mode and collisions (trajectory sampler sets these) ----

    def set_env_mode(self, env_mode: str):
        assert env_mode in ("ground-truth", "neural")
        self.env_mode = env_mode
        # For now, AxionEnv always runs ground-truth dynamics; env_mode is kept
        # for API compatibility with trainers/evaluators.

    def set_eval_collisions(self, eval_collisions: bool):
        self.env.set_eval_collisions(eval_collisions)

    def wrap2PI(self, states):
        self.utils_provider.wrap2PI(states)

    # ---- State sync  ----

    def _update_states(self, states: Optional[torch.Tensor] = None):
        if states is None:
            if not getattr(self.env, "uses_generalized_coordinates", True):
                warp_utils.eval_ik(self.env.model, self.env.state)
            warp_utils.acquire_states_to_torch(self.env, self.states)
        else:
            self.states.copy_(states)

        # Update cached root pose and keep utils_provider buffers in sync.
        self.root_body_q.copy_(
            wp.to_torch(self.sim_states.body_q)[0 :: self.bodies_per_env, :].view(
                self.num_envs, 7
            )
        )

        self.utils_provider.states.copy_(self.states)
        self.utils_provider.root_body_q.copy_(self.root_body_q)
        # Gravity dir is static and initialized in the utils_provider.
        self.utils_provider.wrap2PI(self.utils_provider.states)
        self.utils_provider.append_current_state_to_history(joint_acts=self.joint_acts)

        if states is not None:
            warp_utils.assign_states_from_torch(self.env, self.states)
            warp_utils.eval_fk(self.env.model, self.env.state)

    # ---- Step and reset (trajectory sampler uses these) ----

    def step(
        self,
        actions: torch.Tensor,
        env_mode: Optional[str] = None,
    ) -> torch.Tensor:
        if env_mode is None:
            env_mode = self.env_mode
        self.set_env_mode(env_mode)

        if self.action_dim > 0:
            self.env.assign_control(
                wp.from_torch(actions),
                self.env.control,
                self.env.state,
            )
            self.joint_acts.copy_(
                wp.to_torch(self.env.control.joint_act).view(
                    self.num_envs,
                    self.joint_act_dim,
                )
            )

        self.env.update()
        self._update_states()
        return self.states

    def step_with_joint_act(
        self,
        joint_acts: torch.Tensor,
        env_mode: Optional[str] = None,
    ) -> torch.Tensor:
        if env_mode is None:
            env_mode = self.env_mode
        self.set_env_mode(env_mode)

        if self.joint_act_dim > 0:
            self.env.joint_act.assign(wp.array(joint_acts.view(-1)))
            self.joint_acts.copy_(
                wp.to_torch(self.env.control.joint_act).view(
                    self.num_envs,
                    self.joint_act_dim,
                )
            )

        self.env.update()
        self._update_states()
        return self.states

    def reset(self, initial_states: Optional[torch.Tensor] = None):
        if initial_states is not None:
            assert initial_states.shape[0] == self.num_envs
            assert initial_states.device == self.torch_device or str(
                initial_states.device
            ) == str(self.torch_device)
            self._update_states(initial_states)
        else:
            self.env.reset()
            self._update_states()
        self.utils_provider.reset()

    def close(self):
        self.env.close()
