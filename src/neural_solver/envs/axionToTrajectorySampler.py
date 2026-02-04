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
No neural integrator or RL-related logic.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

import warp as wp
import torch
import cv2
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(base_dir)

from src.neural_solver.utils import warp_utils
#from utils.env_utils import create_abstract_contact_env
from src.neural_solver.utils.python_utils import print_ok


class AxionEnvToTrajectorySamplerAdapter:
    """
    Minimal simulation wrapper for trajectory sampling and dataset generation.
    Exposes only: reset, step, step_with_joint_act, states, root_body_q,
    abstract_contacts, model, and the properties used by TrajectorySampler and
    WarpSimDataGenerator. Uses a dummy NeuralIntegrator only for angle wrapping
    (wrap2PI) and reset() no-op.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        warp_env_cfg=None,
        device: str = "cuda:0",
        render: bool = False,
        custom_articulation_builder=None,
    ):
        if warp_env_cfg is None:
            warp_env_cfg = {}
        if custom_articulation_builder is not None:
            warp_env_cfg = {**warp_env_cfg, "custom_articulation_builder": custom_articulation_builder}

        self.env = create_abstract_contact_env(
            env_name=env_name,
            num_envs=num_envs,
            requires_grad=False,
            device=device,
            render=render,
            **warp_env_cfg,
        )

        # Dummy neural integrator only for wrap2PI and reset(); no neural model.
        from integrators.integrator_neural import NeuralIntegrator

        self._integrator_dummy = NeuralIntegrator(
            model=self.env.model,
            neural_model=None,
        )

        self.env_mode = "ground-truth"

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
        ].view(self.num_envs, 7)

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
        return self.env.dof_q_per_env

    @property
    def dof_qd_per_env(self):
        return self.env.dof_qd_per_env

    @property
    def state_dim(self):
        return self.env.dof_q_per_env + self.env.dof_qd_per_env

    @property
    def bodies_per_env(self):
        return self.env.bodies_per_env

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
        # We only ever run ground-truth; no integrator swap.

    def set_eval_collisions(self, eval_collisions: bool):
        self.env.set_eval_collisions(eval_collisions)

    # ---- State sync  ----

    def _update_states(self, states: Optional[torch.Tensor] = None):
        if states is None:
            if not getattr(self.env, "uses_generalized_coordinates", True):
                warp_utils.eval_ik(self.env.model, self.env.state)
            warp_utils.acquire_states_to_torch(self.env, self.states)
        else:
            self.states.copy_(states)

        self._integrator_dummy.wrap2PI(self.states)

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
        self._integrator_dummy.reset()

    # ---- Video export (trajectory sampler can use these) ----

    def start_video_export(self, video_export_filename: str):
        self.export_video = True
        self.video_export_filename = os.path.join("gifs", video_export_filename)
        self.video_tmp_folder = os.path.join(
            Path(video_export_filename).parent,
            "tmp",
        )
        os.makedirs(self.video_tmp_folder, exist_ok=False)
        self.video_frame_cnt = 0

    def end_video_export(self):
        self.export_video = False
        frame_rate = round(1.0 / self.env.frame_dt)
        images_path = os.path.join(self.video_tmp_folder, r"%d.png")

        if not os.path.exists(os.path.dirname(self.video_export_filename)):
            os.makedirs(os.path.dirname(self.video_export_filename), exist_ok=False)

        os.system("ffmpeg -i {} -vf palettegen palette.png".format(images_path))
        os.system(
            "ffmpeg -framerate {} -i {} "
            "-i palette.png -lavfi paletteuse {}".format(
                frame_rate,
                images_path,
                self.video_export_filename,
            )
        )

        os.remove("palette.png")
        shutil.rmtree(self.video_tmp_folder)
        print_ok("Export video to {}".format(self.video_export_filename))

        self.video_export_filename = None
        self.video_tmp_folder = None
        self.video_frame_cnt = 0

    def render(self):
        self.env.render()
        if self.export_video:
            img = wp.zeros(
                (
                    self.env.renderer.screen_height,
                    self.env.renderer.screen_width,
                    3,
                ),
                dtype=wp.uint8,
            )
            self.env.renderer.get_pixels(
                img,
                split_up_tiles=False,
                mode="rgb",
                use_uint8=True,
            )
            cv2.imwrite(
                os.path.join(
                    self.video_tmp_folder,
                    "{}.png".format(self.video_frame_cnt),
                ),
                img.numpy()[:, :, ::-1],
            )
            self.video_frame_cnt += 1
        time.sleep(self.env.frame_dt)

    def close(self):
        self.env.close()
