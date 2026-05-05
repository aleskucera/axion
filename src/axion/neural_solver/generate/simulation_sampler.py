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

import math
from abc import abstractmethod
from typing import Union
import warp as wp

import numpy as np

import torch

from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from examples.double_pendulum.pendulum_articulation_definition import LINK_LENGTH

'''
Compute the contact point 1 from contact point 0, contact normal and contact depth.
Computed contact point 1 is in world frame.
'''

@wp.kernel(enable_backward=False)
def compute_contact_points_0_world(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    # outputs
    contact_point0_world: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    body = shape_body[shape]
    contact_point0_world[contact_id] = wp.transform_point(
        body_q[body], 
        contact_point0[contact_id]
    )

# [NOTE]: Assume contact point 1 is in world frame
@wp.kernel(enable_backward=False)
def compute_contact_points_1(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    # outputs
    contact_point1: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    body = shape_body[shape]
    point0_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    contact_point1[contact_id] = point0_world \
        - contact_depth[contact_id] * contact_normal[contact_id]
    
class Sampler:
    """Abstract class of data sampler."""
    @abstractmethod
    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        pass

    def sample_plane_normals_near_z(
        self,
        batch_size: int,
        max_angle_rad: float,
        data: torch.Tensor,
    ) -> None:
        """Sample unit vectors (plane normals) with angle to world z-axis (0,0,1) at most max_angle_rad.
        Fills data of shape (batch_size, 3) with unit vectors, in-place."""
        raise NotImplementedError

    def sample_plane_d_coefficients(
        self,
        batch_size: int,
        max_z_coord: float,
        data: torch.Tensor,
    ) -> None:
        """Sample plane d coefficient for n·x + d = 0. Fills data of shape (batch_size, 1) in-place."""
        raise NotImplementedError

class SobolSampler(Sampler):
    """Systematic sampling using Sobol sequences."""

    def __init__(
        self,
        seed=None,
        scramble=False,
    ):
        self.scramble = scramble
        self.seed = seed

    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        soboleng = torch.quasirandom.SobolEngine(
            data.shape[1], scramble=self.scramble, seed=self.seed
        )
        soboleng.draw(batch_size, dtype=torch.float32, out=data)
        data[...] = data * (high - low) + low

    def sample_plane_normals(
        self,
        batch_size: int,
        max_angle_rad: float,
        data: torch.Tensor,
    ) -> None:
        raise NotImplementedError
        # """Sample unit vectors on the spherical cap using 2D Sobol (n_z, phi)."""

    def sample_plane_d_coefficients(
        self,
        batch_size: int,
        max_z_coord: float,
        data: torch.Tensor,
    ) -> None:
        raise NotImplementedError
        # """Sample plane d coefficient for n·x + d = 0."""
        # assert data.shape[0] == batch_size and data.shape[1] == 3
        # cos_max = math.cos(max_angle_rad)
        # soboleng = torch.quasirandom.SobolEngine(2, scramble=self.scramble, seed=self.seed)
        # u = soboleng.draw(batch_size, dtype=torch.float32, device=data.device)
        # n_z = cos_max + (1.0 - cos_max) * u[:, 0]
        # phi = 2 * math.pi * u[:, 1]
        # r_xy = torch.sqrt((1 - n_z.square()).clamp(min=0))
        # data[:, 0] = r_xy * torch.cos(phi)
        # data[:, 1] = r_xy * torch.sin(phi)
        # data[:, 2] = n_z


class UniformSampler(Sampler):
    """Random sampling using uniform distribution."""
    def __init__(self):
        pass

    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        assert data.shape[0] == batch_size
        data.uniform_()
        data[...] = data * (high - low) + low

    def sample_plane_normals(
        self,
        batch_size: int,
        data: torch.Tensor,
    ) -> None:
        """
        Sample plane normals as (x,y,z) vector with limits on their coords.
        """
        assert data.shape[0] == batch_size and data.shape[1] == 3
        data.uniform_()
        min_a, max_a = -3 ,3
        min_b, max_b = 0, 0
        min_c, max_c = -3, 0
        data[:, 0] = data[:, 0] * (max_a - min_a) + min_a
        data[:, 1] = data[:, 1] * (max_b - min_b) + min_b
        #data[:, 2] = data[:, 2] * (max_c - min_c) + min_c
        data[:,2] = -torch.sqrt((2*LINK_LENGTH)**2*torch.ones_like(data[:,0]) - data[:,0]*data[:,0])
        #data = -1*data  # normals should point up (against gravity)

    def sample_plane_d_coefficient_offsets(
        self,
        batch_size: int,
        max_d_offset: float,
        data: torch.Tensor,
    ) -> None:
        assert data.shape[0] == batch_size and data.shape[1] == 1
        data.uniform_()
        min_d_offset = 0 
        data[...] = data * (max_d_offset - min_d_offset) + min_d_offset


class WarpSimDataGenerator:
    """Generic data generator for WarpSim environments."""

    def __init__(
        self,
        env: NnTrainingInterface,
        joint_q_min: Union[float, np.ndarray],
        joint_q_max: Union[float, np.ndarray],
        joint_qd_min: Union[float, np.ndarray],
        joint_qd_max: Union[float, np.ndarray],
        joint_act_scale: Union[float, np.ndarray],
        contact_prob: float = 0.,
        sampler=UniformSampler()
    ):
        self.env = env
        self.num_envs = env.num_envs

        # joint position limits
        if isinstance(joint_q_min, np.ndarray):
            assert len(joint_q_min) == self.env.dof_q_per_env
            q_lower = joint_q_min.copy()
        else:
            q_lower = np.full(self.env.dof_q_per_env, joint_q_min)
    
        if isinstance(joint_q_max, np.ndarray):
            assert len(joint_q_max) == self.env.dof_q_per_env
            q_upper = joint_q_max.copy()
        else:
            q_upper = np.full(self.env.dof_q_per_env, joint_q_max)
        
        # joint velocity limits
        if isinstance(joint_qd_min, np.ndarray):
            assert len(joint_qd_min) == self.env.dof_qd_per_env
            qd_lower = joint_qd_min
        else:
            qd_lower = np.full(self.env.dof_qd_per_env, joint_qd_min)
        if isinstance(joint_qd_max, np.ndarray):
            assert len(joint_qd_max) == self.env.dof_qd_per_env
            qd_upper = joint_qd_max
        else:
            qd_upper = np.full(self.env.dof_qd_per_env, joint_qd_max)

        states_min = np.concatenate([q_lower, qd_lower])
        states_max = np.concatenate([q_upper, qd_upper])

        actions_min = np.full(self.env.action_dim, -1.)
        actions_max = np.full(self.env.action_dim, 1.)
        for i in range(env.action_dim):
            actions_min[i] = env.action_limits[i][0]
            actions_max[i] = env.action_limits[i][1]

        if isinstance(joint_act_scale, np.ndarray):
            assert len(joint_act_scale) == self.env.action_dim
            self.joint_act_scale = torch.tensor(
                joint_act_scale, 
                dtype=torch.float32, 
                device=self.torch_device
            )
        else:
            self.joint_act_scale = torch.full(
                (self.env.action_dim,),
                joint_act_scale, 
                dtype=torch.float32, 
                device=self.torch_device
            )

        self.contact_prob = contact_prob

        self.states_range = torch.tensor(
            states_max - states_min,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.states_min = torch.tensor(
            states_min, 
            dtype=torch.float32, 
            device=self.torch_device
        )
        self.states_max = torch.tensor(
            states_max,
            dtype=torch.float32,
            device=self.torch_device
        )

        self.actions_range = torch.tensor(
            actions_max - actions_min,
            dtype=torch.float32,
            device=self.torch_device
        )
        self.actions_min = torch.tensor(
            actions_min, 
            dtype=torch.float32, 
            device=self.torch_device
        )
        self.actions_max = torch.tensor(
            actions_max, 
            dtype=torch.float32, 
            device=self.torch_device
        )

        self.sampler = sampler

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def action_dim(self):
        return self.env.action_dim
    
    @property
    def torch_device(self):
        return wp.device_to_torch(self.env.device)
