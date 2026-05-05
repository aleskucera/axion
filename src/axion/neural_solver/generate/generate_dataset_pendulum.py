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
Generate dataset for Pendulum
"""
import sys
import os

# Project root (directory containing 'src') so that "from axion.neural_solver..." works
# when run as: python src/axion/neural_solver/generate/generate_dataset_pendulum.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
_src_root = os.path.join(_project_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# Default directory for generated datasets (inside project)
_DEFAULT_DATASET_DIR = os.path.join(_project_root, "src/axion/neural_solver/datasets")

import os
import argparse
import numpy as np
import h5py
from axion.core.types import JointMode
from axion.neural_solver.generate.trajectory_sampler_pendulum import TrajectorySamplerPendulum
from axion.neural_solver.utils.python_utils import set_random_seed
from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.utils.commons import (
    JOINT_Q_MIN,
    JOINT_Q_MAX,
    JOINT_QD_MIN,
    JOINT_QD_MAX,
)

def collect_dataset(
    env_name, 
    num_envs, 
    num_transitions, 
    trajectory_length, 
    contact_prob, 
    with_contacts: bool,
    dataset_path, 
    device: str = "cuda:0",
    passive = False,
    seed = 0, 
    render = False,
    export_video = False,
    export_video_path = None
):
    data_writer = h5py.File(dataset_path, 'w')
    data_grp = data_writer.create_group('data')
    data_grp.attrs['env'] = env_name
    data_grp.attrs['mode'] = "trajectory"
    
    joint_dof_mode = JointMode.NONE if passive else JointMode.TARGET_POSITION

    env = NnTrainingInterface(
        env_name = env_name,
        num_envs = num_envs,
        utils_provider_cfg = {},
        neural_model = None,
        default_env_mode = "ground-truth",
        device = device,
        warp_env_cfg = {
            "seed": seed,
            "with_contacts": with_contacts,
            "joint_dof_mode": joint_dof_mode,
        },
        render = render
    )
    
    robot_name = env.robot_name
    simulation_sampler = \
        TrajectorySamplerPendulum(
            env,
            joint_q_min = JOINT_Q_MIN[robot_name],
            joint_q_max = JOINT_Q_MAX[robot_name],
            joint_qd_min = JOINT_QD_MIN[robot_name],
            joint_qd_max = JOINT_QD_MAX[robot_name],
            contact_prob = contact_prob,
            with_contacts = with_contacts,
            joint_target_min=np.array([0.0, 0.0 - np.pi / 3.0], dtype=np.float64),
            joint_target_max=np.array([np.pi, np.pi / 3.0], dtype=np.float64),
        )
    
    rollouts = \
        simulation_sampler.sample_trajectories(
            num_transitions, 
            trajectory_length, 
            passive,
            render=render,
            export_video=export_video,
            export_video_path=export_video_path
        )

    data_grp.attrs['total_trajectories'] = rollouts['states'].shape[1]
    data_grp.attrs['total_transitions'] = rollouts['states'].shape[0] * rollouts['states'].shape[1]

    data_grp.create_dataset(
        name = 'gravity_dir',
        data = rollouts['gravity_dir'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'plane_coefficients',
        data = rollouts['plane_coefficients'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'root_body_q',
        data = rollouts['root_body_q'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'states', 
        data = rollouts['states'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_normals',
        data = rollouts['contacts']['contact_normals'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_depths',
        data = rollouts['contacts']['contact_depths'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_points_0',
        data = rollouts['contacts']['contact_points_0'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_points_1',
        data = rollouts['contacts']['contact_points_1'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_thicknesses',
        data = rollouts['contacts']['contact_thicknesses'].detach().cpu().numpy()
    )
    axion_contacts_grp = data_grp.create_group('axion_contacts')
    axion_contacts_grp.create_dataset(
        name='contact_count',
        data=rollouts['axion_contacts']['contact_count'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_point0',
        data=rollouts['axion_contacts']['contact_point0'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_point1',
        data=rollouts['axion_contacts']['contact_point1'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_normal',
        data=rollouts['axion_contacts']['contact_normal'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_shape0',
        data=rollouts['axion_contacts']['contact_shape0'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_shape1',
        data=rollouts['axion_contacts']['contact_shape1'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_thickness0',
        data=rollouts['axion_contacts']['contact_thickness0'].detach().cpu().numpy()
    )
    axion_contacts_grp.create_dataset(
        name='contact_thickness1',
        data=rollouts['axion_contacts']['contact_thickness1'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name="joint_target_pos",
        data=rollouts["joint_target_pos"].detach().cpu().numpy(),
    )
    data_grp.create_dataset(
        name = 'next_states', 
        data = rollouts['next_states'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'lambdas', 
        data = rollouts['lambdas'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'next_lambdas', 
        data = rollouts['next_lambdas'].detach().cpu().numpy()
    )
    data_grp.attrs['state_dim'] = rollouts['states'].shape[-1]
    # data_grp.attrs['contact_prob'] = contact_prob
    data_grp.attrs['num_contacts_per_env'] = rollouts['contacts']['contact_depths'].shape[-1]
    data_grp.attrs['axion_contacts_format'] = (
        "batched AxionContacts arrays captured before convert_newton_contacts_to_contacts_for_nn_model"
    )
    data_grp.attrs["joint_target_dim"] = rollouts["joint_target_pos"].shape[-1]
    data_grp.attrs['next_state_dim'] = rollouts['next_states'].shape[-1]

    
    data_writer.flush()
    data_writer.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', 
                        type=str, 
                        default=_DEFAULT_DATASET_DIR,
                        help='Directory to store the generated datasets (default: <project_root>/datasets).')
    parser.add_argument('--env-name', 
                        type=str, 
                        default='Pendulum', 
                        choices=['Pendulum'],
                        help='Environment to generate the dataset.' )
    parser.add_argument('--num-transitions', 
                        type=int,
                        default=1000000,
                        help='The total number of transitions to be collected. ')
    parser.add_argument('--trajectory-length', 
                        type=int,
                        default=100,
                        help='The length of each trajectory. Valid only if mode is trajectory.')
    parser.add_argument('--passive',
                        action='store_true',
                        help="Whether use passive simulation.")
    parser.add_argument('--dataset-name',
                        type=str,
                        default='dataset.hdf5',
                        help='The filename of the newly collected dataset.')
    parser.add_argument('--num-envs',
                        type=int,
                        default=1024,
                        help='The number of parallel environments.')
    parser.add_argument('--contact-prob',
                        type=float,
                        default=0.5,
                        help='The probablity to sample a valid contact.')
    parser.add_argument(
        '--without-contacts',
        action='store_true',
        help="Disable the tilted contact plane in the simulator. By default, dataset generation uses contacts.",
    )
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='The random seed for sampling.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help="Simulation device (e.g. 'cuda:0', 'cuda:1', or 'cpu').",
    )
    parser.add_argument('--render',
                        action='store_true',
                        help='Whether to render the simulation.')
    parser.add_argument('--export-video',
                        action = 'store_true')
    parser.add_argument('--export-video-path',
                        type = str,
                        default = 'video.gif')
    
    args = parser.parse_args()

    set_random_seed(args.seed)

    dataset_path = os.path.join(args.dataset_dir, args.env_name, args.dataset_name)
    if os.path.exists(dataset_path):
        answer = input(f'Dataset exists in the specified path {dataset_path}, do you want to clean the old dataset [y/n]')
        if answer != 'y' and answer != 'Y':
            exit()
    
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    collect_dataset(
        env_name=args.env_name, 
        num_envs=args.num_envs,
        num_transitions=args.num_transitions, 
        trajectory_length=args.trajectory_length,
        contact_prob=args.contact_prob,
        with_contacts=not args.without_contacts,
        dataset_path=dataset_path,
        device=args.device,
        passive=args.passive,
        seed=args.seed,
        render=args.render,
        export_video=args.export_video,
        export_video_path=args.export_video_path
    )

