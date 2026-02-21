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

import argparse
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

import yaml
import warp as wp
wp.config.verify_cuda = True

import wandb

from axion.neural_solver.utils.python_utils import get_time_stamp, set_random_seed
from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
from axion.neural_solver.envs.axionToTrajectorySampler import AxionEnvToTrajectorySamplerAdapter


def _parse_args():
    p = argparse.ArgumentParser(description='Train transformer from YAML config.')
    p.add_argument('--cfg', required=True, help='Path to config YAML')
    p.add_argument('--logdir', required=True, help='Directory for logs and checkpoints')
    p.add_argument('--test', action='store_true', help='Run evaluation instead of training')
    p.add_argument('--checkpoint', default=None, help='Checkpoint to restore')
    p.add_argument('--no-time-stamp', action='store_true', help='No timestamp subfolder under logdir')
    p.add_argument('--device', default='cuda:0', help='Device (e.g. cuda:0)')
    return p.parse_args()


if __name__ == '__main__':
    # Initiate weights and biases logging
    wandb.login()

    args = _parse_args()

    # Read train/cfg/Pendulum/transformer.yaml for example
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Require transformer config
    assert 'transformer' in cfg['network'], "Only transformer model is supported; config must define network.transformer"
    neural_integrator_name = cfg['env']['neural_integrator_cfg']['name']
    assert neural_integrator_name == 'TransformerNeuralIntegrator', (
        "Only TransformerNeuralIntegrator is supported. Got: " + neural_integrator_name
    )
    assert (
        cfg['env']['neural_integrator_cfg'].get('num_states_history') ==
        cfg['algorithm']['sample_sequence_length']
    ), (
        "'num_states_history' must equal 'sample_sequence_length' for the transformer."
    )

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    seed = cfg['algorithm'].get('seed', 0)
    set_random_seed(seed)
    cfg['algorithm']['seed'] = seed

    args.train = not args.test

    cfg['cli'] = {
        'logdir': args.logdir,
        'train': args.train,
        'render': cfg['env'].get('render', False),
        'save_interval': cfg['algorithm'].get('save_interval', 50),
        'log_interval': cfg['algorithm'].get('log_interval', 1),
        'eval_interval': cfg['algorithm'].get('eval_interval', 1),
        'skip_check_log_override': False,
    }

    neural_env = AxionEnvToTrajectorySamplerAdapter(**cfg['env'], device=args.device)

    algo = SequenceModelTrainer(
        neural_env=neural_env,
        model_checkpoint_path=args.checkpoint,
        cfg=cfg,
        device=args.device
    )

    if args.train:
        print("Begin torch module training")
        algo.train()
    else:
        print("Begin torch module testing")
        algo.test()
