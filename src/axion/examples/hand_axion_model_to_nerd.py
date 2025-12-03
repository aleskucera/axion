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
Simple script to test NeRD model with PendulumWithContactEnvironment.
- Loads pretrained NeRD model
- Initializes environment with visualization
- Sets non-trivial initial conditions
- Runs simulation with rendering
"""
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import torch
import yaml
import numpy as np
import newton
import warp as wp
from axion.core.control_utils import JointMode
from pathlib import Path

from nerd.envs.neural_environment import NeuralEnvironment
from nerd.utils.torch_utils import num_params_torch_model
from nerd.utils.python_utils import set_random_seed

base_dir = Path(__file__).resolve().parents[3]

def create_model_builder_with_custom_attributes():
    builder = newton.ModelBuilder()

    # --- Add custom attributes to the model class ---

    # integral constant (PID control)
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="joint_target_ki",
            frequency=newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.float32,
            default=0.0,  # Explicit default value
            assignment=newton.ModelAttributeAssignment.MODEL,
        )
    )

    # previous instance of the control error (PID control)
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="joint_err_prev",
            frequency=newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.float32,
            default=0.0,
            assignment=newton.ModelAttributeAssignment.CONTROL,
        )
    )

    # cummulative error of the integral part (PID control)
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="joint_err_i",
            frequency=newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.float32,
            default=0.0,
            assignment=newton.ModelAttributeAssignment.CONTROL,
        )
    )

    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="joint_dof_mode",
            frequency=newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.int32,
            default=JointMode.NONE,
            assignment=newton.ModelAttributeAssignment.MODEL,
        )
    )

    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="joint_target",
            frequency=newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.float32,
            default=0,
            assignment=newton.ModelAttributeAssignment.CONTROL,
        )
    )
    
    return builder


def create_custom_pendulum_builder():
    builder = create_model_builder_with_custom_attributes()
    
    chain_width = 1.5
    shape_ke = 1.0e4
    shape_kd = 1.0e3
    shape_kf = 1.0e4

    hx = chain_width*0.5

    link_0 = builder.add_body(armature=0.1)
    link_config = newton.ModelBuilder.ShapeConfig(density=500.0, ke = shape_ke, kd = shape_kd, kf = shape_kf)
    capsule_shape_transform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi/2))
    builder.add_shape_capsule(link_0,
                                    xform= capsule_shape_transform,
                                    radius=0.1, 
                                    half_height=chain_width*0.5,
                                    cfg = link_config)

    link_1 = builder.add_body(armature=0.1)
    builder.add_shape_capsule(link_1,
                                xform = capsule_shape_transform,
                                radius=0.1, 
                                half_height=chain_width*0.5,
                                cfg = link_config)

    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        custom_attributes={
            "joint_target_ki": [0.5],
            "joint_dof_mode": [JointMode.NONE],
        },
    )
    builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=500.0,
        target_kd=5.0,
        custom_attributes={
            "joint_target_ki": [0.5],
            "joint_dof_mode": [JointMode.NONE],
        },
        armature=0.1,
    )

    builder.add_ground_plane()

    return builder


if __name__ == '__main__':
    # Configuration
    device = 'cuda:0'
    model_path = base_dir /'third_party'/ 'nerd'/ 'nerd' /'pretrained_models' / 'NeRD_models' / 'Pendulum' / 'model' / 'nn' / 'model.pt'
    num_envs = 1
    num_steps = 5000
    seed = 42
    
    set_random_seed(seed)
    
    # Load pretrained NeRD model
    print("Loading pretrained NeRD model...")
    neural_model, robot_name = torch.load(model_path, map_location=device, weights_only=False)
    print(f'Number of Model Parameters: {num_params_torch_model(neural_model)}')
    neural_model.to(device)
    
    # Load model configuration
    train_dir = (model_path.parent.parent).resolve()
    cfg_path = os.path.join(train_dir, 'cfg.yaml')
    print(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]
    
    # Create the custom articulation builder
    custom_articulation_builder = create_custom_pendulum_builder()
    print(f"Custom pendulum created: {custom_articulation_builder.body_count} bodies, "
          f"{custom_articulation_builder.joint_count} joints")

    # Initialize environment with visualization
    print("Initializing PendulumWithContactEnvironment with visualization...")
    env_cfg = {
        "env_name": "PendulumWithContact",
        "num_envs": num_envs,
        "render": True,  # Enable visualization
        "warp_env_cfg": {
            "seed": seed
        },
        "neural_integrator_cfg": neural_integrator_cfg,
        "neural_model": neural_model,
        "default_env_mode": "neural",  # Use NeRD model
        "device": device,
        "custom_articulation_builder": custom_articulation_builder  # Pass custom builder
    }
    
    neural_env = NeuralEnvironment(**env_cfg)
    
    assert neural_env.robot_name == robot_name, \
        "neural_env.robot_name is not equal to neural_model's robot_name."
    
    print(f"Environment initialized. State dimension: {neural_env.state_dim}")
    print(f"Action dimension: {neural_env.action_dim}")
    
    # Set non-trivial initial conditions
    # For Pendulum: state = [θ1, θ2, θ̇1, θ̇2] (positions + velocities)
    # State dimension = dof_q_per_env + dof_qd_per_env = 2 + 2 = 4
    print("\nSetting non-trivial initial conditions...")
    initial_states = torch.zeros((num_envs, neural_env.state_dim), device=device)
    
    # Set initial joint angles (in radians)
    # First joint: 45 degrees, Second joint: -30 degrees
    initial_states[0, 0] = np.deg2rad(-90)   # θ1 = 45°
    initial_states[0, 1] = np.deg2rad(-90)  # θ2 = -30°
    
    # Set initial joint velocities (in rad/s)
    # Give some initial angular velocities for more interesting motion
    initial_states[0, 2] = 1.0   # θ̇1 = 1.0 rad/s
    initial_states[0, 3] = -0.5  # θ̇2 = -0.5 rad/s
    
    print(f"Initial state:")
    print(f"  θ1 = {np.rad2deg(initial_states[0, 0].item()):.2f}°")
    print(f"  θ2 = {np.rad2deg(initial_states[0, 1].item()):.2f}°")
    print(f"  θ̇1 = {initial_states[0, 2].item():.2f} rad/s")
    print(f"  θ̇2 = {initial_states[0, 3].item():.2f} rad/s")
    
    # Reset environment with initial states
    neural_env.reset(initial_states=initial_states)
    
    # Run simulation loop
    print(f"\nRunning simulation for {num_steps} steps...")
    print("Close the visualization window to stop the simulation.")
    
    # For passive motion, we use zero actions (no control input)
    zero_actions = torch.zeros((num_envs, neural_env.action_dim), device=device)
    
    try:
        for step in range(num_steps):
            # Step forward with zero actions (passive motion)
            states = neural_env.step(zero_actions, env_mode='neural')
            
            # Render the simulation
            neural_env.render()
            
            # Print state every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: θ1 = {np.rad2deg(states[0, 0].item()):.2f}°, "
                      f"θ2 = {np.rad2deg(states[0, 1].item()):.2f}°, "
                      f"θ̇1 = {states[0, 2].item():.2f} rad/s, "
                      f"θ̇2 = {states[0, 3].item():.2f} rad/s")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    print("Simulation completed.")
    neural_env.close()

