#!/usr/bin/env python3
"""
Example script demonstrating how to use the standalone NeRD predictor.

This script shows how to:
1. Load a pretrained NeRD model
2. Create a NeuralPredictor instance
3. Prepare input data (states, actions, contacts, gravity)
4. Make predictions for next robot states

Example usage:
    python example_usage.py --model-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/nn/model.pts \
                             --cfg-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/cfg.yaml
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[4]  # standalone → neural_solver → axion → src → repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.axion.neural_solver.standalone.neural_predictor import NeuralPredictor

def load_model_and_config(model_path, cfg_path):
    """Load pretrained model and configuration."""
    print(f"Loading model from: {model_path}")
    model, robot_name = torch.load(model_path, map_location='cuda:0', weights_only= False)
    print(f"Loaded model for robot: {robot_name}")
    
    print(f"Loading configuration from: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    return model, robot_name, cfg


def create_pendulum_predictor(model, cfg, device='cuda:0'):
    """
    Create NeuralPredictor for Pendulum robot.

    Pendulum configuration:
    - 2 revolute joints (double pendulum)
    - Each joint: 1 DOF (angle) + 1 velocity
    - Total: 2 position DOFs + 2 velocity DOFs = 4 state dimensions
    """
    # NeuralPredictor reads cfg['env']['neural_integrator_cfg']; training uses utils_provider_cfg
    env_cfg = cfg.get('env', {})
    if 'neural_integrator_cfg' not in env_cfg and 'utils_provider_cfg' in env_cfg:
        cfg = {**cfg, 'env': {**env_cfg, 'neural_integrator_cfg': env_cfg['utils_provider_cfg']}}
    return NeuralPredictor(
        nn_model=model,
        cfg=cfg,
        device=device,
        # Robot configuration for Pendulum
        dof_q_per_env=2,      # 2 revolute joints, each with 1 angle
        dof_qd_per_env=2,     # 2 revolute joints, each with 1 angular velocity
        # Joint types: both are REVOLUTE (Newton JointType.REVOLUTE = 1; BALL = 2)
        joint_types=[1, 1],
        # Joint DOF start indices in q vector
        joint_q_start=[0, 1],
        # Joint DOF end indices in q vector
        joint_q_end=[1, 2],
        # Which DOFs are angular (for state embedding): position DOFs only
        # [angle1, angle2, vel1, vel2]
        is_angular_dof=[True, True, False, False],
        # Which DOFs are continuous (unwrapped angles)
        # Position DOFs (angles) are continuous, velocities are not
        is_continuous_dof=[True, True, False, False]
    )


def create_example_inputs(num_envs=1, device='cuda:0'):
    """
    Create example input data for prediction.

    Returns:
        states: Current robot states (num_envs, state_dim)
        joint_acts: Joint actions/torques (num_envs, dof_qd_per_env)
        root_body_q: Root body pose (num_envs, 7) [x, y, z, qx, qy, qz, qw]
        contacts: Dictionary with contact information (empty for pendulum)
        gravity_dir: Gravity direction (num_envs, 3)
    """
    # Example: Double pendulum - states: [angle1, angle2, angular_vel1, angular_vel2]
    states = torch.zeros((num_envs, 4), device=device)
    states[:, 0] = torch.tensor([np.deg2rad(0.0)], device=device)
    states[:, 1] = torch.tensor([np.deg2rad(0.0)], device=device)
    states[:, 2] = torch.tensor([0.0], device=device)
    states[:, 3] = torch.tensor([0.0], device=device)

    # Joint actions (2 torques for double pendulum); zeros for passive rollout
    joint_acts = torch.zeros((num_envs, 2), device=device)

    # Root body pose [x, y, z, qx, qy, qz, qw]
    root_body_q = torch.zeros((num_envs, 7), device=device)
    root_body_q[:, 0:3] = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    root_body_q[:, 3:7] = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)

    # No contacts for pendulum
    num_contacts = 0
    contacts = {
        'contact_normals': torch.zeros((num_envs, num_contacts * 3), device=device),
        'contact_depths': torch.zeros((num_envs, num_contacts), device=device),
        'contact_thicknesses': torch.zeros((num_envs, num_contacts), device=device),
        'contact_points_0': torch.zeros((num_envs, num_contacts * 3), device=device),
        'contact_points_1': torch.zeros((num_envs, num_contacts * 3), device=device),
    }

    # Gravity direction (Y-up: negative Y)
    gravity_dir = torch.zeros((num_envs, 3), device=device)
    gravity_dir[:, 1] = -1.0

    return states, root_body_q, gravity_dir


def run_prediction_example(model_path, cfg_path, device='cuda:0', num_steps=10):
    """Run a complete prediction example."""
    print("=" * 60)
    print("NeRD Standalone Predictor Example")
    print("=" * 60)
    
    # Step 1: Load model and configuration
    model, robot_name, cfg = load_model_and_config(model_path, cfg_path)
    print(f"\n✓ Model and config loaded successfully")
    
    # Step 2: Create predictor
    print("\nCreating NeuralPredictor...")
    # Note: You'll need to adjust the robot configuration based on your specific robot
    # This example uses Pendulum configuration - adjust for your robot!
    predictor = create_pendulum_predictor(model, cfg, device=device)
    print(f"✓ Predictor created for device: {device}")
    print(f"  State dimension: {predictor.state_dim} (dof_q={predictor.dof_q_per_env}, dof_qd={predictor.dof_qd_per_env})")

    # Step 3: Create example inputs
    print("\nCreating example input data...")
    states, root_body_q, gravity_dir = create_example_inputs(
        num_envs=1, device=device
    )
    print(f"✓ Input data created")
    print(f"  States shape: {states.shape}")
    print(f"  Root body q shape: {root_body_q.shape}")
    print(f"  Gravity dir shape: {gravity_dir.shape}")

    # Step 4: Run predictions
    print("\nRunning predictions...")
    print("-" * 60)

    predictor.reset()
    current_states = states.clone()
    trajectory = [current_states.cpu().numpy()]

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}:")
        print(f"  Current state: {current_states[0].cpu().numpy()}")
        print(f"    Joint angles (deg): {np.rad2deg(current_states[0, :2].cpu().numpy())}")
        print(f"    Joint velocities: {current_states[0, 2:].cpu().numpy()}")

        # Process inputs (fills predictor.nn_model_inputs), then run inference
        predictor.process_inputs(
            states=current_states,
            root_body_q=root_body_q,
            gravity_dir=gravity_dir,
        )
        next_states = predictor.predict()
        
        print(f"  Next state: {next_states[0].cpu().numpy()}")
        print(f"    Joint angles (deg): {np.rad2deg(next_states[0, :2].cpu().numpy())}")
        print(f"    Joint velocities: {next_states[0, 2:].cpu().numpy()}")
        print(f"  State change: {(next_states - current_states)[0].cpu().numpy()}")
        
        # Update for next iteration
        current_states = next_states
        trajectory.append(current_states.cpu().numpy())
        
        print()
    
    print("-" * 60)
    print(f"✓ Completed {num_steps} prediction steps")
    print(f"  Trajectory shape: {len(trajectory)} steps x {trajectory[0].shape}")
    
    return trajectory

def run_argparser():
    parser = argparse.ArgumentParser(description='Example usage of NeRD standalone predictor')
    parser.add_argument(
        '--model-path',
        type=str,
        default='src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/nn/model.pt',
        help='Path to pretrained model file (.pt)'
    )
    parser.add_argument(
        '--cfg-path',
        type=str,
        default='src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/cfg.yaml',
        help='Path to model configuration file (.yaml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=500,
        help='Number of prediction steps to run'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = run_argparser()
    
    filesFound = True

    # Check if files exist
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("Please provide a valid path to a pretrained NeRD model.")
        filesFound = False
    
    if not Path(args.cfg_path).exists():
        print(f"Error: Config file not found: {args.cfg_path}")
        print("Please provide a valid path to the model configuration file.")
        filesFound = False
    
    if filesFound:
        try:
            trajectory = run_prediction_example(
                model_path=args.model_path,
                cfg_path=args.cfg_path,
                device=args.device,
                num_steps=args.num_steps
            )
            print("\n" + "=" * 60)
            print("Example completed successfully!")
            print("=" * 60)
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Make sure you have:")
            print("  1. Correct robot configuration in create_pendulum_predictor()")
            print("  2. Correct input data shapes matching your robot")
            print("  3. Valid pretrained model and config files")