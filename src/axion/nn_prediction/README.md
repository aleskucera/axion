# Standalone NeRD Model Integration

This directory contains a minimal, standalone implementation for using pretrained NeRD models as black-box robot dynamics predictors. This code can be integrated into other projects without requiring the full NeRD codebase, Warp simulator, or RL training infrastructure.

## Overview

The standalone implementation provides:
- **NeRDPredictor**: A simple class interface for loading and using pretrained models
- **Model architectures**: Complete neural network implementations (MLP, RNN, Transformer)
- **State processing**: Coordinate frame conversions, state wrapping, contact processing
- **Utilities**: Quaternion operations, coordinate transformations, normalization

## Files Structure

```
nerd_standalone/
├── __init__.py
├── nerd_predictor.py          # Main predictor class
├── models/
│   ├── __init__.py
│   ├── base_models.py         # MLP, LSTM, GRU base architectures
│   ├── model_transformer.py   # Transformer architecture
│   ├── model_utils.py         # Model utilities
│   └── models.py              # Main model wrapper (ModelMixedInput)
├── utils/
│   ├── __init__.py
│   ├── torch_utils.py         # Quaternion and coordinate transform utilities
│   ├── running_mean_std.py    # Normalization utilities
│   └── commons.py              # Constants
└── integrators/
    ├── __init__.py
    └── state_processor.py     # State processing and conversion functions
```

## Quick Start

See `example_usage.py` for a complete working example. Run it with:

```bash
python example_usage.py --model-path pretrained_models/NeRD_models/Pendulum/model/nn/model.pt \
                         --cfg-path pretrained_models/NeRD_models/Pendulum/model/cfg.yaml
```

## Usage

### Basic Example

```python
import torch
import yaml
from nerd_standalone import NeRDPredictor

# Load pretrained model
model_path = "pretrained_models/NeRD_models/Pendulum/model/nn/model.pt"
cfg_path = "pretrained_models/NeRD_models/Pendulum/model/cfg.yaml"

model, robot_name = torch.load(model_path, map_location='cuda:0')
with open(cfg_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

# Create predictor with robot-specific configuration
predictor = NeRDPredictor(
    model=model,
    cfg=cfg,
    device='cuda:0',
    # Robot configuration (example for Pendulum)
    dof_q_per_env=4,      # 2 joints * 2 DOFs each (position + orientation)
    dof_qd_per_env=4,     # 2 joints * 2 DOFs each (angular velocity)
    joint_act_dim=1,      # 1 actuator
    num_contacts_per_env=1,
    joint_types=[0, 2],   # FREE joint, REVOLUTE joint
    joint_q_start=[0, 7],
    joint_q_end=[7, 9],
    is_angular_dof=[False, False, False, False, True, True, True, True, False, False, False, False],
    is_continuous_dof=[False, False, False, False, False, False, False, False, False, False, False, False]
)

# Predict next state
next_states = predictor.predict(
    states=current_states,  # (num_envs, state_dim)
    joint_acts=joint_torques,  # (num_envs, joint_act_dim)
    root_body_q=root_pose,  # (num_envs, 7) [x,y,z,qx,qy,qz,qw]
    contacts={
        'contact_normals': contact_normals,  # (num_envs, num_contacts * 3)
        'contact_depths': contact_depths,  # (num_envs, num_contacts)
        'contact_thicknesses': contact_thickness,  # (num_envs, num_contacts)
        'contact_points_0': contact_points_0,  # (num_envs, num_contacts * 3)
        'contact_points_1': contact_points_1,  # (num_envs, num_contacts * 3)
    },
    gravity_dir=gravity  # (num_envs, 3)
)
```

## Input Format

### States
States should be in generalized coordinates: `[joint_q, joint_qd]`
- `joint_q`: Joint positions (num_envs, dof_q_per_env)
  - For FREE joints: [x, y, z, qx, qy, qz, qw]
  - For REVOLUTE/PRISMATIC: joint angle/position
- `joint_qd`: Joint velocities (num_envs, dof_qd_per_env)
  - For FREE joints: [ωx, ωy, ωz, vx, vy, vz]
  - For REVOLUTE/PRISMATIC: joint angular/linear velocity

### Contacts
Contact information should be provided as a dictionary with:
- `contact_normals`: (num_envs, num_contacts * 3) - Contact normal vectors
- `contact_depths`: (num_envs, num_contacts) - Penetration depths
- `contact_thicknesses`: (num_envs, num_contacts) - Contact thickness
- `contact_points_0`: (num_envs, num_contacts * 3) - Contact point on body 0
- `contact_points_1`: (num_envs, num_contacts * 3) - Contact point on body 1

### Root Body Pose
- `root_body_q`: (num_envs, 7) - Root body pose [x, y, z, qx, qy, qz, qw]

### Gravity
- `gravity_dir`: (num_envs, 3) - Gravity direction vector (typically [0, 0, -1] or [0, -1, 0])

## Robot Configuration

You need to provide robot-specific configuration when creating the predictor:

- **dof_q_per_env**: Number of position DOFs
- **dof_qd_per_env**: Number of velocity DOFs
- **joint_act_dim**: Number of joint actuators
- **num_contacts_per_env**: Number of contact pairs
- **joint_types**: List of joint types (0=FREE, 1=BALL, 2=REVOLUTE, 3=PRISMATIC, 4=FIXED, 5=DISTANCE)
- **joint_q_start**: Start indices for each joint in the q vector
- **joint_q_end**: End indices for each joint in the q vector
- **is_angular_dof**: Boolean array indicating which DOFs are angular
- **is_continuous_dof**: Boolean array indicating which DOFs are continuous (unwrapped angles)

## Dependencies

- PyTorch
- NumPy
- PyYAML (for loading config files)
- warp (optional, but can be used - note: `warp.sim` is NOT required)

## Notes

- The model expects inputs with a time dimension: `(num_envs, T, dim)` where T=1 for non-transformer models
- For transformer models, T should match the sequence length used during training
- Contact information must be provided in the same format as the original NeRD implementation
- States are automatically wrapped to [-π, π] for continuous angular DOFs
- Coordinate frame conversions (world ↔ body) are handled automatically based on configuration

