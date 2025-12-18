"""
Standalone NeRD Predictor

This module provides a simplified interface for using pretrained NeRD models
as black-box robot dynamics predictors without requiring Warp or the full
NeRD codebase infrastructure.

Example usage:
    import torch
    import yaml
    from nerd_standalone.nerd_predictor import NeRDPredictor
    
    # Load model and config
    model, robot_name = torch.load("pretrained_models/NeRD_models/Pendulum/model/nn/model.pt")
    with open("pretrained_models/NeRD_models/Pendulum/model/cfg.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Create predictor
    predictor = NeRDPredictor(
        model=model,
        cfg=cfg,
        device='cuda:0',
        # Robot-specific configuration
        dof_q_per_env=4,  # Number of position DOFs
        dof_qd_per_env=4,  # Number of velocity DOFs
        joint_act_dim=1,  # Number of joint actuators
        num_contacts_per_env=1,  # Number of contact pairs
        joint_types=[0, 2],  # Joint types (0=FREE, 1=BALL, 2=REVOLUTE, etc.)
        joint_q_start=[0, 7],  # Start indices for each joint in q vector
        joint_q_end=[7, 9],  # End indices for each joint in q vector
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
"""

import torch
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
from collections import deque

from .models.models import ModelMixedInput
from .integrators.state_processor import (
    get_contact_masks,
    wrap2PI,
    convert_coordinate_frame,
    convert_states_back_to_world,
    embed_states,
    convert_prediction_to_next_states
)

from axion.nn_prediction.utils.analysis_utils import write_model_inputs_to_csv

class NeRDPredictor:
    """
    Standalone NeRD model predictor.
    
    This class provides a simplified interface for using pretrained NeRD models
    to predict next robot states given current states, actions, contacts, and gravity.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict,
        device: str = 'cuda:0',
        # Robot-specific configuration
        dof_q_per_env: int = None,
        dof_qd_per_env: int = None,
        joint_act_dim: int = None,
        num_contacts_per_env: int = None,
        joint_types: list = None,
        joint_q_start: list = None,
        joint_q_end: list = None,
        is_angular_dof: list = None,
        is_continuous_dof: list = None,
    ):
        """
        Initialize NeRD predictor.
        
        Args:
            model: Pretrained NeRD model (loaded via torch.load)
            cfg: Configuration dictionary from cfg.yaml
            device: Device to run on ('cuda:0', 'cpu', etc.)
            dof_q_per_env: Number of position DOFs per environment
            dof_qd_per_env: Number of velocity DOFs per environment
            joint_act_dim: Number of joint actuators
            num_contacts_per_env: Number of contact pairs per environment
            joint_types: List of joint types (0=FREE, 1=BALL, 2=REVOLUTE, etc.)
            joint_q_start: Start indices for each joint in q vector
            joint_q_end: End indices for each joint in q vector
            is_angular_dof: Boolean array indicating which DOFs are angular
            is_continuous_dof: Boolean array indicating which DOFs are continuous (unwrapped)
        """
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()
        
        # Load configuration
        self.neural_integrator_cfg = cfg.get('env', {}).get('neural_integrator_cfg', {})
        self.states_frame = self.neural_integrator_cfg.get('states_frame', 'body')
        self.anchor_frame_step = self.neural_integrator_cfg.get('anchor_frame_step', 'every')
        self.prediction_type = self.neural_integrator_cfg.get('prediction_type', 'relative')
        self.orientation_prediction_parameterization = self.neural_integrator_cfg.get(
            'orientation_prediction_parameterization', 'quaternion'
        )
        self.states_embedding_type = self.neural_integrator_cfg.get('states_embedding_type', None)
        self.num_states_history = self.neural_integrator_cfg.get('num_states_history', 1)   # history window, defaults to 1

        # state history double ended queue
        self.states_history = deque(maxlen=self.num_states_history)

        # Robot configuration
        if dof_q_per_env is None:
            raise ValueError("dof_q_per_env must be provided")
        if dof_qd_per_env is None:
            raise ValueError("dof_qd_per_env must be provided")
        if joint_act_dim is None:
            raise ValueError("joint_act_dim must be provided")
        if num_contacts_per_env is None:
            raise ValueError("num_contacts_per_env must be provided")
        
        self.dof_q_per_env = dof_q_per_env
        self.dof_qd_per_env = dof_qd_per_env
        self.state_dim = dof_q_per_env + dof_qd_per_env
        self.joint_act_dim = joint_act_dim
        self.num_contacts_per_env = num_contacts_per_env
        
        # Joint configuration
        if joint_types is None:
            raise ValueError("joint_types must be provided")
        if joint_q_start is None:
            raise ValueError("joint_q_start must be provided")
        if joint_q_end is None:
            raise ValueError("joint_q_end must be provided")
        if is_angular_dof is None:
            raise ValueError("is_angular_dof must be provided")
        if is_continuous_dof is None:
            raise ValueError("is_continuous_dof must be provided")
        
        self.num_joints_per_env = len(joint_types)
        self.joint_types = np.array(joint_types)
        self.joint_q_start = np.array(joint_q_start)
        self.joint_q_end = np.array(joint_q_end)
        self.is_angular_dof = np.array(is_angular_dof)
        self.is_continuous_dof = np.array(is_continuous_dof)
        
        # Prepare model_inputs dict (input to torch model, to be filled by process_inputs method)
        self.model_inputs = {}

        # Compute state embedding dimension
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            self.state_embedding_dim = self.state_dim
        elif self.states_embedding_type == "sinusoidal":
            self.state_embedding_dim = self.state_dim + self.is_angular_dof.sum()
        else:
            raise NotImplementedError(f"Unknown states_embedding_type: {self.states_embedding_type}")
    
    def reset(self):
        """Reset the history buffer (call at start of new trajectory)."""
        self.states_history.clear()

    def predict(
        self,
        step
    ) -> torch.Tensor:
        """
        Predict next robot state.
        
        Args:
            states: Current states (num_envs, state_dim) in generalized coordinates [joint_q, joint_qd]
            joint_acts: Joint actions/torques (num_envs, joint_act_dim)
            root_body_q: Root body pose (num_envs, 7) [x, y, z, qx, qy, qz, qw]
            contacts: Dictionary with contact information:
                - 'contact_normals': (num_envs, num_contacts * 3)
                - 'contact_depths': (num_envs, num_contacts)
                - 'contact_thicknesses': (num_envs, num_contacts)
                - 'contact_points_0': (num_envs, num_contacts * 3)
                - 'contact_points_1': (num_envs, num_contacts * 3)
            gravity_dir: Gravity direction vector (num_envs, 3)
            dt: Time step (optional, not used in current implementation)
        
        Returns:
            next_states: Next states (num_envs, state_dim)
        """

        # FIX: zeroing the output of _process_inputs once again,remove it ideally
        self.model_inputs["contact_normals"] = torch.zeros_like(self.model_inputs["contact_normals"])
        self.model_inputs["contact_depths"] = torch.zeros_like(self.model_inputs["contact_depths"])
        self.model_inputs["contact_thicknesses"] = torch.zeros_like(self.model_inputs["contact_thicknesses"])
        self.model_inputs["contact_points_0"] = torch.zeros_like(self.model_inputs["contact_points_0"])
        self.model_inputs["contact_points_1"] = torch.zeros_like(self.model_inputs["contact_points_1"])
        self.model_inputs["contact_masks"] = torch.zeros_like(self.model_inputs["contact_masks"])

        model_inputs_csv_filename = Path(__file__).parent / 'pendulum_model_inputs.csv'
        write_model_inputs_to_csv(model_inputs_csv_filename, step, self.model_inputs)

        # Run model inference
        with torch.no_grad():
            prediction = self.model.evaluate(self.model_inputs)  # (num_envs, 1, pred_dim)
            # Take prediction from last timestep
            if prediction.shape[1] > 1:
                prediction = prediction[:, -1, :]  # (num_envs, pred_dim)
            else:
                prediction = prediction.squeeze(1)  # (num_envs, pred_dim)
        
        # Convert prediction to next states
        cur_states = self.model_inputs["states"][:, -1, :]  # (num_envs, state_dim)
        next_states = convert_prediction_to_next_states(
            cur_states,
            prediction,
            self.prediction_type,
            self.orientation_prediction_parameterization,
            self.dof_q_per_env,
            self.dof_qd_per_env,
            self.num_joints_per_env,
            self.joint_q_start,
            self.joint_q_end,
            self.joint_types
        )
        
        # Convert back to world frame if needed
        next_states = convert_states_back_to_world(
            self.model_inputs["root_body_q"],
            next_states,
            self.states_frame,
            self.anchor_frame_step,
            self.state_dim,
            self.dof_q_per_env,
            self.joint_types
        )
        
        # Wrap continuous DOFs
        wrap2PI(next_states, self.is_continuous_dof)
        
        return next_states
    
    def process_inputs(
        self,
        states: torch.Tensor,
        joint_acts: torch.Tensor,
        root_body_q: torch.Tensor,
        contacts: Dict[str, torch.Tensor],
        gravity_dir: torch.Tensor,
        dt: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process model inputs: coordinate frame conversion, state embedding, contact masking.
        """
        
        # Reset model_inputs dict
        self.model_inputs = {}

        # Ensure tensors are on correct device
        states = states.to(self.device)
        joint_acts = joint_acts.to(self.device)
        root_body_q = root_body_q.to(self.device)
        gravity_dir = gravity_dir.to(self.device)
        for key in contacts:
            contacts[key] = contacts[key].to(self.device)
        
        # Compute contact masks
        contact_masks = get_contact_masks(
            contacts['contact_depths'],
            contacts['contact_thicknesses']
        )

        # Add current state to history BEFORE prediction - TO-DO FIX contacts for zero
        history_entry = {
            "root_body_q": root_body_q.clone(),
            "states": states.clone(),
            "joint_acts": joint_acts.clone(),
            "gravity_dir": gravity_dir.clone(),
            "contact_normals": torch.zeros_like(contacts['contact_normals']), #contacts['contact_normals'].clone(),
            "contact_depths": torch.zeros_like(contacts['contact_depths']), #contacts['contact_depths'].clone(),
            "contact_thicknesses": torch.zeros_like(contacts['contact_thicknesses']), #contacts['contact_thicknesses'].clone(),
            "contact_points_0": torch.zeros_like(contacts['contact_points_0']), #contacts['contact_points_0'].clone(),
            "contact_points_1": torch.zeros_like(contacts['contact_points_1']), #contacts['contact_points_1'].clone(),
            "contact_masks": torch.zeros_like(contact_masks) #contact_masks.clone(),
        }
        self.states_history.append(history_entry)

        # Assemble model inputs
        for key in self.states_history[0].keys():       # pick the first in the queue
            stacked = torch.stack([entry[key] for entry in self.states_history], dim=1)
            self.model_inputs[key] = stacked  # (num_envs, T, dim)

        # Convert coordinate frame
        (
            self.model_inputs["states"],
            _,
            self.model_inputs["contact_points_1"],
            self.model_inputs["contact_normals"],
            self.model_inputs["gravity_dir"]
        ) = convert_coordinate_frame(
            self.model_inputs["root_body_q"],
            self.model_inputs["states"],
            None,  # next_states
            self.model_inputs["contact_points_1"],
            self.model_inputs["contact_normals"],
            self.model_inputs["gravity_dir"],
            self.states_frame,
            self.anchor_frame_step,
            self.state_dim,
            self.dof_q_per_env,
            self.joint_types,
            self.num_contacts_per_env
        )
        
        # Wrap continuous DOFs
        # Reshape states to (num_envs * T, state_dim) for wrapping
        B, T, D = self.model_inputs["states"].shape
        states_flat = self.model_inputs["states"].view(B * T, D)
        wrap2PI(states_flat, self.is_continuous_dof)
        self.model_inputs["states"] = states_flat.view(B, T, D)
        
        # State embedding
        states_embedding = embed_states(
            self.model_inputs["states"],
            self.states_embedding_type,
            self.state_embedding_dim,
            self.is_angular_dof,
            self.dof_q_per_env
        )
        self.model_inputs["states_embedding"] = states_embedding
        
        # Apply contact mask
        contact_masks = self.model_inputs["contact_masks"]  # (num_envs, T, num_contacts)
        for key in self.model_inputs.keys():
            if key.startswith('contact_') and key != 'contact_masks':
                # Reshape to (num_envs, T, num_contacts, dim_per_contact)
                if key in ['contact_depths', 'contact_thicknesses']:
                    dim_per_contact = 1
                    original_shape = self.model_inputs[key].shape
                    reshaped = self.model_inputs[key].view(
                        original_shape[0], original_shape[1], self.num_contacts_per_env, dim_per_contact
                    )
                    masked = torch.where(
                        contact_masks.unsqueeze(-1) < 1e-5,
                        0.,
                        reshaped
                    )
                    self.model_inputs[key] = masked.view(original_shape)
                else:  # contact_normals, contact_points_0, contact_points_1
                    dim_per_contact = 3
                    original_shape = self.model_inputs[key].shape
                    reshaped = self.model_inputs[key].view(
                        original_shape[0], original_shape[1], self.num_contacts_per_env, dim_per_contact
                    )
                    masked = torch.where(
                        contact_masks.unsqueeze(-1) < 1e-5,
                        0.,
                        reshaped
                    )
                    self.model_inputs[key] = masked.view(original_shape)
        
        return self.model_inputs

