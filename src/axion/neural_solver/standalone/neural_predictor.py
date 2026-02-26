"""
Standalone Neural Predictor

This module provides a simplified interface for using pretrained neural models
as black-box robot dynamics predictors.
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import deque
try:
    from src.axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        convert_prediction_to_next_states_orientation_dofs,
        convert_prediction_to_next_states_regular_dofs
    )
    from src.axion.neural_solver.utils import torch_utils
except ModuleNotFoundError:
    from axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        convert_prediction_to_next_states_orientation_dofs,
        convert_prediction_to_next_states_regular_dofs
    )
    from axion.neural_solver.utils import torch_utils

from newton import JointType
JOINT_FREE = JointType.FREE
JOINT_BALL = JointType.BALL
JOINT_REVOLUTE = JointType.REVOLUTE
JOINT_PRISMATIC = JointType.PRISMATIC
JOINT_FIXED = JointType.FIXED
JOINT_DISTANCE = JointType.DISTANCE

class NeuralPredictor:
    """
    Standalone neural model predictor of dynamics.
    
    This class provides a simplified interface for using pretrained neural models
    to predict next robot states given current states, actions, contacts, and gravity.
    """
    
    def __init__(
        self,
        nn_model: torch.nn.Module,
        cfg: dict,
        device: str = 'cuda:0',
        # Robot-specific configuration
        dof_q_per_env: int = None,
        dof_qd_per_env: int = None,
        joint_types: list = None,
        joint_q_start: list = None,
        joint_q_end: list = None,
        is_angular_dof: list = None,
        is_continuous_dof: list = None,
    ):
        """
        Initialize NeRD predictor.
        
        Args:
            nn_model: Pretrained NeRD model (loaded via torch.load)
            cfg: Configuration dictionary from cfg.yaml
            device: Device to run on ('cuda:0', 'cpu', etc.)
            dof_q_per_env: Number of position DOFs per environment
            dof_qd_per_env: Number of velocity DOFs per environment
            joint_types: List of joint types (0=FREE, 1=BALL, 2=REVOLUTE, etc.)
            joint_q_start: Start indices for each joint in q vector
            joint_q_end: End indices for each joint in q vector
            is_angular_dof: Boolean array indicating which DOFs are angular
            is_continuous_dof: Boolean array indicating which DOFs are continuous (unwrapped)
        """
        self.device = device
        self.nn_model = nn_model
        self.nn_model.to(device)
        self.nn_model.eval()
        
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
        
        self.dof_q_per_env = dof_q_per_env
        self.dof_qd_per_env = dof_qd_per_env
        self.state_dim = dof_q_per_env + dof_qd_per_env
        
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
        self.nn_model_inputs = {}

        # Compute state embedding dimension
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            self.state_embedding_dim = self.state_dim
        else:
            raise NotImplementedError(f"Unknown states_embedding_type: {self.states_embedding_type}")
    
    def reset(self):
        """Reset the history buffer (call at start of new trajectory)."""
        self.states_history.clear()

    def process_inputs(
        self,
        states: torch.Tensor,
        root_body_q: torch.Tensor,
        gravity_dir: torch.Tensor,
        dt: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process model inputs: coordinate frame conversion, state embedding.
        """
        
        # Reset model_inputs dict
        self.nn_model_inputs = {}

        # Ensure tensors are on correct device
        states = states.to(self.device)
        root_body_q = root_body_q.to(self.device)
        gravity_dir = gravity_dir.to(self.device)

        # Add current state to history BEFORE prediction
        history_entry = {
            "root_body_q": root_body_q.clone(),
            "states": states.clone(),
            "gravity_dir": gravity_dir.clone(),
        }
        self.states_history.append(history_entry)

        # Assemble model inputs
        for key in self.states_history[0].keys():       # pick the first in the queue
            stacked = torch.stack([entry[key] for entry in self.states_history], dim=1)
            self.nn_model_inputs[key] = stacked  # (num_envs, T, dim)
        
        # Wrap continuous DOFs
        # Reshape states to (num_envs * T, state_dim) for wrapping
        B, T, D = self.nn_model_inputs["states"].shape
        states_flat = self.nn_model_inputs["states"].view(B * T, D)
        wrap2PI(states_flat, self.is_continuous_dof)
        self.nn_model_inputs["states"] = states_flat.view(B, T, D)
        
        # State embedding
        states_embedding = self._embed_states(self.nn_model_inputs["states"])
        self.nn_model_inputs["states_embedding"] = states_embedding
        
        return self.nn_model_inputs
    
    def predict(self) -> torch.Tensor:
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

        # Run model inference
        with torch.no_grad():
            prediction = self.nn_model.evaluate(self.nn_model_inputs)  # (num_envs, 1, pred_dim)
            # Take prediction from last timestep
            if prediction.shape[1] > 1:
                prediction = prediction[:, -1, :]  # (num_envs, pred_dim)
            else:
                prediction = prediction.squeeze(1)  # (num_envs, pred_dim)
        
        # Convert prediction to next states
        cur_states = self.nn_model_inputs["states"][:, -1, :]  # (num_envs, state_dim)
        next_states = self._convert_prediction_to_next_states(cur_states, prediction)
        
        # Convert back to world frame if needed
        print("After _convert_prediction_to_next_states", next_states)
        # next_states = self._convert_states_back_to_world(
        #     self.nn_model_inputs["root_body_q"],
        #     next_states
        # )
        # print("After _convert_states_back_to_world", next_states)
        
        # Wrap continuous DOFs
        wrap2PI(next_states, self.is_continuous_dof)
        
        return next_states
    
    def _embed_states(self, states):
        """
        Embed states into a new representation.
        
        Args:
            states: (..., state_dim)
        
        Returns:
            states_embedding: (..., state_embedding_dim)
        """
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            return states.clone()
        else:
            raise NotImplementedError

    def _convert_prediction_to_next_states(self, states, prediction):
        """
        Convert model prediction to next states.

        Args:
            states: (num_envs, state_dim)
            prediction: (num_envs, pred_dim)

        Returns:
            next_states: (num_envs, state_dim)
        """
        next_states = torch.empty_like(states)

        if self.prediction_type in ["absolute", "relative"]:
            prediction_dof_offset = 0

            # Compute position components of the next states for each joint individually
            for joint_id in range(self.num_joints_per_env):
                joint_dof_start = self.joint_q_start[joint_id]
                if self.joint_types[joint_id] == JOINT_FREE:
                    # position dofs
                    prediction_dof_offset += convert_prediction_to_next_states_regular_dofs(
                        states[..., joint_dof_start:joint_dof_start + 3],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start:joint_dof_start + 3],
                        self.prediction_type
                    )
                    # 3d orientation dofs
                    prediction_dof_offset += convert_prediction_to_next_states_orientation_dofs(
                        states[..., joint_dof_start + 3:joint_dof_start + 7],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start + 3:joint_dof_start + 7],
                        self.prediction_type,
                        self.orientation_prediction_parameterization
                    )
                elif self.joint_types[joint_id] == JOINT_BALL:
                    prediction_dof_offset += convert_prediction_to_next_states_orientation_dofs(
                        states[..., joint_dof_start:joint_dof_start + 4],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start:joint_dof_start + 4],
                        self.prediction_type,
                        self.orientation_prediction_parameterization
                    )
                else:
                    joint_dof_end = self.joint_q_end[joint_id]
                    prediction_dof_offset += convert_prediction_to_next_states_regular_dofs(
                        states[..., joint_dof_start:joint_dof_end],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start:joint_dof_end],
                        self.prediction_type
                    )

            # Compute velocity components of the next states
            if self.prediction_type == "absolute":
                next_states[..., self.dof_q_per_env:].copy_(
                    prediction[..., prediction_dof_offset:]
                )
            elif self.prediction_type == "relative":
                next_states[..., self.dof_q_per_env:] = (
                    states[..., self.dof_q_per_env:]
                    + prediction[..., prediction_dof_offset:]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return next_states

    def _convert_states_back_to_world(self, root_body_q, states):
        """
        Convert states from body frame back to world frame.

        Args:
            root_body_q: (B, T, 7)
            states: (B, dof_states)

        Returns:
            states_world: (B, dof_states)
        """
        if self.states_frame == "world":
            return states
        elif self.states_frame == "body" or self.states_frame == "body_translation_only":
            if self.anchor_frame_step == "first":
                anchor_step = 0
            elif self.anchor_frame_step == "last" or self.anchor_frame_step == "every":
                anchor_step = -1
            else:
                raise NotImplementedError

            shape = states.shape

            anchor_frame_q = root_body_q[:, anchor_step, :]

            anchor_frame_pos = anchor_frame_q[:, :3]
            if self.states_frame == "body":
                anchor_frame_quat = anchor_frame_q[:, 3:7]
            elif self.states_frame == "body_translation_only":
                anchor_frame_quat = torch.zeros_like(anchor_frame_q[:, 3:7])
                anchor_frame_quat[:, 3] = 1.

            assert states.shape[0] == anchor_frame_q.shape[0]
            states_world = states.clone()
            # only need to convert the states of the first joint in the articulation
            if len(self.joint_types) > 0 and self.joint_types[0] == JOINT_FREE:
                (
                    states_world[:, 0:3],
                    states_world[:, 3:7],
                    states_world[:, self.dof_q_per_env:self.dof_q_per_env + 3],
                    states_world[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6]
                ) = torch_utils.convert_states_b2w(
                    anchor_frame_pos,
                    anchor_frame_quat,
                    p=states[:, 0:3],
                    quat=states[:, 3:7],
                    omega=states[:, self.dof_q_per_env:self.dof_q_per_env + 3],
                    nu=states[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6]
                )
            return states_world.view(*shape)
        else:
            raise NotImplementedError

    