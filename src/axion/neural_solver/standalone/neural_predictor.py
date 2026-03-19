"""
Standalone Neural Predictor

This module provides a simplified interface for using pretrained neural models
as black-box robot dynamics predictors.
"""

import numpy as np
import torch
import warp as wp
import newton
from typing import Dict, Optional
from collections import deque

try:
    from src.axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        convert_prediction_to_next_states_orientation_dofs,
        convert_prediction_to_next_states_regular_dofs,
        get_contact_masks,
        convert_contacts_w2b_batched,
        apply_contact_mask,
        convert_gravity_w2b_batched,
    )
    from src.axion.neural_solver.utils import torch_utils
    from src.axion.types import reorder_ground_contacts_kernel, contact_penetration_depth_kernel
    from src.axion.core.contacts import AxionContacts
except ModuleNotFoundError:
    from axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        convert_prediction_to_next_states_orientation_dofs,
        convert_prediction_to_next_states_regular_dofs,
        get_contact_masks,
        convert_contacts_w2b_batched,
        apply_contact_mask,
        convert_gravity_w2b_batched,
    )
    from axion.neural_solver.utils import torch_utils
    from axion.types import reorder_ground_contacts_kernel, contact_penetration_depth_kernel
    from axion.core.contacts import AxionContacts


JOINT_FREE = newton.JointType.FREE
JOINT_BALL = newton.JointType.BALL
JOINT_REVOLUTE = newton.JointType.REVOLUTE
JOINT_PRISMATIC = newton.JointType.PRISMATIC
JOINT_FIXED = newton.JointType.FIXED
JOINT_DISTANCE = newton.JointType.DISTANCE

PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL = 4

class NeuralPredictor:
    """
    Standalone neural model predictor of dynamics.
    
    This class provides a simplified interface for using pretrained neural models
    to predict next robot states given current states, actions, contacts, and gravity.
    """
    
    def __init__(
        self,
        newton_model: newton.Model,
        nn_model: torch.nn.Module,
        nn_cfg: dict,
        device: str = 'cuda:0',
        # Robot-specific configuration that would be too cumbersome to infer
        joint_q_end=[1, 2],                 # Joint DOF end indices in q vector
        is_angular_dof=[True, True, True, True],      # Which DOFs are angular (for state embedding)
        is_continuous_dof=[True, True, False, False]  # Which DOFs are continuous (unwrapped angles) Position DOFs (angles) are continuous, velocities are not
    ):
        """
        Initialize NeRD predictor.

        Robot configuration (dofs, joint types, q start/end, angular/continuous flags)
        is inferred from newton_model.

        Args:
            newton_model: Newton physics model (robot + scene); used to infer DOFs and joint layout.
            nn_model: Pretrained NeRD model (loaded via torch.load).
            nn_cfg: Configuration dictionary from cfg.yaml.
            device: Device to run on ('cuda:0', 'cpu', etc.).
        """
        self.device = device
    
        # Robot model reference (used for contacts and body count)
        self.robot_model = newton_model
        self.num_worlds = 1
        self.dof_q_per_env = int(newton_model.joint_coord_count) // self.num_worlds
        self.dof_qd_per_env = int(newton_model.joint_dof_count) // self.num_worlds
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.num_joints_per_env = int(newton_model.joint_count) // self.num_worlds
        self.bodies_per_world = int(newton_model.body_count) // self.num_worlds
        joint_type_np = newton_model.joint_type.numpy()
        joint_q_start_global = newton_model.joint_q_start.numpy()
        self.joint_types = joint_type_np[:self.num_joints_per_env].copy()
        self.joint_q_start = (joint_q_start_global[:self.num_joints_per_env] % self.dof_q_per_env).tolist()
        self.joint_q_end = joint_q_end
        self.is_angular_dof= np.array(is_angular_dof)
        self.is_continuous_dof= np.array(is_continuous_dof)
        self.gravity_vector = torch.zeros((self.num_worlds, 3), device= str(self.device))
        self.gravity_vector[:, self.robot_model.up_axis] = -1.0 # copy the gravity dir from model (should be along Z) 
        # Root joint pivot in first-link body (COM) frame: from model joint child xform (index 0 = root)
        joint_X_c = self.robot_model.joint_X_c.numpy()
        root_joint_idx = 0
        pivot_in_body = joint_X_c[root_joint_idx, :3].astype("float32")
        self._com_to_pivot_offset = torch.as_tensor(pivot_in_body, dtype=torch.float32, device=self.device)

        # NN model 
        self.nn_model = nn_model
        self.nn_model.to(device)
        self.nn_model.eval()

        # Load NN model configuration
        self.neural_integrator_cfg = nn_cfg.get('env', {}).get('neural_integrator_cfg', {})
        self.states_frame = self.neural_integrator_cfg.get('states_frame', 'body')
        self.anchor_frame_step = self.neural_integrator_cfg.get('anchor_frame_step', 'every')
        self.prediction_type = self.neural_integrator_cfg.get('prediction_type', 'relative')
        self.orientation_prediction_parameterization = self.neural_integrator_cfg.get('orientation_prediction_parameterization', 'quaternion')
        self.states_embedding_type = self.neural_integrator_cfg.get('states_embedding_type', None)
        self.num_states_history = self.neural_integrator_cfg.get('num_states_history', 1)   # history window, defaults to 1

        # state history double ended queue
        self.states_history = deque(maxlen=self.num_states_history)

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

    
    def _convert_newton_contacts_to_contacts_for_nn_model(
        self,
        state_in,
        axion_contacts,
        root_body_q: torch.Tensor,
        ):
        """
        1.  Reorder the contacts from newton such that points_0 are always on the robot body and
            points_1 are the corresponding points on the external object (the contact plane)
        2.  Calculate penetration depth (used for contact masking later)
        3.  Convert the contact data to torch tensors.
        4.  Calculate the contact mask (mask that defines active contacts)
        5.  Convert points_1 and contact normals to the body frame (robot body frame)
        6.  Apply the contact mask
        """

        # Reorder batched contacts such that points_0 are on body and points_1 are ground
        num_shapes_per_world = self.robot_model.shape_count // self.num_worlds
        shape = (self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL)
        device = str(self.device)
        reordered_point0 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_point1 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_normal = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_thickness0 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_thickness1 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_body_shape = wp.full(shape, -1, dtype=wp.int32, device=device)
        body_contact_count = wp.zeros((self.num_worlds, self.bodies_per_world), dtype=wp.int32, device=device)

        shape_body_2d = self.robot_model.shape_body.reshape((self.num_worlds, num_shapes_per_world))

        wp.launch(
            kernel=reorder_ground_contacts_kernel,
            dim=(self.num_worlds, axion_contacts.max_contacts),
            inputs=[
                axion_contacts.contact_count,
                axion_contacts.contact_shape0,
                axion_contacts.contact_shape1,
                axion_contacts.contact_point0,
                axion_contacts.contact_point1,
                axion_contacts.contact_normal,
                axion_contacts.contact_thickness0,
                axion_contacts.contact_thickness1,
                shape_body_2d,
                self.bodies_per_world,  # Newton uses global body indices; kernel converts to per-world
                body_contact_count,
            ],
            outputs=[
                reordered_point0,  # Always body
                reordered_point1,  # Always ground
                reordered_normal,
                reordered_thickness0,  # Always body
                reordered_thickness1,  # Always ground
                reordered_body_shape,  # Body shape index for each contact
            ],
            device=str(self.device)
        )

        # Calculate Penetration depth using reordered contact data
        contact_depths_wp_array = wp.zeros((self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL), dtype=wp.float32, device=str(self.device))
        body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))

        wp.launch(
            kernel=contact_penetration_depth_kernel,
            dim=(self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
            inputs=[
                body_q_2d,
                shape_body_2d,
                self.bodies_per_world,  # Newton uses global body indices; kernel converts to per-world
                reordered_point0,  # Body points (reordered)
                reordered_point1,  # Ground points (reordered)
                reordered_normal,  # Normal from body to ground (reordered)
                reordered_thickness0,  # Body thickness (reordered)
                reordered_thickness1,  # Ground thickness (reordered)
                reordered_body_shape,  # Body shape indices
            ],
            outputs=[
                contact_depths_wp_array
            ],
            device=str(self.device)
        )

        # Convert to torch — shapes: (num_worlds, num_contacts, 3) for vec3,
        # (num_worlds, num_contacts) for scalars.
        contact_depths = wp.to_torch(contact_depths_wp_array)
        contact_normals = wp.to_torch(reordered_normal)
        contact_thickness = wp.to_torch(reordered_thickness0)  # Body thickness
        contact_points_0 = wp.to_torch(reordered_point0) # Body points  
        contact_points_1 = wp.to_torch(reordered_point1)  # Ground points
        contacts = {
            "contact_normals": contact_normals,
            "contact_depths": contact_depths,
            "contact_thicknesses": contact_thickness,
            "contact_points_0": contact_points_0,
            "contact_points_1": contact_points_1
        }

        contact_masks = get_contact_masks(
            contacts['contact_depths'],
            contacts['contact_thicknesses']
        )

        # Convert contact points_1 and normals from world to body frame, then to pivot frame
        contact_points_1_body, contact_normals_body = convert_contacts_w2b_batched(
            root_body_q,
            contact_points_1,
            contact_normals,
            translation_only=False,
            com_to_pivot_offset=self._com_to_pivot_offset,
        )

        contacts["contact_points_1"] = contact_points_1_body
        contacts["contact_normals"] = contact_normals_body

        # Zero out inactive contacts
        apply_contact_mask(contacts, contact_masks)

        return contacts # processed contacts: converted to body reference frame and masked


    def _convert_gravity_vec_w2b(self, root_body_q: torch.Tensor):
        """
        Convert gravity vector via the root_body_q + additional translation transform.
        root_body_q: (num_worlds, 7)
        """
        return convert_gravity_w2b_batched(root_body_q, self.gravity_vector)

    def create_axion_contacts(self, newton_contacts):
        """
        Create AxionContacts object from Newton contacts.
        """
        axion_contacts = AxionContacts(model= self.robot_model, max_contacts_per_world= PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL)
        axion_contacts.load_contact_data(newton_contacts, self.robot_model)
        return axion_contacts

    def process_inputs(
        self,
        state_in, #newton.State,
        axion_contacts, #newton.Contacts,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Process model inputs: coordinate frame conversion, state embedding.
        """
        
        # Get min coordinate representation from newton's state
        state_min_coords = torch.cat( (wp.to_torch(state_in.joint_q), wp.to_torch(state_in.joint_qd)))
        state_min_coords = state_min_coords.unsqueeze(0)  # shape (1,4)
        states = state_min_coords.to(self.device)
        # Get root body q (q of first pendulum link) once; (num_worlds, 7)
        body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))
        body_q_torch = wp.to_torch(body_q_2d)  # (num_worlds, bodies_per_world, 7)
        root_body_q = body_q_torch[:, 0, :].to(self.device)  # (num_worlds, 7)

        # Process contacts 
        processed_contacts = self._convert_newton_contacts_to_contacts_for_nn_model(state_in, axion_contacts, root_body_q)

        # Convert gravity
        gravity_in_body = self._convert_gravity_vec_w2b(root_body_q)

        # Add current state to history BEFORE prediction
        history_entry = {
            "root_body_q": root_body_q.clone(),
            "states": states.clone(),
            "gravity_dir": gravity_in_body.clone(),
            "contact_normals": processed_contacts['contact_normals'].clone(),
            "contact_depths": processed_contacts['contact_depths'].clone(), 
            "contact_points_1": processed_contacts['contact_points_1'].clone(), 
        }
        self.states_history.append(history_entry)

        # Assemble model inputs
        self.nn_model_inputs = {}   # Reset model_inputs dict
        for key in self.states_history[0].keys():       # pick the first in the queue
            stacked = torch.stack([entry[key] for entry in self.states_history], dim=1)
            self.nn_model_inputs[key] = stacked  # (num_envs, T, dim)
        
        # Flatten contact tensors to match training format: (B, T, num_contacts, 3) -> (B, T, num_contacts*3)
        B, T = self.nn_model_inputs["states"].shape[0], self.nn_model_inputs["states"].shape[1]
        self.nn_model_inputs["contact_normals"] = self.nn_model_inputs["contact_normals"].view(B, T, -1)
        self.nn_model_inputs["contact_points_1"] = self.nn_model_inputs["contact_points_1"].view(B, T, -1)
        
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
        #print("After _convert_prediction_to_next_states", next_states)
        
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

    def _apply_contact_mask(self,):
        """
        Once self.model_inputs are assembled (TODO), apply the contact mask 
        """        
        contact_masks = self.nn_model_inputs["contact_masks"]  # (num_envs, T, num_contacts)
        for key in self.nn_model_inputs.keys():
            if key.startswith('contact_') and key != 'contact_masks':
                # Reshape to (num_envs, T, num_contacts, dim_per_contact)
                if key in ['contact_depths', 'contact_thicknesses']:
                    dim_per_contact = 1
                    original_shape = self.nn_model_inputs[key].shape
                    reshaped = self.nn_model_inputs[key].view(
                        original_shape[0], original_shape[1], self.num_contacts_per_env, dim_per_contact
                    )
                    masked = torch.where(
                        contact_masks.unsqueeze(-1) < 1e-5,
                        0.,
                        reshaped
                    )
                    self.nn_model_inputs[key] = masked.view(original_shape)
                else:  # contact_normals, contact_points_0, contact_points_1
                    dim_per_contact = 3
                    original_shape = self.nn_model_inputs[key].shape
                    reshaped = self.nn_model_inputs[key].view(
                        original_shape[0], original_shape[1], self.num_contacts_per_env, dim_per_contact
                    )
                    masked = torch.where(
                        contact_masks.unsqueeze(-1) < 1e-5,
                        0.,
                        reshaped
                    )
                    self.nn_model_inputs[key] = masked.view(original_shape)