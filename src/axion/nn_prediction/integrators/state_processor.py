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
State processing utilities extracted from NeuralIntegrator.
These functions handle coordinate frame conversions, state wrapping, 
contact processing, and prediction-to-state conversions.
"""

import numpy as np
import torch
from ..utils import torch_utils
from ..utils.commons import CONTACT_DEPTH_UPPER_RATIO, MIN_CONTACT_EVENT_THRESHOLD

# Try to import joint type constants from warp (if available at top level)
# TO-DO: this older enum does not agree with the new newton jointType enum, fix! 
# https://newton-physics.github.io/newton/api/_generated/newton.JointType.html#newton.JointType
# Otherwise use our own constants matching Warp's values
try:
    import warp as wp
    # Check if constants are available at warp level (not warp.sim)
    if hasattr(wp, 'JOINT_FREE'):
        JOINT_FREE = wp.JOINT_FREE
        JOINT_BALL = wp.JOINT_BALL
        JOINT_REVOLUTE = wp.JOINT_REVOLUTE
        JOINT_PRISMATIC = wp.JOINT_PRISMATIC
        JOINT_FIXED = wp.JOINT_FIXED
        JOINT_DISTANCE = wp.JOINT_DISTANCE
    else:
        # Fallback to numeric constants (matching Warp's joint types)
        JOINT_FREE = 0
        JOINT_BALL = 1
        JOINT_REVOLUTE = 2
        JOINT_PRISMATIC = 3
        JOINT_FIXED = 4
        JOINT_DISTANCE = 5
except ImportError:
    # Warp not available, use numeric constants
    JOINT_FREE = 0
    JOINT_BALL = 1
    JOINT_REVOLUTE = 2
    JOINT_PRISMATIC = 3
    JOINT_FIXED = 4
    JOINT_DISTANCE = 5


def get_contact_masks(contact_depths, contact_thickness):
    """
    Compute contact event masks.
    
    Args:
        contact_depths: (num_envs, (T), num_contacts_per_env)
        contact_thickness: (num_envs, (T), num_contacts_per_env)
    
    Returns:
        contact_masks: (num_envs, (T), num_contacts_per_env)
    """
    contact_event_threshold = CONTACT_DEPTH_UPPER_RATIO * contact_thickness
    contact_event_threshold = torch.where(
        contact_event_threshold < MIN_CONTACT_EVENT_THRESHOLD,
        MIN_CONTACT_EVENT_THRESHOLD,
        contact_event_threshold
    )
    
    contact_masks = (contact_depths < contact_event_threshold)
    
    return contact_masks


def wrap2PI(states, is_continuous_dof):
    """
    Fix continuous angular dofs in the states vector (in-place operation).
    
    Args:
        states: Tensor of shape (..., state_dim)
        is_continuous_dof: Boolean array of shape (state_dim,) indicating which DOFs are continuous
    """
    if not is_continuous_dof.any():
        return
    assert states.shape[-1] == is_continuous_dof.shape[0]
    wrap_delta = torch.floor(
        (states[..., is_continuous_dof] + np.pi) / (2 * np.pi)
    ) * (2 * np.pi)
    states[..., is_continuous_dof] -= wrap_delta


def _convert_contacts_w2b(root_body_q, contact_points_1, contact_normals, translation_only):
    """
    Convert contacts from world to body frame.
    
    Args:
        root_body_q: (B, T, 7) or (B, T, num_contacts, 7)
        contact_points_1: (B, T, num_contacts * 3)
        contact_normals: (B, T, num_contacts * 3)
        translation_only: bool
    
    Returns:
        contact_points_1_body: (B, T, num_contacts * 3)
        contact_normals_body: (B, T, num_contacts * 3)
    """
    shape = contact_points_1.shape
    root_body_q = root_body_q.reshape(-1, 7)
    contact_points_1 = contact_points_1.reshape(-1, 3)
    contact_normals = contact_normals.reshape(-1, 3)

    body_frame_pos = root_body_q[:, :3]
    if translation_only:
        body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
        body_frame_quat[:, 3] = 1.
    else:
        body_frame_quat = root_body_q[:, 3:7]

    assert contact_points_1.shape[0] == root_body_q.shape[0]
    contact_points_1_body = torch_utils.transform_point_inverse(
        body_frame_pos, body_frame_quat, contact_points_1).view(*shape)
    
    assert contact_normals.shape[0] == root_body_q.shape[0]
    if translation_only:
        contact_normals_body = contact_normals.view(*shape)
    else:
        contact_normals_body = torch_utils.quat_rotate_inverse(
            body_frame_quat, contact_normals).view(*shape)        
    
    return contact_points_1_body, contact_normals_body


def _convert_states_w2b(root_body_q, states, state_dim, dof_q_per_env, joint_types, translation_only):
    """
    Convert states from world frame to body frame.
    
    Args:
        root_body_q: (B, T, 7)
        states: (B, T, dof_states)
        state_dim: int
        dof_q_per_env: int
        joint_types: array of joint types
        translation_only: bool
    
    Returns:
        states_body: (B, T, dof_states)
    """
    shape = states.shape
    root_body_q = root_body_q.reshape(-1, 7)
    states = states.reshape(-1, state_dim)

    body_frame_pos = root_body_q[:, :3]
    if translation_only:
        body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
        body_frame_quat[:, 3] = 1.
    else:
        body_frame_quat = root_body_q[:, 3:7]

    assert states.shape[0] == root_body_q.shape[0]
    states_body = states.clone()
    if len(joint_types) > 0 and joint_types[0] == JOINT_FREE:
        (
            states_body[:, 0:3], 
            states_body[:, 3:7], 
            states_body[:, dof_q_per_env:dof_q_per_env + 3], 
            states_body[:, dof_q_per_env + 3:dof_q_per_env + 6]
        ) = torch_utils.convert_states_w2b(
                body_frame_pos,
                body_frame_quat,
                p = states[:, 0:3],
                quat = states[:, 3:7],
                omega = states[:, dof_q_per_env:dof_q_per_env + 3],
                nu = states[:, dof_q_per_env + 3:dof_q_per_env + 6]
            )
            
    return states_body.view(*shape)


def _convert_gravity_w2b(root_body_q, gravity_dir, translation_only):
    """
    Convert gravity direction from world to body frame.
    
    Args:
        root_body_q: (B, T, 7)
        gravity_dir: (B, T, 3)
        translation_only: bool
    
    Returns:
        gravity_dir_body: (B, T, 3)
    """
    if translation_only:
        return gravity_dir
    
    shape = gravity_dir.shape
    root_body_q = root_body_q.reshape(-1, 7)
    gravity_dir = gravity_dir.reshape(-1, 3)

    body_frame_quat = root_body_q[:, 3:7]

    assert gravity_dir.shape[0] == body_frame_quat.shape[0]
    gravity_dir_body = torch_utils.quat_rotate_inverse(
        body_frame_quat, gravity_dir).view(*shape)    
    
    return gravity_dir_body


def convert_coordinate_frame(
    root_body_q,  # (B, T, 7)
    states,  # (B, T, dof_states)
    next_states,  # (B, T, dof_states), can be None
    contact_points_1,  # (B, T, num_contacts * 3)
    contact_normals,  # (B, T, num_contacts * 3)
    gravity_dir,  # (B, T, 3)
    states_frame,  # 'world', 'body', or 'body_translation_only'
    anchor_frame_step,  # 'first', 'last', or 'every'
    state_dim,
    dof_q_per_env,
    joint_types,
    num_contacts_per_env
):
    """
    Convert coordinate frame for states, contacts, and gravity.
    
    Returns:
        states_body, next_states_body, contact_points_1_body, contact_normals_body, gravity_dir_body
    """
    print("states.shape", states.shape)
    assert len(states.shape) == 3

    if states_frame == 'world':
        return states, next_states, contact_points_1, contact_normals, gravity_dir
    elif states_frame == 'body' or states_frame == 'body_translation_only':
        B, T = states.shape[0], states.shape[1]

        if anchor_frame_step == "first":
            anchor_frame_body_q = root_body_q[:, 0:1, :].expand(B, T, 7)
        elif anchor_frame_step == "last":
            anchor_frame_body_q = root_body_q[:, -1:, :].expand(B, T, 7)
        elif anchor_frame_step == "every":
            anchor_frame_body_q = root_body_q
        else:
            raise NotImplementedError

        # convert contacts
        contact_points_1_body, contact_normals_body = \
            _convert_contacts_w2b(
                anchor_frame_body_q.view(B, T, 1, 7).expand(
                    B, T, num_contacts_per_env, 7
                ), 
                contact_points_1, 
                contact_normals,
                translation_only = (states_frame == "body_translation_only")
            )
        
        # convert states
        states_body = _convert_states_w2b(
            anchor_frame_body_q, 
            states,
            state_dim,
            dof_q_per_env,
            joint_types,
            translation_only = (states_frame == "body_translation_only")
        )
        if next_states is not None:
            next_states_body = _convert_states_w2b(
                anchor_frame_body_q, 
                next_states,
                state_dim,
                dof_q_per_env,
                joint_types,
                translation_only = (states_frame == "body_translation_only")
            )
        else:
            next_states_body = None
        
        # convert gravity
        gravity_dir_body = _convert_gravity_w2b(
            anchor_frame_body_q, 
            gravity_dir,
            translation_only = (states_frame == "body_translation_only")
        )

        return states_body, next_states_body, contact_points_1_body, contact_normals_body, gravity_dir_body
    else:
        raise NotImplementedError


def convert_states_back_to_world(
    root_body_q,  # (B, T, 7)
    states,  # (B, dof_states)
    states_frame,  # 'world', 'body', or 'body_translation_only'
    anchor_frame_step,  # 'first', 'last', or 'every'
    state_dim,
    dof_q_per_env,
    joint_types
):
    """
    Convert states from body frame back to world frame.
    
    Args:
        root_body_q: (B, T, 7)
        states: (B, dof_states)
        states_frame: str
        anchor_frame_step: str
        state_dim: int
        dof_q_per_env: int
        joint_types: array
    
    Returns:
        states_world: (B, dof_states)
    """
    if states_frame == "world":
        return states
    elif states_frame == "body" or states_frame == "body_translation_only":
        if anchor_frame_step == "first":
            anchor_step = 0
        elif anchor_frame_step == "last" or anchor_frame_step == "every":
            anchor_step = -1
        else:
            raise NotImplementedError
        
        shape = states.shape

        anchor_frame_q = root_body_q[:, anchor_step, :]

        anchor_frame_pos = anchor_frame_q[:, :3]
        if states_frame == "body":
            anchor_frame_quat = anchor_frame_q[:, 3:7]
        elif states_frame == "body_translation_only":
            anchor_frame_quat = torch.zeros_like(anchor_frame_q[:, 3:7])
            anchor_frame_quat[:, 3] = 1.

        assert states.shape[0] == anchor_frame_q.shape[0]
        states_world = states.clone()
        # only need to convert the states of the first joint in the articulation
        if len(joint_types) > 0 and joint_types[0] == JOINT_FREE:
            (
                states_world[:, 0:3], 
                states_world[:, 3:7], 
                states_world[:, dof_q_per_env:dof_q_per_env + 3], 
                states_world[:, dof_q_per_env + 3:dof_q_per_env + 6] 
            ) = torch_utils.convert_states_b2w(
                    anchor_frame_pos,
                    anchor_frame_quat,
                    p = states[:, 0:3],
                    quat = states[:, 3:7],
                    omega = states[:, dof_q_per_env:dof_q_per_env + 3],
                    nu = states[:, dof_q_per_env + 3:dof_q_per_env + 6]
                )
        return states_world.view(*shape)
    else:
        raise NotImplementedError


def embed_states(states, states_embedding_type, state_embedding_dim, is_angular_dof, dof_q_per_env):
    """
    Embed states into a new representation.
    
    Args:
        states: (..., state_dim)
        states_embedding_type: None, "identical", or "sinusoidal"
        state_embedding_dim: int
        is_angular_dof: boolean array
        dof_q_per_env: int
    
    Returns:
        states_embedding: (..., state_embedding_dim)
    """
    if states_embedding_type is None or states_embedding_type == "identical":
        return states.clone()
    elif states_embedding_type == "sinusoidal":
        states_embedding = torch.zeros(
            (*states.shape[:-1], state_embedding_dim), 
            device = states.device
        )
        idx = 0
        for dof_idx in range(len(is_angular_dof)):
            if not is_angular_dof[dof_idx]:
                states_embedding[..., idx] = states[..., dof_idx].clone()
                idx += 1
            else:
                states_embedding[..., idx] = torch.sin(states[..., dof_idx])
                states_embedding[..., idx + 1] = torch.cos(states[..., dof_idx])
                idx += 2
        states_embedding[..., idx:] = states[..., dof_q_per_env :].clone()
        return states_embedding
    else:
        raise NotImplementedError


def convert_prediction_to_next_states_regular_dofs(states, prediction, next_states, prediction_type):
    """
    Convert prediction to next states for regular DOFs.
    
    Args:
        states: (..., dofs)
        prediction: (..., pred_dims)
        next_states: (..., dofs) - output tensor
        prediction_type: 'absolute' or 'relative'
    
    Returns:
        Number of DOFs processed
    """
    assert states.shape[-1] == next_states.shape[-1]
    dofs = states.shape[-1]
    if prediction_type == 'absolute':
        next_states.copy_(prediction[..., :dofs])
        return dofs
    elif prediction_type == 'relative':
        next_states.copy_(states + prediction[..., :dofs])
        return dofs
    else:
        raise NotImplementedError


def convert_prediction_to_next_states_orientation_dofs(
    states, prediction, next_states, prediction_type, orientation_prediction_parameterization
):
    """
    Convert prediction to next states for orientation DOFs (quaternions).
    
    Args:
        states: (..., 4) - current quaternion
        prediction: (..., pred_dims)
        next_states: (..., 4) - output tensor
        prediction_type: 'absolute' or 'relative'
        orientation_prediction_parameterization: 'quaternion', 'exponential', or 'naive'
    
    Returns:
        Number of prediction DOFs used
    """
    assert states.shape[-1] == 4 and next_states.shape[-1] == 4

    # Parse the prediction into quaternion
    if orientation_prediction_parameterization == 'naive':
        predicted_quaternion = prediction[..., :4]
        prediction_dofs = 4
    elif orientation_prediction_parameterization == 'quaternion':
        predicted_quaternion = prediction[..., :4]
        predicted_quaternion = torch_utils.normalize(predicted_quaternion)
        prediction_dofs = 4
    elif orientation_prediction_parameterization == 'exponential':
        predicted_quaternion = torch_utils.exponential_coord_to_quat(prediction[..., :3])
        prediction_dofs = 3
    else:
        raise NotImplementedError
    
    # Apply quaternion/delta quaternion to the states to acquire next_states
    if prediction_type == 'absolute':
        raw_next_quaternion = predicted_quaternion
    elif prediction_type == 'relative':
        if orientation_prediction_parameterization == 'naive':
            raw_next_quaternion = states + predicted_quaternion
        else:
            raw_next_quaternion = torch_utils.quat_mul(predicted_quaternion, states)
    else:
        raise NotImplementedError
    
    # Normalize the next_states quaternion
    next_states.copy_(torch_utils.normalize(raw_next_quaternion))

    return prediction_dofs


def convert_prediction_to_next_states(
    states,  # (num_envs, state_dim)
    prediction,  # (num_envs, pred_dim)
    prediction_type,  # 'absolute' or 'relative'
    orientation_prediction_parameterization,  # 'quaternion', 'exponential', or 'naive'
    dof_q_per_env,
    dof_qd_per_env,
    num_joints_per_env,
    joint_q_start,  # array
    joint_q_end,  # array
    joint_types  # array
):
    """
    Convert model prediction to next states.
    
    Args:
        states: (num_envs, state_dim)
        prediction: (num_envs, pred_dim)
        prediction_type: str
        orientation_prediction_parameterization: str
        dof_q_per_env: int
        dof_qd_per_env: int
        num_joints_per_env: int
        joint_q_start: array
        joint_q_end: array
        joint_types: array
    
    Returns:
        next_states: (num_envs, state_dim)
    """
    next_states = torch.empty_like(states)
    
    if prediction_type in ["absolute", "relative"]:
        prediction_dof_offset = 0

        # Compute position components of the next states for each joint individually
        for joint_id in range(num_joints_per_env):
            joint_dof_start = joint_q_start[joint_id]
            if joint_types[joint_id] == JOINT_FREE:
                # position dofs
                prediction_dof_offset += convert_prediction_to_next_states_regular_dofs(
                    states[..., joint_dof_start:joint_dof_start + 3],
                    prediction[..., prediction_dof_offset:],
                    next_states[..., joint_dof_start:joint_dof_start + 3],
                    prediction_type
                )
                # 3d orientation dofs
                prediction_dof_offset += convert_prediction_to_next_states_orientation_dofs(
                    states[..., joint_dof_start + 3:joint_dof_start + 7],
                    prediction[..., prediction_dof_offset:],
                    next_states[..., joint_dof_start + 3:joint_dof_start + 7],
                    prediction_type,
                    orientation_prediction_parameterization
                )
            elif joint_types[joint_id] == JOINT_BALL:
                prediction_dof_offset += convert_prediction_to_next_states_orientation_dofs(
                    states[..., joint_dof_start:joint_dof_start + 4],
                    prediction[..., prediction_dof_offset:],
                    next_states[..., joint_dof_start:joint_dof_start + 4],
                    prediction_type,
                    orientation_prediction_parameterization
                )
            else:
                joint_dof_end = joint_q_end[joint_id]
                prediction_dof_offset += convert_prediction_to_next_states_regular_dofs(
                    states[..., joint_dof_start:joint_dof_end],
                    prediction[..., prediction_dof_offset:],
                    next_states[..., joint_dof_start:joint_dof_end],
                    prediction_type
                )
                
        # Compute velocity components of the next states
        if prediction_type == "absolute":
            next_states[..., dof_q_per_env:].copy_(
                prediction[..., prediction_dof_offset:]
            )
        elif prediction_type == "relative":
            next_states[..., dof_q_per_env:] = (
                states[..., dof_q_per_env:] + 
                prediction[..., prediction_dof_offset:]
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return next_states

