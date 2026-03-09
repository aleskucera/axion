
import numpy as np
import torch
try:
    from src.axion.neural_solver.utils import torch_utils
except ModuleNotFoundError:
    from axion.neural_solver.utils import torch_utils

from newton import JointType
JOINT_FREE = JointType.FREE

CONTACT_DEPTH_UPPER_RATIO = 4
MIN_CONTACT_EVENT_THRESHOLD = 0.12

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

def convert_contacts_w2b_batched(
    root_body_q,
    contact_points_1,
    contact_normals,
    translation_only,
    com_to_pivot_offset,
):
    """
    Convert contacts from world to body frame for batched multi-world data.
    Express points in a frame anchored at the root joint pivot (same orientation
    as the first link, origin at the pivot) by subtracting the pivot position in body frame.

    Args:
        root_body_q: (num_worlds, 7) - root body pose per world
        contact_points_1: (num_worlds, num_contacts, 3) - ground contact points in world frame
        contact_normals: (num_worlds, num_contacts, 3) - contact normals in world frame
        translation_only: bool
        com_to_pivot_offset: (3,) or (1, 1, 3) tensor - position of the root joint
            pivot in the first link's COM frame (body frame).
    Returns:
        contact_points_1_body: (num_worlds, num_contacts, 3) in body or pivot frame
        contact_normals_body: (num_worlds, num_contacts, 3)
    """
    num_worlds, num_contacts, _ = contact_points_1.shape
    body_q_expanded = root_body_q.unsqueeze(1).expand(num_worlds, num_contacts, 7)

    body_q_flat = body_q_expanded.reshape(-1, 7)
    points_flat = contact_points_1.reshape(-1, 3)
    normals_flat = contact_normals.reshape(-1, 3)

    body_frame_pos = body_q_flat[:, :3]
    if translation_only:
        body_frame_quat = torch.zeros_like(body_q_flat[:, 3:7])
        body_frame_quat[:, 3] = 1.0
    else:
        body_frame_quat = body_q_flat[:, 3:7]

    contact_points_1_body = torch_utils.transform_point_inverse(
        body_frame_pos, body_frame_quat, points_flat
    ).view(num_worlds, num_contacts, 3)

    if translation_only:
        contact_normals_body = contact_normals.clone()
    else:
        contact_normals_body = torch_utils.quat_rotate_inverse(
            body_frame_quat, normals_flat
        ).view(num_worlds, num_contacts, 3)

    # Express points in pivot frame (origin at root joint pivot, same orientation as body)
    offset = com_to_pivot_offset.view(1, 1, 3).to(
        device=contact_points_1_body.device, 
        dtype=contact_points_1_body.dtype
    )
    contact_points_1_body = contact_points_1_body - offset

    return contact_points_1_body, contact_normals_body


def apply_contact_mask(contacts, contact_masks):
    """
    Zero out inactive contacts using the contact mask.
    Args:
        contacts: dict with tensors shaped (num_worlds, num_contacts, ...).
                  Keys starting with 'contact_' are masked.
        contact_masks: (num_worlds, num_contacts) boolean tensor, True = active.
    Returns:
        contacts dict with inactive contact entries zeroed out.
    """
    for key in contacts:
        if not key.startswith('contact_') or key == 'contact_masks':
            continue
        val = contacts[key]
        mask = contact_masks
        while mask.ndim < val.ndim:
            mask = mask.unsqueeze(-1)
        contacts[key] = torch.where(mask, val, torch.zeros_like(val))
    return contacts

def convert_contacts_w2b(root_body_q, contact_points_1, contact_normals, translation_only):
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


def convert_states_w2b(root_body_q, states, state_dim, dof_q_per_env, joint_types, translation_only):
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


def convert_gravity_w2b(root_body_q, gravity_dir, translation_only):
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


def convert_gravity_w2b_batched(root_body_q, gravity_dir):
    """
    Convert gravity direction from world to body frame for batched (num_worlds, dim) inputs.
    Always applies full rotation (no translation_only option).
    Args:
        root_body_q: (num_worlds, 7) - root body pose per world
        gravity_dir: (num_worlds, 3) - gravity direction in world frame (modified in-place or use return value)
    Returns:
        gravity_dir_body: (num_worlds, 3) - gravity direction in body frame
    """
    body_frame_quat = root_body_q[:, 3:7]
    return torch_utils.quat_rotate_inverse(body_frame_quat, gravity_dir)


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
            convert_contacts_w2b(
                anchor_frame_body_q.view(B, T, 1, 7).expand(
                    B, T, num_contacts_per_env, 7
                ), 
                contact_points_1, 
                contact_normals,
                translation_only = (states_frame == "body_translation_only")
            )
        
        # convert states
        states_body = convert_states_w2b(
            anchor_frame_body_q, 
            states,
            state_dim,
            dof_q_per_env,
            joint_types,
            translation_only = (states_frame == "body_translation_only")
        )
        if next_states is not None:
            next_states_body = convert_states_w2b(
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
        gravity_dir_body = convert_gravity_w2b(
            anchor_frame_body_q, 
            gravity_dir,
            translation_only = (states_frame == "body_translation_only")
        )

        return states_body, next_states_body, contact_points_1_body, contact_normals_body, gravity_dir_body
    else:
        raise NotImplementedError