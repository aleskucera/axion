
import numpy as np
import torch
try:
    from src.axion.neural_solver.utils import torch_utils
except ModuleNotFoundError:
    from axion.neural_solver.utils import torch_utils

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
