"""
Transformer-oriented neural model utilities provider (single-script, no inheritance).

Functionally identical to TransformerNeuralModelUtilsProvider but self-contained:
inlines behavior from NeuralModelUtilsProvider and StatefulNeuralModelUtilsProvider.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import warp as wp

def _ensure_bt(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, T, D). Accepts (B, D) or (B, T, D)."""
    if x is None:
        return x
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (B,D) or (B,T,D), got shape={tuple(x.shape)}")


def _wrap_to_pi_(angles: torch.Tensor) -> torch.Tensor:
    """In-place wrap of angles to [-pi, pi)."""
    two_pi = 2.0 * torch.pi
    angles.add_(torch.pi).remainder_(two_pi).sub_(torch.pi)
    return angles


@dataclass
class NeuralModelUtilsProviderCfg:
    prediction_type: str = "relative"  # "relative" or "absolute"
    states_embedding_type: Optional[str] = "identical"  # None/"identical"
    angular_q_indices: Optional[Sequence[int]] = None  # indices in q to wrap; default: all q


class TransformerNeuralModelUtilsProvider:
    """
    Transformer-oriented neural model utilities provider.

    It keeps a rolling history of state snapshots and produces (B, T, dim)
    tensors without flattening the time dimension, which matches what the
    transformer-based ModelMixedInput expects.
    """

    def __init__(
        self,
        robot_model,
        neural_model: Optional[torch.nn.Module] = None,
        *,
        num_states_history: int = 1,
        cfg: Optional[dict] = None,
        prediction_type: str = "relative",
        states_embedding_type: Optional[str] = "identical",
        angular_q_indices: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.robot_model = robot_model

        if device is not None:
            self.torch_device = torch.device(device) if isinstance(device, str) else device
        else:
            model_wp_device = getattr(self.robot_model, "device", None)
            self.torch_device = (
                wp.device_to_torch(model_wp_device) if model_wp_device is not None else torch.device("cpu")
            )

        if cfg is not None:
            prediction_type = cfg.get("prediction_type", prediction_type)
            states_embedding_type = cfg.get("states_embedding_type", states_embedding_type)
            angular_q_indices = cfg.get("angular_q_indices", angular_q_indices)

        if prediction_type not in ("relative", "absolute"):
            raise ValueError(f"prediction_type must be 'relative' or 'absolute', got {prediction_type!r}")
        if states_embedding_type not in (None, "identical"):
            raise ValueError(
                "states_embedding_type must be None or 'identical'. "
                f"Got {states_embedding_type!r}"
            )

        self.prediction_type = prediction_type
        self.states_embedding_type = states_embedding_type

        num_worlds = int(getattr(self.robot_model, "num_worlds", 1))
        self.num_envs = num_worlds

        joint_coord_count = int(getattr(self.robot_model, "joint_coord_count", 0))
        joint_dof_count = int(getattr(self.robot_model, "joint_dof_count", 0))
        if joint_coord_count <= 0 or joint_dof_count <= 0:
            raise ValueError(
                "NeuralModelUtilsProvider expected a Newton/Axion model with "
                f"joint_coord_count/joint_dof_count, got {joint_coord_count}/{joint_dof_count}."
            )

        self.dof_q_per_env = joint_coord_count // self.num_envs
        self.dof_qd_per_env = joint_dof_count // self.num_envs
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env

        self.prediction_dim = self.state_dim
        self.state_embedding_dim = self.state_dim

        if angular_q_indices is None:
            self.angular_q_indices = torch.arange(self.dof_q_per_env, device="cpu")
        else:
            self.angular_q_indices = torch.tensor(list(angular_q_indices), dtype=torch.long, device="cpu")

        self.root_body_q = torch.zeros((self.num_envs, 7), device=self.torch_device, dtype=torch.float32)
        self.gravity_dir = torch.zeros((self.num_envs, 3), device=self.torch_device, dtype=torch.float32)
        up_axis = int(getattr(self.robot_model, "up_axis", 2))
        if 0 <= up_axis < 3:
            self.gravity_dir[:, up_axis] = -1.0
        self.states = torch.zeros((self.num_envs, self.state_dim), device=self.torch_device, dtype=torch.float32)

        self.neural_model = None
        self.set_neural_model(neural_model)

        self._simulation_step = 0

        self.num_states_history = int(num_states_history)
        self.reset_states_history()

    def set_neural_model(self, neural_model: Optional[torch.nn.Module]):
        self.neural_model = neural_model
        if self.neural_model is not None:
            self.neural_model.to(self.torch_device)

    def reset(self):
        self._simulation_step = 0
        self.reset_states_history()

    def reset_states_history(self):
        # For transformer we do not pre-fill the history; it starts empty and
        # grows as the env is stepped.
        self.states_history: deque = deque(maxlen=self.num_states_history)

    def append_current_state_to_history(
        self,
        *,
        joint_acts: Optional[torch.Tensor] = None,
    ):
        """
        Append a snapshot of the current env-related tensors to the history.

        Caller (e.g. env wrapper) is expected to keep self.states, self.root_body_q,
        self.gravity_dir, and possibly joint_acts in sync with the simulator.
        """
        entry: Dict[str, torch.Tensor] = {
            "root_body_q": self.root_body_q.clone(),
            "states": self.states.clone(),
            "gravity_dir": self.gravity_dir.clone(),
        }

        if self.states_embedding_type in (None, "identical"):
            entry["states_embedding"] = entry["states"].clone()

        if joint_acts is not None:
            entry["joint_acts"] = joint_acts.clone()

        self.states_history.append(entry)

    def wrap2PI(self, states: torch.Tensor):
        """
        Wrap angular coordinates in-place.

        For pendulum states, we assume q occupies [:dof_q_per_env] and all q entries are angles.
        """
        if states is None:
            return
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"wrap2PI expected last dim {self.state_dim}, got {states.shape[-1]}")
        if self.angular_q_indices.numel() == 0:
            return
        q = states[..., : self.dof_q_per_env]
        q_sel = q.index_select(-1, self.angular_q_indices.to(q.device))
        _wrap_to_pi_(q_sel)
        q.index_copy_(-1, self.angular_q_indices.to(q.device), q_sel)

    def get_contact_masks(self, contact_depths, contact_thicknesses):
        """
        Present for trainer compatibility. For states-only datasets, contacts are absent.

        If contact tensors are provided, returns an "all true" mask of appropriate shape.
        """
        if contact_depths is None:
            return None
        return torch.ones_like(contact_depths, dtype=torch.bool, device=contact_depths.device)

    def process_neural_model_inputs(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize/complete the model_inputs dict in-place:
        - ensures (B,T,dim) shapes
        - provides states_embedding if requested
        - wraps angles for states / next_states
        """
        if "states" not in model_inputs:
            raise KeyError("model_inputs must contain key 'states'")

        model_inputs["states"] = _ensure_bt(model_inputs["states"])
        if "next_states" in model_inputs and model_inputs["next_states"] is not None:
            model_inputs["next_states"] = _ensure_bt(model_inputs["next_states"])

        if "root_body_q" in model_inputs and model_inputs["root_body_q"] is not None:
            model_inputs["root_body_q"] = _ensure_bt(model_inputs["root_body_q"])
        if "gravity_dir" in model_inputs and model_inputs["gravity_dir"] is not None:
            model_inputs["gravity_dir"] = _ensure_bt(model_inputs["gravity_dir"])

        if self.states_embedding_type in (None, "identical"):
            if "states_embedding" not in model_inputs or model_inputs["states_embedding"] is None:
                model_inputs["states_embedding"] = model_inputs["states"]
            else:
                model_inputs["states_embedding"] = _ensure_bt(model_inputs["states_embedding"])

        self.wrap2PI(model_inputs["states"])
        if model_inputs.get("next_states", None) is not None:
            self.wrap2PI(model_inputs["next_states"])

        return model_inputs

    def get_neural_model_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Assemble model inputs for a transformer.

        If history is empty (e.g. dummy call for network construction),
        returns zero tensors with a singleton time dimension.
        """
        if len(self.states_history) == 0:
            processed_model_inputs: Dict[str, torch.Tensor] = {
                "root_body_q": torch.zeros_like(self.root_body_q).unsqueeze(1),
                "states": torch.zeros_like(self.states).unsqueeze(1),
                "gravity_dir": torch.zeros_like(self.gravity_dir).unsqueeze(1),
            }
            if self.states_embedding_type in (None, "identical"):
                processed_model_inputs["states_embedding"] = processed_model_inputs[
                    "states"
                ].clone()
            return self.process_neural_model_inputs(processed_model_inputs)

        model_inputs: Dict[str, torch.Tensor] = torch.utils.data.default_collate(
            list(self.states_history)
        )
        for k in model_inputs:
            model_inputs[k] = model_inputs[k].permute(1, 0, 2)

        processed_model_inputs = self.process_neural_model_inputs(model_inputs)
        return processed_model_inputs

    def convert_next_states_to_prediction(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert dataset (states, next_states) into prediction target.
        Both tensors can be (B,T,D) or (B,D).
        """
        states_bt = _ensure_bt(states)
        next_bt = _ensure_bt(next_states)

        if states_bt.shape[-1] != self.state_dim or next_bt.shape[-1] != self.state_dim:
            raise ValueError("states/next_states last dim must equal state_dim")

        if self.prediction_type == "absolute":
            pred = next_bt.clone()
        else:
            pred = (next_bt - states_bt)

            q_delta = pred[..., : self.dof_q_per_env]
            q_sel = q_delta.index_select(-1, self.angular_q_indices.to(q_delta.device))
            _wrap_to_pi_(q_sel)
            q_delta.index_copy_(-1, self.angular_q_indices.to(q_delta.device), q_sel)

        if pred.shape[-1] != self.prediction_dim:
            raise RuntimeError("Internal error: pred dim mismatch")
        return pred

    def convert_prediction_to_next_states(
        self,
        states: torch.Tensor,
        prediction: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert model output prediction back into next_states.
        Tensors can be (B,T,*) or (B,*).
        """
        states_bt = _ensure_bt(states)
        pred_bt = _ensure_bt(prediction)

        if pred_bt.shape[-1] < self.prediction_dim:
            raise ValueError(f"prediction last dim must be at least {self.prediction_dim}")

        pred_bt = pred_bt[..., : self.prediction_dim]

        if self.prediction_type == "absolute":
            next_states = pred_bt.clone()
        else:
            next_states = states_bt + pred_bt

        self.wrap2PI(next_states)
        return next_states

    def calculate_total_energy(self, state_min_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculate total energy (kinetic + potential) per sample.

        state_min_coords: (..., state_dim) with state_dim >= 4, layout [q0, q1, q0_dot, q1_dot].
        Returns: (...,) same shape as input without the last dimension (one energy per sample).
        """
        q0 = state_min_coords[..., 0] - torch.pi / 2
        q1 = q0 + state_min_coords[..., 1]
        q0_dot = state_min_coords[..., 2]
        q1_dot = q0_dot + state_min_coords[..., 3]
        l = 1  # TODO: get from model
        m = 1  # TODO: get from model
        g = 9.81
        E_tot = (
            0.5 * m * g * l * (-3 * torch.cos(q0) - torch.cos(q1))
            + (1 / 6) * m * l**2 * (q1_dot**2 + 4 * q0_dot**2 + 3 * q0_dot * q1_dot * torch.cos(q0 - q1))
        )
        return E_tot
