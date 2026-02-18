"""
Newton/Axion-friendly stateful integrator.

This mirrors the structure of StatefulNeuralIntegrator from the original
Warp-based code, but builds on top of NewtonBasedNeuralIntegrator and
keeps a short history of state-like inputs in a deque.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import torch

from axion.neural_solver.integrators.newton_based_integrator_neural import (
    NewtonBasedNeuralIntegrator,
)


class NewtonBasedStatefulNeuralIntegrator(NewtonBasedNeuralIntegrator):
    """
    Stateful version of the Newton-based integrator.

    It maintains a fixed-length history of model-input dicts (1 per step), to be
    used by sequence models (RNNs, transformers, etc.).
    """

    def __init__(
        self,
        num_states_history: int = 1,
        **kwargs,
    ):
        self.num_states_history = int(num_states_history)
        super().__init__(**kwargs)
        self.reset_states_history()

    # ---- History management -------------------------------------------------

    def reset_states_history(self):
        self.states_history: deque[Dict[str, torch.Tensor]] = deque(
            maxlen=self.num_states_history
        )

    def reset(self):
        super().reset()
        self.reset_states_history()

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

    # ---- Model input assembly ----------------------------------------------

    def get_neural_model_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Assemble model inputs from the history.

        Shape: each value is (B, T, dim). For non-sequence usage, T will be 1.
        """
        if len(self.states_history) == 0:
            # Fallback to a single-step sample from base class.
            return super().get_neural_model_inputs()

        # default_collate stacks list[dict] into dict[key] -> (T, B, dim)
        model_inputs: Dict[str, torch.Tensor] = torch.utils.data.default_collate(
            list(self.states_history)
        )
        for k in model_inputs:
            # (T, B, dim) -> (B, T, dim)
            model_inputs[k] = model_inputs[k].permute(1, 0, 2)

        processed_model_inputs = self.process_neural_model_inputs(model_inputs)
        return processed_model_inputs

