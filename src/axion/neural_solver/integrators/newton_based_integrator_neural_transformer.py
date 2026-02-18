"""
Newton/Axion-friendly transformer integrator.

This mirrors TransformerNeuralIntegrator from the original Warp-based code,
but is built on top of NewtonBasedStatefulNeuralIntegrator and prepares
sequence inputs suitable for transformer models.
"""

from __future__ import annotations

from collections import deque
from typing import Dict

import torch

from axion.neural_solver.integrators.newton_based_integrator_neural_stateful import (
    NewtonBasedStatefulNeuralIntegrator,
)


class NewtonBasedTransformerNeuralIntegrator(NewtonBasedStatefulNeuralIntegrator):
    """
    Transformer-oriented stateful integrator for Newton/Axion environments.

    It keeps a rolling history of state snapshots and produces (B, T, dim)
    tensors without flattening the time dimension, which matches what the
    transformer-based ModelMixedInput expects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_states_history(self):
        # For transformer we do not pre-fill the history; it starts empty and
        # grows as the env is stepped.
        self.states_history = deque(maxlen=self.num_states_history)

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

