"""
WarmupGPTEngine — classical Axion solver for the first N steps, then pure NN.

During the warmup phase every physics step is solved by the full Newton-Raphson
Axion solver (identical to AxionEngine).  At the same time, process_inputs() is
called each warmup step so the NeuralPredictor's history deque is filled with
real trajectory data rather than the repeated-initial-state that prewarm() would
give.  Once the warmup budget is exhausted the engine switches permanently to
pure neural-network integration (identical to GPTEngine).

No TensorRT support — PyTorch path only.  Use execution: no_graph (the default
in warmup_gpt_pendulum.yaml); a Python step counter is incompatible with CUDA-
graph replay.
"""

from pathlib import Path
from typing import Optional

import warp as wp
import newton
import yaml
import torch
import sys

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
from axion.nn_prediction import models, utils

# Allow pickled checkpoints that reference classes under "models.*" and "utils.*"
# (saved when the trainer was run with a different sys.path) to load correctly.
sys.modules['models'] = models
sys.modules['utils'] = utils

NN_BASE_PATH = Path.cwd() / "src" / "axion" / "neural_solver" / "train" / "trained_models" / "mse" / "05-12-2026-17-30-11"
NN_PENDULUM_PT_PATH = NN_BASE_PATH / "nn" / "best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH / "cfg.yaml"


class WarmupGPTEngine(AxionEngineBase):
    """
    Physics engine that uses the classical Axion Newton-Raphson solver for the
    first ``warmup_steps`` steps, then switches permanently to pure neural-network
    integration.

    The NeuralPredictor's history buffer is populated during warmup with the
    ground-truth classical-solver trajectory, giving the NN a well-conditioned
    context when it takes over.

    WARNING: This physics solver is robot-model-dependent (double pendulum).
    TensorRT is not supported; use execution: no_graph.
    """

    def __init__(
        self,
        model: newton.Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        differentiable_simulation: bool = False,
        nn_model_path: Path = NN_PENDULUM_PT_PATH,
        nn_cfg_path: Path = NN_PENDULUM_CFG_PATH,
    ):
        if config is None:
            config = AxionEngineConfig()
        if logging_config is None:
            logging_config = LoggingConfig()

        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

        # Read warmup budget from config (WarmupGPTEngineConfig adds this field;
        # fall back to 10 when constructed with a plain AxionEngineConfig).
        self._warmup_steps: int = int(getattr(config, "warmup_steps", 10))
        self._step_count: int = 0

        print("WarmupGPTEngine is using the device =", self.device)
        print(f"  warmup_steps = {self._warmup_steps}")

        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, "r") as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(
            nn_model_path, map_location=str(self.device), weights_only=False
        )
        print(f"Loaded model for robot: {robot_name}")

        self.nn_predictor = NeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
        )

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        axion_contacts = self.nn_predictor.create_axion_contacts(contacts)

        if self._step_count < self._warmup_steps:
            # ------------------------------------------------------------------
            # Warmup: run the classical Axion solver and populate NN history.
            # process_inputs is called with state_in so the history holds
            # state[t], consistent with the training convention where the model
            # predicts state[t+1] from state[t].
            # ------------------------------------------------------------------
            self.nn_predictor.process_inputs(state_in, axion_contacts, dt)

            self.load_data(state_in, control, contacts, dt)
            wp.copy(dest=self.data.body_pose, src=state_in.body_q)
            wp.copy(dest=self.data.body_vel, src=state_in.body_qd)
            self.data._constr_force.zero_()
            self.data._constr_force_prev_iter.zero_()
            self._solve()
            wp.copy(dest=state_out.body_q, src=self.data.body_pose)
            wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
            newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)

            if self._step_count == self._warmup_steps - 1:
                print(
                    f"[WarmupGPTEngine] Warmup complete after {self._warmup_steps} steps."
                    " Switching to NN integration."
                )
        else:
            # ------------------------------------------------------------------
            # Post-warmup: pure neural-network integration (same as GPTEngine).
            # ------------------------------------------------------------------
            self.nn_predictor.process_inputs(state_in, axion_contacts, dt)
            state_predicted, _ = self.nn_predictor.predict(dt=dt)

            dof_q = self.nn_predictor.dof_q_per_env
            wp.copy(
                dest=state_out.joint_q,
                src=wp.from_torch(state_predicted[0, :dof_q]),
            )
            wp.copy(
                dest=state_out.joint_qd,
                src=wp.from_torch(state_predicted[0, dof_q:]),
            )
            newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        self._step_count += 1
