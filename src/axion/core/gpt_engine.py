"""
Documentation in docs/nerd-module-in-axion/neural-pendulum.md
"""

# typing and paths
from pathlib import Path
# from typing import Callable
from typing import Optional

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverBase
from newton import Model

# classic axion
from .engine_config import GPTEngineConfig
from .engine_logger import EngineLogger

# axion - nn prediction
import yaml
import torch
import sys
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
from axion.neural_solver.standalone.fast_neural_predictor import FastNeuralPredictor
from axion.neural_solver.standalone.neural_predictor_helpers import shift_body_qd_to_com_frame
from axion.nn_prediction import models, utils

# Allow pickled checkpoints that reference classes under "models.*" and "utils.*"
# (saved when the trainer was run with a different sys.path) to load correctly.
sys.modules['models'] = models
sys.modules['utils'] = utils

# Default to the MSE checkpoint that already has a built .plan / .engine_meta.pt
# (see `src/axion/neural_solver/docs/torch_to_tensorrt_conversion.md`). The
# torch-only path still works against this .pt file.
NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/"mse"/"05-12-2026-08-49-22"
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"

# Flip to True after running export_to_onnx.py + build_tensorrt_engine.py.
# Required for `execution: cuda_graph` — only the TRT path is capture-safe.
USE_TENSORRT_ENGINE = False
NN_PENDULUM_PLAN_PATH = NN_PENDULUM_PT_PATH.with_suffix(".plan")
NN_PENDULUM_META_PATH = NN_PENDULUM_PT_PATH.with_suffix(".engine_meta.pt")

# Flip to False to skip the post-eval_fk body_qd frame correction for testing.
# Should normally stay True: eval_fk writes body_qd at the parent-side joint
# anchor; this shift moves it to the CoM frame expected by the rest of the
# pipeline (NeuralPredictor uses eval_ik which re-derives joint_qd from body_qd).
SHIFT_BODY_QD_TO_COM = False
 
class GPTEngine(SolverBase):
    """
    This class implements a neural physics solver.
    It predicts the next step of the simulation based on 
    a trained neural network.
    
    WARNING: This physics solver is robot-model-dependent.
    """

    def __init__(self, 
        model: Model,
        logger: EngineLogger,
        config: Optional[GPTEngineConfig] = GPTEngineConfig(),
        nn_model_path: Path = NN_PENDULUM_PT_PATH ,
        nn_cfg_path: Path = NN_PENDULUM_CFG_PATH,
        ):
        """
        Initialize the neural engine for the given robot model, neural network model and neural network configuration.

        Args:
            model: The warp.sim.Model physics model containing bodies, joints, and other physics properties.
            config: Configuration parameters for the engine of type EngineConfig.
            logger: Optional HDF5Logger or NullLogger for recording simulation data.
            nn_model_path: Filepath to the desired .pt file of the neural network.
            nn_cfg_path: Filepath to the desired .yaml configuration file of the neural network. 
        """

        # TODO: add some assertions if model != nn_model_path != nn_cfg_path

        #########################################
        #  Classic AxionEngine-like initialization
        #########################################

        super().__init__(model)

        self.logger = logger
        self.config = config
        self.model = model

        #########################################
        #  Neural network initialization
        #########################################
        
        print("GPTEngine is using the device = ", self.device)

        # Load the nn config file (always — needed for NeuralPredictor regardless of backend)
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Load either the TensorRT engine wrapper (duck-types MSEModel) or the
        # torch .pt checkpoint, depending on the toggle above.
        if USE_TENSORRT_ENGINE:
            from axion.neural_solver.fast_inference.tensorrt_mse_engine import (
                TensorRTMSEEngine,
            )
            print(f"Loading TensorRT engine: {NN_PENDULUM_PLAN_PATH}")
            loaded_nn_model = TensorRTMSEEngine(
                plan_path=NN_PENDULUM_PLAN_PATH,
                meta_path=NN_PENDULUM_META_PATH,
                device=str(self.device),
            )
            # The TRT path is the only capture-safe one — use the fast
            # predictor so subsequent `step()` calls become pure kernel
            # launches under `wp.ScopedCapture()`.
            self.nn_predictor = FastNeuralPredictor(
                newton_model=self.model,
                nn_model=loaded_nn_model,
                nn_cfg=loaded_nn_cfg,
                device=str(self.device),
            )
        else:
            print(f"Loading model from: {nn_model_path}")
            loaded_nn_model, robot_name = torch.load(
                nn_model_path, map_location=str(self.device), weights_only=False
            )
            print(f"Loaded model for robot: {robot_name}")
            # PyTorch path
            self.nn_predictor = NeuralPredictor(
                newton_model=self.model,
                nn_model=loaded_nn_model,
                nn_cfg=loaded_nn_cfg,
                device=str(self.device),
            )

        # Preallocated scratch for shift_body_qd_to_com_frame. Must be
        # allocated once (not per step) to remain CUDA-graph capture-safe.
        self._raw_body_qd = wp.empty(
            self.model.body_count,
            dtype=wp.spatial_vector,
            device=self.device,
        )

    def prewarm(
        self,
        state_in: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        """Seed the predictor's history buffer with the current state so the
        first captured step has a valid input. Called once eagerly by the
        InteractiveSimulator before `wp.ScopedCapture()` when the predictor
        supports it (FastNeuralPredictor only)."""
        prewarm_fn = getattr(self.nn_predictor, "prewarm", None)
        if prewarm_fn is None:
            return
        axion_contacts = self.nn_predictor.create_axion_contacts(contacts)
        prewarm_fn(state_in, axion_contacts, dt)

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        # PASS ALL THE INPUTS TO THE NN------------------------------------------------------------
        # Create AxionContacts object from Newton contacts
        axion_contacts = self.nn_predictor.create_axion_contacts(contacts)

        # Process the inputs (Neural Predictor does this internally using process_inputs)
        self.nn_predictor.process_inputs(state_in, axion_contacts, dt)

        # Predict using self.nn_predictor
        state_predicted, _ = self.nn_predictor.predict(dt=dt)

        # Write the prediction back to state_out via wp.copy(...)
        wp.copy(
            dest=state_out.joint_q,
            src=wp.from_torch(state_predicted[0, :self.nn_predictor.dof_q_per_env]),
        )
        wp.copy(
            dest=state_out.joint_qd,
            src=wp.from_torch(state_predicted[0, self.nn_predictor.dof_q_per_env:]),
        )

        newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        if SHIFT_BODY_QD_TO_COM:
            shift_body_qd_to_com_frame(self.model, state_out, self._raw_body_qd, self.device)
