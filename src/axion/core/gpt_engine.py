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
from axion.neural_solver.standalone.neural_predictor import (
    NeuralPredictor,
    control_active_mask_from_newton_model,
)
from axion.neural_solver.standalone.fast_neural_predictor import FastNeuralPredictor
from axion.nn_prediction import models, utils

# Allow pickled checkpoints that reference classes under "models.*" and "utils.*"
# (saved when the trainer was run with a different sys.path) to load correctly.
sys.modules['models'] = models
sys.modules['utils'] = utils


BEST_STATE_ONLY_MODEL = "03-17-2026-15-12-19"   # debatable. # bet_eval_model.pt
MODEL_27 = "03-12-2026-16-09-14"
BEST_STATE_AND_LAMBDA_MODEL = Path("mse") / "05-12-2026-17-30-11"  # best_valid_valid_model.pt

BEST_TRAINED_FROM_CONTACT_SWEEP = Path("sweep9e86ytgi")/"a4fvu450"    # still bad
BEST_STATE_ONLY_MODEL_CONTACT_INTER = "03-25-2026-11-47-56"

NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/ BEST_TRAINED_FROM_CONTACT_SWEEP
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_eval_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"

# Flip to True after running export_to_onnx.py + build_tensorrt_engine.py.
# Required for `execution: cuda_graph` — only the TRT path is capture-safe.
USE_TENSORRT_ENGINE = False
NN_PENDULUM_PLAN_PATH = NN_PENDULUM_PT_PATH.with_suffix(".plan")
NN_PENDULUM_META_PATH = NN_PENDULUM_PT_PATH.with_suffix(".engine_meta.pt")

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
            # Older checkpoints were saved before has_state_head / has_lambda_head (Backfill sensible defaults)
            if not hasattr(loaded_nn_model, 'has_state_head'):
                loaded_nn_model.has_state_head = True
                print("Warning: checkpoint predates 'has_state_head' — assuming True (state-only model)")
            if not hasattr(loaded_nn_model, 'has_lambda_head'):
                loaded_nn_model.has_lambda_head = False
            # PyTorch path
            self.nn_predictor = NeuralPredictor(
                newton_model=self.model,
                nn_model=loaded_nn_model,
                nn_cfg=loaded_nn_cfg,
                device=str(self.device),
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

        # Extract joint target positions from control (shape: dof_q -> (1, dof_q)).
        joint_target_pos = wp.to_torch(control.joint_target_pos).unsqueeze(0)
        pred = self.nn_predictor
        control_active = control_active_mask_from_newton_model(
            self.model,
            dof_q_per_env=pred.dof_q_per_env,
            num_worlds=pred.num_worlds,
            num_joints_per_env=pred.num_joints_per_env,
            joint_q_start=pred.joint_q_start,
            joint_q_end=pred.joint_q_end,
            device=pred.device,
        )

        # Process the inputs (Neural Predictor does this internally using process_inputs)
        self.nn_predictor.process_inputs(
            state_in,
            axion_contacts,
            dt,
            joint_target_pos=joint_target_pos,
            control_active=control_active,
        )

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
