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
from .engine_config import NeuralEngineConfig
from .engine_logger import EngineLogger

# axion - nn prediction
import yaml
import torch
import sys
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
from axion.nn_prediction import models, utils

# Allow pickled checkpoints that reference classes under "models.*" and "utils.*"
# (saved when the trainer was run with a different sys.path) to load correctly.
sys.modules['models'] = models
sys.modules['utils'] = utils

NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/"03-12-2026-16-09-14" #"03-01-2026-10-46-58" 
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_eval_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"
 
class NeuralEngine(SolverBase):
    """
    This class implements a neural physics solver.
    It predicts the next step of the simulation based on 
    a trained neural network.
    
    WARNING: This physics solver is robot-model-dependent.
    """

    def __init__(self, 
        model: Model,
        #init_state_fn: Callable[[State, State, Contacts, float], None],
        logger: EngineLogger,
        config: Optional[NeuralEngineConfig] = NeuralEngineConfig(),
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

        #self.init_state_fn = init_state_fn
        self.logger = logger
        self.config = config
        self.model = model

        #########################################
        #  Neural network initialization
        #########################################
        
        print("NeuralEngine is using the device = ", self.device)

        # Load the nn .pt file and .cfg file correctly
        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(nn_model_path, map_location= str(self.device), weights_only= False)
        print(f"Loaded model for robot: {robot_name}")
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize NeRDPredictor: robot config is inferred from self.model (newton.Model)
        self.nn_predictor = NeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
        )

        self.num_models = 1    # num_worlds

        # NeRD was learned in an environment where Y axis was the up axis
        self.gravity_vector = torch.zeros((self.num_models, 3), device= str(self.device))
        self.gravity_vector[:, self.model.up_axis] = -1.0 # copy the gravity dir from model (should be along Z) 

        # Infer root joint height from model (world-space position of joint with parent=-1)
        # Used for root_body_q position so it matches the pendulum height (e.g. PENDULUM_HEIGHT)
        joint_parent_np = self.model.joint_parent.numpy()
        root_joint_idx = None
        for j in range(self.model.joint_count):
            if joint_parent_np[j] == -1:
                root_joint_idx = j
                break

        joint_X_p = self.model.joint_X_p.numpy()
        root_joint_pos = joint_X_p[root_joint_idx][:3]
        self._root_joint_height = float(root_joint_pos[self.model.up_axis])

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):

        # PASS ALL THE INPUTS TO THE NN------------------------------------------------------------
        # Process the inputs (Neural Predictor does this internally using process_inputs)
        self.nn_predictor.process_inputs(state_in, contacts, dt)

        # Predict using self.nn_predictor
        state_predicted = self.nn_predictor.predict()

        # Write into state_out 
        state_out.joint_q = wp.from_torch(state_predicted[0,:2].reshape(2,))
        state_out.joint_qd = wp.from_torch(state_predicted[0,2:].reshape(2,))
        
        newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)
        