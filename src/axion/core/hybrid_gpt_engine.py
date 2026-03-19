from typing import Optional

from newton import Contacts
from newton import Control
from newton import Model
from newton import State
import warp as wp

from .base_engine import AxionEngineBase
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig

# Neural network imports:
from pathlib import Path
import yaml
import torch
from newton import eval_fk
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/"03-14-2026-13-48-45" 
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_eval_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"


class HybridGPTEngine(AxionEngineBase):
    """
    Engine that uses GPT to predict initial guess for the Newton-Raphson solver.
    After that, it uses AxionEngine to solve the system of equations.
    """
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

        #########################################
        #  Neural network initialization
        #########################################

        print("GPTEngine is using the device = ", self.device)
        nn_model_path = NN_PENDULUM_PT_PATH
        nn_cfg_path = NN_PENDULUM_CFG_PATH

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

    def _neural_init_state_fn(
        self,
        state_in: State,
        state_out: State,
        axion_contacts: Contacts,
        dt: float,
    ) -> None:
        """
        Perform neural network model inference to get a initial guess for the Newton method.
        WARNING: For planar double pendulum only!
        """
        # Process inputs: coordinate frame conversion, state embedding.
        self.nn_predictor.process_inputs(state_in, axion_contacts, dt)
        
        # Predict using self.nn_predictor
        state_predicted = self.nn_predictor.predict() # (1, 4)
        pred_joint_q = wp.from_torch(state_predicted[0, :2].reshape(2,))
        pred_joint_qd = wp.from_torch(state_predicted[0, 2:].reshape(2,))
        
        # Perfrom FK: joint_q -> body_q
        wp.copy(dest=state_out.joint_q, src=pred_joint_q)
        wp.copy(dest=state_out.joint_qd, src=pred_joint_qd)
        eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)   # newton.eval_fk()
    
        # Transfer neural prediction of states into solver's working arrays:
        wp.copy(dest=self.data.body_pose, src=state_out.body_q)
        wp.copy(dest=self.data.body_vel, src=state_out.body_qd)

        # Inititial guess of lambda (constraint forces)
        self.compute_warm_start_forces()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        # Process Newton Contacts -> Axion Contacts
        self.load_data(state_in, control, contacts, dt)

        # Perform neural network model inference to get a initial guess for the Newton method.
        self._neural_init_state_fn(state_in, state_out, self.axion_contacts, dt)

        # Call Newton solver
        self._solve()
        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)

