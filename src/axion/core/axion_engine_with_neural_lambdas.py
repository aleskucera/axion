from typing import Optional

import warp as wp
from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig

# Neural network imports:
from pathlib import Path
import yaml
import torch
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
from axion.neural_solver.models.lambda_models import LambdaClassificationModel
from axion.neural_solver.utils.neural_lambda_hdf5_logger import NeuralLambdaHDF5Logger
NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/"lambda_classifier"/"04-01-2026-15-45-58"#"04-01-2026-12-43-28"
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"

class AxionEngineWithNeuralLambdas(AxionEngineBase):
    @staticmethod
    def _is_lambda_classification_model(nn_model: torch.nn.Module) -> bool:
        if isinstance(nn_model, LambdaClassificationModel):
            return True
        model_cls = type(nn_model)
        if model_cls.__name__ == "LambdaClassificationModel":
            return True
        # Fallback for legacy/import-path-mismatched checkpoints.
        has_state_head = bool(getattr(nn_model, "has_state_head", True))
        has_lambda_head = bool(getattr(nn_model, "has_lambda_head", False))
        return (not has_state_head) and has_lambda_head and ("lambda_models" in model_cls.__module__)

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
        #  Neural dataset logger initialization
        #########################################

        self.neural_dataset_logger = None
        self._neural_steps_logged = 0
        if bool(getattr(self.logging_config, "enable_neural_lambdas_logging", False)):
            # Disable shared core dataset logger for this one-off experiment logger path.
            self.dataset_logger = None
            self.neural_dataset_logger = NeuralLambdaHDF5Logger(
                output_path=self.logging_config.neural_lambdas_log_file
            )
            self._neural_max_steps = int(
                self.logging_config.neural_lambdas_simulation_steps
            )
        else:
            self._neural_max_steps = int(sim_steps)

        #########################################
        #  Neural network initialization
        #########################################

        print("AxionEngineWithNeuralLambdas is using the device = ", self.device)
        nn_model_path = NN_PENDULUM_PT_PATH
        nn_cfg_path = NN_PENDULUM_CFG_PATH

        # Load the nn .pt file and .cfg file correctly
        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(nn_model_path, map_location= str(self.device), weights_only= False)
        self._use_lambda_classification = self._is_lambda_classification_model(loaded_nn_model)
        print(f"Loaded model for robot: {robot_name}")
        print(
            "Loaded neural lambda model mode:",
            "classification" if self._use_lambda_classification else "regression",
        )
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize NeRDPredictor: robot config is inferred from self.model (newton.Model)
        self.nn_predictor = NeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
            lambda_prediction_only=True,
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def _log_neural_step(
        self,
        state_in: State,
        state_out: State,
        nn_inputs: dict[str, torch.Tensor],
        predicted_next_lambdas: Optional[torch.Tensor],
        lambda_activity: Optional[torch.Tensor],
    ) -> None:
        if self.neural_dataset_logger is None:
            return

        # Convert state_out to training-compatible state tensor.
        next_states = torch.cat(
            (wp.to_torch(state_out.joint_q), wp.to_torch(state_out.joint_qd)), dim=0
        ).unsqueeze(0)

        # Recompute full contact fields (including points_0/thicknesses) for dataset parity.
        body_q_2d = state_in.body_q.reshape(
            (self.nn_predictor.num_worlds, self.nn_predictor.bodies_per_world)
        )
        body_q_torch = wp.to_torch(body_q_2d)
        root_body_q = body_q_torch[:, 0, :].to(self.nn_predictor.device)
        processed_contacts = self.nn_predictor._convert_newton_contacts_to_contacts_for_nn_model(  # noqa: SLF001
            state_in,
            self.axion_contacts,
            root_body_q,
        )

        lambdas = None
        if "lambdas" in nn_inputs:
            lambdas = nn_inputs["lambdas"][:, -1, :]

        self.neural_dataset_logger.append_step(
            states=self._to_numpy(nn_inputs["states"][:, -1, :]),
            next_states=self._to_numpy(next_states),
            contact_normals=self._to_numpy(processed_contacts["contact_normals"]),
            contact_depths=self._to_numpy(processed_contacts["contact_depths"]),
            contact_points_0=self._to_numpy(processed_contacts["contact_points_0"]),
            contact_points_1=self._to_numpy(processed_contacts["contact_points_1"]),
            contact_thicknesses=self._to_numpy(processed_contacts["contact_thicknesses"]),
            lambdas=self._to_numpy(lambdas) if lambdas is not None else None,
            next_lambdas=self._to_numpy(wp.to_torch(self.data._constr_force)),
            predicted_next_lambdas=(
                self._to_numpy(predicted_next_lambdas)
                if predicted_next_lambdas is not None else None
            ),
            lambda_activity=(
                self._to_numpy(lambda_activity) if lambda_activity is not None else None
            ),
            gravity_dir=self._to_numpy(nn_inputs["gravity_dir"][:, -1, :]),
            root_body_q=self._to_numpy(nn_inputs["root_body_q"][:, -1, :]),
        )
        self._neural_steps_logged += 1

    def save_logs(self):
        super().save_logs()
        if self.neural_dataset_logger is not None:
            self.neural_dataset_logger.save()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        self.load_data(state_in, control, contacts, dt)

        # Initial guess: use current state (zero-order warm start)
        wp.copy(dest=self.data.body_pose, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel, src=state_in.body_qd)

        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        # Predict lambdas only
        self.nn_predictor.process_inputs(state_in, self.axion_contacts, dt)
        nn_inputs = self.nn_predictor.nn_model_inputs
        predicted_next_lambdas = None
        lambda_activity = None
        if self._use_lambda_classification:
            with torch.no_grad():
                lambda_logits = self.nn_predictor.nn_model.evaluate(nn_inputs).get("lambda", None)
            lambda_logits = lambda_logits[:, -1, :] if lambda_logits.shape[1] > 1 else lambda_logits.squeeze(1)
            lambda_activity = (torch.sigmoid(lambda_logits) >= 0.5).to(dtype=torch.float32)
        else:
            predicted_next_lambdas = self.nn_predictor.predict_lambdas_only(dt)

        self._solve()

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
        self._log_neural_step(
            state_in=state_in,
            state_out=state_out,
            nn_inputs=nn_inputs,
            predicted_next_lambdas=predicted_next_lambdas,
            lambda_activity=lambda_activity,
        )
