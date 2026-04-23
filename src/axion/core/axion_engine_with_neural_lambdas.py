from typing import Optional

import warp as wp
import newton
from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig

# Neural network imports:
from pathlib import Path
import sys
import yaml
import torch
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
from axion.neural_solver.models.lambda_models import LambdaClassificationModel
from axion.neural_solver.models.mse_model import MSEModel
from axion.neural_solver.models.mtl_model import MTLModel
from axion.neural_solver.models import residual_model as residual_model_module
from axion.neural_solver.utils.neural_lambda_hdf5_logger import NeuralLambdaHDF5Logger
from axion.neural_solver.train.trained_models.selected_trained_models import CONTACT_MODELS, MTL_JUMP_MODELS

NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/MTL_JUMP_MODELS[0]
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"final_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"

# Binary simulator activity mask: |next_lambdas - lambdas| >= threshold (matches Pendulum MTL datasets "Th05" / cfg classification_prob_threshold).
SIM_LAMBDA_ACTIVITY_ABS_DELTA_THRESHOLD = 100

# Backward compatibility for checkpoints saved before residual_model.py rename.
if not hasattr(residual_model_module, "VelAndLambdaModel"):
    residual_model_module.VelAndLambdaModel = residual_model_module.ResidualModel
sys.modules.setdefault(
    "axion.neural_solver.models.vel_and_lambda_model",
    residual_model_module,
)

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

    @staticmethod
    def _is_residual_model(nn_model: torch.nn.Module) -> bool:
        model_cls = type(nn_model)
        if model_cls.__name__ in ("ResidualModel", "VelAndLambdaModel"):
            return True
        module_name = model_cls.__module__
        return bool(
            getattr(nn_model, "has_state_head", False)
            and getattr(nn_model, "has_lambda_head", False)
            and ("vel_and_lambda_model" in module_name or "residual_model" in module_name)
        )

    @staticmethod
    def _is_mtl_model(nn_model: torch.nn.Module) -> bool:
        if isinstance(nn_model, MTLModel):
            return True
        model_cls = type(nn_model)
        if model_cls.__name__ != "MTLModel":
            return False
        # Support both legacy and current MTL head naming.
        has_legacy_heads = hasattr(nn_model, "regression_head") and hasattr(
            nn_model, "classification_head"
        )
        has_current_heads = all(
            hasattr(nn_model, attr)
            for attr in ("state_head", "cls_head", "base_head", "jump_head")
        )
        return has_legacy_heads or has_current_heads

    @staticmethod
    def _is_mse_model(nn_model: torch.nn.Module) -> bool:
        if isinstance(nn_model, MSEModel):
            return True
        model_cls = type(nn_model)
        if model_cls.__name__ == "MSEModel":
            return True
        return (
            hasattr(nn_model, "regression_head")
            and not hasattr(nn_model, "classification_head")
            and hasattr(nn_model, "state_output_dim")
            and hasattr(nn_model, "lambda_output_dim")
            and hasattr(nn_model, "regression_output_dim")
        )

    @staticmethod
    def _decode_lambda_activity_from_logits(lambda_logits: torch.Tensor) -> torch.Tensor:
        if lambda_logits.ndim == 3:
            lambda_logits = (
                lambda_logits[:, -1, :]
                if lambda_logits.shape[1] > 1
                else lambda_logits.squeeze(1)
            )
            return (torch.sigmoid(lambda_logits) >= 0.5).to(dtype=torch.float32)
        if lambda_logits.ndim == 4:
            lambda_logits_last = (
                lambda_logits[:, -1, :, :]
                if lambda_logits.shape[1] > 1
                else lambda_logits.squeeze(1)
            )
            return torch.argmax(lambda_logits_last, dim=-1).to(dtype=torch.float32)
        raise RuntimeError(
            f"Unexpected lambda/classification logits shape ndim={lambda_logits.ndim}"
        )

    @staticmethod
    def _squeeze_last_step_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor[:, -1, :] if tensor.shape[1] > 1 else tensor.squeeze(1)
        return tensor

    def _parse_mtl_outputs(
        self, out: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize old/new MTL outputs to:
          - state_prediction: (B, state_output_dim)
          - lambda_prediction: (B, lambda_output_dim)
          - lambda_activity_logits: (B, lambda_output_dim)
          - lambda_jump: (B, lambda_output_dim) or None if the model has no jump head
        """
        state_prediction = None
        lambda_prediction = None
        lambda_activity_logits = None
        lambda_jump = None

        # New MTL output schema: state/logits/lambda_hat
        if "state" in out and "lambda_hat" in out:
            state_prediction = out["state"]
            lambda_prediction = out["lambda_hat"]
            lambda_activity_logits = out.get("logits", None)
            lambda_jump = out.get("jump", None)
        # Legacy schema: regression (state + lambda) / classification
        elif "regression" in out:
            regression = out["regression"]
            mtl = self.nn_predictor.nn_model
            sod = int(mtl.state_output_dim)
            regression = self._squeeze_last_step_if_needed(regression)
            state_prediction = regression[:, :sod]
            lambda_prediction = regression[:, sod:]
            lambda_activity_logits = out.get("classification", None)

        if state_prediction is None or lambda_prediction is None:
            raise RuntimeError(
                "MTL engine expected model.evaluate(...) to return either "
                "new-style keys ('state', 'lambda_hat') or legacy key ('regression'). "
                f"Got keys={sorted(list(out.keys()))} for model={type(self.nn_predictor.nn_model).__name__}."
            )
        if lambda_activity_logits is None:
            raise RuntimeError(
                "MTL engine expected classification/logit outputs under "
                "'logits' (new) or 'classification' (legacy). "
                f"Got keys={sorted(list(out.keys()))} for model={type(self.nn_predictor.nn_model).__name__}."
            )

        state_prediction = self._squeeze_last_step_if_needed(state_prediction)
        lambda_prediction = self._squeeze_last_step_if_needed(lambda_prediction)
        lambda_jump_out = (
            self._squeeze_last_step_if_needed(lambda_jump)
            if lambda_jump is not None
            else None
        )
        return state_prediction, lambda_prediction, lambda_activity_logits, lambda_jump_out

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
        loaded_nn_model, robot_name = torch.load(
            nn_model_path, map_location=str(self.device), weights_only=False
        )
        self._use_mtl_model = self._is_mtl_model(loaded_nn_model)
        self._use_residual_model = self._is_residual_model(loaded_nn_model)
        self._use_lambda_classification = self._is_lambda_classification_model(loaded_nn_model)
        self._use_mse_model = self._is_mse_model(loaded_nn_model)
        print(f"Loaded model for robot: {robot_name}")
        if self._use_mtl_model:
            model_mode = "mtl"
        elif self._use_lambda_classification:
            model_mode = "classification"
        elif self._use_residual_model:
            model_mode = "residual"
        elif self._use_mse_model:
            model_mode = "mse"
        else:
            model_mode = "regression"
        print("Loaded neural lambda model mode:", model_mode)
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, "r") as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize NeRDPredictor: robot config is inferred from self.model (newton.Model)
        self.nn_predictor = NeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
            lambda_prediction_only=not (
                self._use_residual_model or self._use_mtl_model or self._use_mse_model
            ),
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def _log_neural_step(
        self,
        state_in: State,
        state_out: State,
        nn_inputs: dict[str, torch.Tensor],
        sim_lambdas_before_step: torch.Tensor,
        predicted_next_lambdas: Optional[torch.Tensor],
        predicted_next_states: Optional[torch.Tensor],
        lambda_activity: Optional[torch.Tensor],
        lambda_jump: Optional[torch.Tensor] = None,
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

        next_lambdas_torch = wp.to_torch(self.data._constr_force)
        lambda_activity_gt = (
            (next_lambdas_torch - sim_lambdas_before_step).abs()
            >= SIM_LAMBDA_ACTIVITY_ABS_DELTA_THRESHOLD
        ).to(dtype=torch.float32)

        self.neural_dataset_logger.append_step(
            states=self._to_numpy(nn_inputs["states"][:, -1, :]),
            next_states=self._to_numpy(next_states),
            contact_normals=self._to_numpy(processed_contacts["contact_normals"]),
            contact_depths=self._to_numpy(processed_contacts["contact_depths"]),
            contact_points_0=self._to_numpy(processed_contacts["contact_points_0"]),
            contact_points_1=self._to_numpy(processed_contacts["contact_points_1"]),
            contact_thicknesses=self._to_numpy(processed_contacts["contact_thicknesses"]),
            lambdas=self._to_numpy(sim_lambdas_before_step),
            next_lambdas=self._to_numpy(next_lambdas_torch),
            predicted_next_lambdas=(
                self._to_numpy(predicted_next_lambdas)
                if predicted_next_lambdas is not None else None
            ),
            predicted_next_states=(
                self._to_numpy(predicted_next_states)
                if predicted_next_states is not None else None
            ),
            lambda_activity=(
                self._to_numpy(lambda_activity) if lambda_activity is not None else None
            ),
            lambda_activity_ground_truth=self._to_numpy(lambda_activity_gt),
            gravity_dir=self._to_numpy(nn_inputs["gravity_dir"][:, -1, :]),
            root_body_q=self._to_numpy(nn_inputs["root_body_q"][:, -1, :]),
            lambda_jump=(
                self._to_numpy(lambda_jump) if lambda_jump is not None else None
            ),
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

        # Keep a copy of current simulation lambdas before warm-start reset.
        sim_lambdas_before_step = wp.to_torch(self.data._constr_force).clone()

        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        # Predict lambdas only
        self.nn_predictor.process_inputs(state_in, self.axion_contacts, dt)
        nn_inputs = self.nn_predictor.nn_model_inputs
        predicted_next_lambdas = None
        predicted_next_states = None
        lambda_activity = None
        lambda_jump = None
        if self._use_mtl_model:
            with torch.no_grad():
                out = self.nn_predictor.nn_model.evaluate(nn_inputs)
            if not isinstance(out, dict):
                raise RuntimeError(
                    "MTL engine expected model.evaluate(...) to return a dict. "
                    f"Got {type(out).__name__} for model={type(self.nn_predictor.nn_model).__name__}."
                )
            (
                state_prediction,
                lambda_prediction,
                lambda_activity_logits,
                lambda_jump,
            ) = self._parse_mtl_outputs(out)
            cur_states = nn_inputs["states"][:, -1, :]
            predicted_next_states = self.nn_predictor._convert_prediction_to_next_states(  # noqa: SLF001
                cur_states,
                state_prediction,
                dt,
            )
            if (self.nn_predictor.lambdas is not None) and ("lambdas" in nn_inputs):
                cur_lambdas = nn_inputs["lambdas"][:, -1, :]
                predicted_next_lambdas = self.nn_predictor._convert_prediction_to_next_lambdas(  # noqa: SLF001
                    cur_lambdas,
                    lambda_prediction,
                )
                self.nn_predictor.lambdas.copy_(predicted_next_lambdas)
            lambda_activity = self._decode_lambda_activity_from_logits(lambda_activity_logits)
        elif self._use_lambda_classification:
            with torch.no_grad():
                lambda_logits = self.nn_predictor.nn_model.evaluate(nn_inputs).get("lambda", None)
            if lambda_logits is None:
                raise RuntimeError(
                    "Lambda classification engine expected model.evaluate(...) to return "
                    "a dict with key 'lambda'. Got None."
                )
            lambda_activity = self._decode_lambda_activity_from_logits(lambda_logits)
        elif self._use_residual_model:
            with torch.no_grad():
                out = self.nn_predictor.nn_model.evaluate(nn_inputs)
                state_prediction = out.get("state", None)
                lambda_prediction = out.get("lambda", None)
            if state_prediction is None or lambda_prediction is None:
                raise RuntimeError(
                    "ResidualModel engine expected model.evaluate(...) to return "
                    "both 'state' and 'lambda' tensors."
                )
            state_prediction = (
                state_prediction[:, -1, :]
                if state_prediction.shape[1] > 1
                else state_prediction.squeeze(1)
            )
            lambda_prediction = (
                lambda_prediction[:, -1, :]
                if lambda_prediction.shape[1] > 1
                else lambda_prediction.squeeze(1)
            )
            cur_states = nn_inputs["states"][:, -1, :]
            predicted_next_states = self.nn_predictor._convert_prediction_to_next_states(  # noqa: SLF001
                cur_states,
                state_prediction,
                dt,
            )
            if (self.nn_predictor.lambdas is not None) and ("lambdas" in nn_inputs):
                cur_lambdas = nn_inputs["lambdas"][:, -1, :]
                predicted_next_lambdas = self.nn_predictor._convert_prediction_to_next_lambdas(  # noqa: SLF001
                    cur_lambdas,
                    lambda_prediction,
                )
                self.nn_predictor.lambdas.copy_(predicted_next_lambdas)
        elif self._use_mse_model:
            with torch.no_grad():
                regression = self.nn_predictor.nn_model.evaluate(nn_inputs)  # (B, 1, regression_output_dim)
            mse = self.nn_predictor.nn_model
            sod = int(mse.state_output_dim)
            regression = (
                regression[:, -1, :] if regression.shape[1] > 1 else regression.squeeze(1)
            )  # (B, regression_output_dim)
            state_prediction = regression[:, :sod]
            lambda_prediction = regression[:, sod:]
            cur_states = nn_inputs["states"][:, -1, :]
            predicted_next_states = self.nn_predictor._convert_prediction_to_next_states(  # noqa: SLF001
                cur_states, state_prediction, dt,
            )
            # MSEModel has no lambda history tracking; zeros work for both absolute and relative
            predicted_next_lambdas = self.nn_predictor._convert_prediction_to_next_lambdas(  # noqa: SLF001
                torch.zeros_like(lambda_prediction),
                lambda_prediction,
            )
            # lambda_activity stays None — no classification head
        else:
            predicted_next_lambdas = self.nn_predictor.predict_lambdas_only(dt)

        self._solve()

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
        # Keep generalized coordinates in sync for logging/debug consumers.
        newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)
        self._log_neural_step(
            state_in=state_in,
            state_out=state_out,
            nn_inputs=nn_inputs,
            sim_lambdas_before_step=sim_lambdas_before_step,
            predicted_next_lambdas=predicted_next_lambdas,
            predicted_next_states=predicted_next_states,
            lambda_activity=lambda_activity,
            lambda_jump=lambda_jump,
        )
