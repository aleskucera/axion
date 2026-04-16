import torch
import torch.nn.functional as F
import warp as wp

from axion.learning.residual_loss_utils import (
    compute_residual_diagnostics,
    validate_warm_start_shapes,
)
from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
from axion.neural_solver.utils.differentiable import pendulum_revolute_minimal_to_maximal_velocities
from axion.neural_solver.utils import warp_utils
from axion.neural_solver.utils.python_utils import print_warning


class VelAndLambdaTrainer(SequenceModelTrainer):
    """Residual-only trainer for velocity + lambda warm-start prediction."""

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        if not self.has_state_head or not self.has_lambda_head:
            raise ValueError(
                "VelAndLambdaTrainer requires both network.enable_state_head and "
                "network.enable_lambda_head to be true."
            )

        sim_wrapper = self.neural_env.simulator_wrapper
        self._sim_wrapper = sim_wrapper
        self._engine = sim_wrapper.engine
        self._control = sim_wrapper.control
        self._dt = sim_wrapper.frame_dt
        self._dims = self._engine.dims
        self._dof_q_per_env = int(self.neural_env.dof_q_per_env)
        self._dof_qd_per_env = int(self.neural_env.dof_qd_per_env)
        self._validate_prediction_type_cfg_for_loss()

        if int(self.batch_size) != int(self._dims.num_worlds):
            raise ValueError(
                "VelAndLambdaTrainer requires algorithm.batch_size == env.num_envs "
                f"(engine num_worlds). Got batch_size={self.batch_size}, "
                f"num_worlds={self._dims.num_worlds}."
            )
        self._setup_pendulum_fk_constants()

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.residual_loss_weight = float(loss_cfg.get("residual_loss_weight", 1.0))
        self.supervised_loss_weight = float(loss_cfg.get("supervised_loss_weight", 1.0))
        self.use_supervised_loss_state = bool(loss_cfg.get("use_supervised_loss_state", False))
        self.use_supervised_loss_lambdas = bool(loss_cfg.get("use_supervised_loss_lambdas", False))

        # Residual-only default: disable rollout eval until residual path is stable.
        if bool(loss_cfg.get("disable_rollout_eval", True)) and self.eval_interval > 0:
            print_warning(
                "VelAndLambdaTrainer disables rollout eval by default for residual-only "
                "training. Set algorithm.loss.disable_rollout_eval=false to re-enable."
            )
            self.eval_interval = 0

    def _compose_total_loss(
        self,
        residual_loss: torch.Tensor,
        supervised_loss: torch.Tensor = None,
    ):
        """Single loss aggregator for residual + optional supervised loss."""
        total_loss = self.residual_loss_weight * residual_loss
        if self.use_supervised_loss_state or self.use_supervised_loss_lambdas:
            total_loss = total_loss + self.supervised_loss_weight * supervised_loss
        return total_loss

    def _compute_supervised_losses(
        self,
        data: dict,
        predicted_next_states: torch.Tensor,
        predicted_next_lambdas: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Compute optional next-step MSE losses against dataset targets."""
        supervised_components = {}

        if self.use_supervised_loss_state:
            next_states = data["next_states"]
            next_states_last = next_states[:, -1, :]
            q_dim = self._dof_q_per_env
            predicted_q = predicted_next_states[:, :q_dim]
            target_q = next_states_last[:, :q_dim]
            # Periodic angular loss robust to wrap discontinuities.
            q_loss = torch.mean(1.0 - torch.cos(predicted_q - target_q))

            predicted_qd = predicted_next_states[:, q_dim:]
            target_qd = next_states_last[:, q_dim:]
            qd_loss = F.mse_loss(predicted_qd, target_qd)

            supervised_components["supervised_state_loss"] = q_loss + qd_loss

        if self.use_supervised_loss_lambdas:
            next_lambdas = data["next_lambdas"]
            next_lambdas_last = next_lambdas[:, -1, :]
            supervised_components["supervised_lambda_loss"] = F.mse_loss(
                predicted_next_lambdas,
                next_lambdas_last,
            )

        if not supervised_components:
            return None, supervised_components

        supervised_loss = sum(supervised_components.values())
        return supervised_loss, supervised_components

    def _setup_pendulum_fk_constants(self) -> None:
        """Cache per-joint FK constants for minimal->maximal velocity conversion."""
        model = self._sim_wrapper.model
        worlds = int(self._dims.num_worlds)
        joints_per_world = int(self._sim_wrapper.num_joints_per_world)

        joint_axis = wp.to_torch(model.joint_axis).to(self.device)
        joint_x_p = wp.to_torch(model.joint_X_p).to(self.device)
        joint_x_c = wp.to_torch(model.joint_X_c).to(self.device)
        body_com = wp.to_torch(model.body_com).to(self.device)

        # Replicated model has identical per-world kinematic constants, use world 0 slice.
        self._fk_joint_axis_parent_23 = (joint_axis.reshape(worlds, joints_per_world, 3)[0].to(torch.float32).contiguous())
        self._fk_joint_x_p_27 = (joint_x_p.reshape(worlds, joints_per_world, 7)[0].to(torch.float32).contiguous())
        self._fk_joint_x_c_27 = (joint_x_c.reshape(worlds, joints_per_world, 7)[0].to(torch.float32).contiguous())
        self._fk_body_com_23 = (
            body_com.reshape(worlds, self._sim_wrapper.bodies_per_world, 3)[0].to(torch.float32).contiguous()
        )

    def _convert_minimal_state_to_maximal_vel(
        self,
        state_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Convert predicted minimal state [q, qd] to maximal N_u velocity."""
        if state_prediction.ndim != 2:
            raise RuntimeError(
                "State prediction must be rank-2 at the last step: "
                f"expected (B, D), got {tuple(state_prediction.shape)}."
            )

        q_for_fk = state_prediction[:, : self._dof_q_per_env]
        qd_for_fk = state_prediction[:, self._dof_q_per_env :]

        body_vel_prediction = pendulum_revolute_minimal_to_maximal_velocities(
            q_b2=q_for_fk,
            qd_b2=qd_for_fk,
            joint_axis_parent_23=self._fk_joint_axis_parent_23,
            joint_x_p_27=self._fk_joint_x_p_27,
            joint_x_c_27=self._fk_joint_x_c_27,
            body_com_23=self._fk_body_com_23,
            body_count=self._sim_wrapper.bodies_per_world,
        )
        if body_vel_prediction.shape[1] != self._dims.N_u:
            raise RuntimeError(
                "Converted maximal velocity dim mismatch: expected "
                f"{self._dims.N_u}, got {body_vel_prediction.shape[1]}."
            )
        return body_vel_prediction

    def _extract_last_step_axion_contacts(self, data: dict) -> dict:
        """Read strict `axion_*` tensors and take the last time step."""
        required = (
            "axion_contact_count",
            "axion_contact_point0",
            "axion_contact_point1",
            "axion_contact_normal",
            "axion_contact_shape0",
            "axion_contact_shape1",
            "axion_contact_thickness0",
            "axion_contact_thickness1",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(
                "Residual training requires recorded axion contact tensors in the dataset. "
                f"Missing keys: {missing}. Expected fields under data/axion_contacts/*."
            )
        axion_contacts_last = {}
        for key in required:
            value = data[key]
            # Works for both (B, T, D) and (B, T): keeps feature dims when present.
            axion_contacts_last[key] = value[:, -1]

        return axion_contacts_last

    def _load_engine_step_from_states_and_contacts(
        self,
        states_bt: torch.Tensor,
        axion_contacts_bt: dict,
    ) -> None:
        """Set simulator state and load recorded Axion contact buffers into the engine."""
        if states_bt.ndim != 2:
            raise RuntimeError(
                f"Expected last-step states as (B, state_dim), got shape={tuple(states_bt.shape)}."
            )
        if states_bt.shape[0] != self._dims.num_worlds:
            raise RuntimeError(
                "Residual path requires one world per batch row in the last-step states: "
                f"expected B={self._dims.num_worlds}, got B={states_bt.shape[0]}."
            )
        expected_state_dim = self.neural_env.state_dim
        if states_bt.shape[1] != expected_state_dim:
            raise RuntimeError(
                f"State dim mismatch: expected {expected_state_dim}, got {states_bt.shape[1]}."
            )

        worlds = self._dims.num_worlds
        max_contacts = int(self._engine.axion_contacts.max_contacts)
        contact_count = axion_contacts_bt["axion_contact_count"]
        if contact_count.ndim == 2 and contact_count.shape[-1] == 1:
            contact_count = contact_count[:, 0]
        if contact_count.shape != (worlds,):
            raise RuntimeError(
                "axion_contact_count shape mismatch: expected "
                f"({worlds},), got {tuple(contact_count.shape)}."
            )
        contact_count_i32 = contact_count.to(device=self.device, dtype=torch.int32)

        warp_utils.assign_states_from_torch(self._sim_wrapper, states_bt)
        warp_utils.eval_fk(self._sim_wrapper.model, self._sim_wrapper.state)

        # Even though `self._engine` is `sim_wrapper.engine`, these `engine.data.*` arrays
        # are solver buffers, not live views into `state`/`control`.
        # We must refresh them per batch (equivalent to the non-contact part of load_data()).
        # `eval_fk` above is still needed so `state.body_q` matches the assigned joint state.
        # This is equivalent to `self._engine.load_data(self._sim_wrapper.state, self._control, self._engine.axion_contacts, self._dt)`.
        self._engine.data.dt = self._dt
        wp.copy(dest=self._engine.data.ext_force, src=self._sim_wrapper.state.body_f)
        wp.copy(dest=self._engine.data.joint_target_pos, src=self._control.joint_target_pos)
        wp.copy(dest=self._engine.data.joint_target_vel, src=self._control.joint_target_vel)
        wp.copy(dest=self._engine.data.body_pose_prev, src=self._sim_wrapper.state.body_q)
        wp.copy(dest=self._engine.data.body_vel_prev, src=self._sim_wrapper.state.body_qd)

        # Inject recorded contact buffers directly into Axion batched contacts.
        contact_point0 = axion_contacts_bt["axion_contact_point0"].to(device=self.device, dtype=torch.float32).reshape(worlds, max_contacts, 3).contiguous()
        contact_point1 = axion_contacts_bt["axion_contact_point1"].to(device=self.device, dtype=torch.float32).reshape(worlds, max_contacts, 3).contiguous()
        contact_normal = axion_contacts_bt["axion_contact_normal"].to(device=self.device, dtype=torch.float32).reshape(worlds, max_contacts, 3).contiguous()
        contact_shape0 = axion_contacts_bt["axion_contact_shape0"].to(device=self.device, dtype=torch.int32).reshape(worlds, max_contacts).contiguous()
        contact_shape1 = axion_contacts_bt["axion_contact_shape1"].to(device=self.device, dtype=torch.int32).reshape(worlds, max_contacts).contiguous()
        contact_thickness0 = axion_contacts_bt["axion_contact_thickness0"].to(device=self.device, dtype=torch.float32).reshape(worlds, max_contacts).contiguous()
        contact_thickness1 = axion_contacts_bt["axion_contact_thickness1"].to(device=self.device, dtype=torch.float32).reshape(worlds, max_contacts).contiguous()

        wp.copy(self._engine.axion_contacts.contact_count, wp.from_torch(contact_count_i32.contiguous()))
        wp.copy(self._engine.axion_contacts.contact_point0, wp.from_torch(contact_point0, dtype=wp.vec3),)
        wp.copy(self._engine.axion_contacts.contact_point1,wp.from_torch(contact_point1, dtype=wp.vec3),)
        wp.copy(self._engine.axion_contacts.contact_normal,wp.from_torch(contact_normal, dtype=wp.vec3),)
        wp.copy(self._engine.axion_contacts.contact_shape0, wp.from_torch(contact_shape0))
        wp.copy(self._engine.axion_contacts.contact_shape1, wp.from_torch(contact_shape1))
        wp.copy(self._engine.axion_contacts.contact_thickness0, wp.from_torch(contact_thickness0))
        wp.copy(self._engine.axion_contacts.contact_thickness1, wp.from_torch(contact_thickness1))

    def _split_prediction_last_step(self, prediction: dict):
        vel_prediction = prediction.get("state")
        lambda_prediction = prediction.get("lambda")
        if vel_prediction is None or lambda_prediction is None:
            raise RuntimeError(
                "VelAndLambdaTrainer expects both prediction['state'] and "
                "prediction['lambda'] to be present."
            )

        if vel_prediction.ndim == 3:
            vel_prediction = vel_prediction[:, -1, :]
        if lambda_prediction.ndim == 3:
            lambda_prediction = lambda_prediction[:, -1, :]

        return vel_prediction, lambda_prediction

    def _validate_prediction_type_cfg_for_loss(self) -> None:
        """Require matching state/lambda prediction semantics in residual-loss training."""
        state_prediction_type = self.utils_provider.state_prediction_type
        lambda_prediction_type = self.utils_provider.lambda_prediction_type
        if state_prediction_type != lambda_prediction_type:
            raise ValueError(
                "VelAndLambdaTrainer requires matching prediction types for residual loss. "
                "Set env.utils_provider_cfg.state_prediction_type and "
                "env.utils_provider_cfg.lambda_prediction_type to the same value, got "
                f"state={state_prediction_type!r}, lambda={lambda_prediction_type!r}."
            )

    def _convert_predictions_to_absolute_for_loss(
        self,
        states_last: torch.Tensor,
        lambdas_last: torch.Tensor,
        state_prediction: torch.Tensor,
        lambda_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert prediction-space outputs into absolute next-state warm starts."""
        next_state_prediction = self.utils_provider.convert_prediction_to_next_states(
            states_last,
            state_prediction,
            dt=self._dt,
        )
        next_lambda_prediction = self.utils_provider.convert_prediction_to_next_lambdas(
            lambdas_last,
            lambda_prediction,
        )

        if next_state_prediction.ndim == 3:
            next_state_prediction = next_state_prediction[:, -1, :]
        if next_lambda_prediction.ndim == 3:
            next_lambda_prediction = next_lambda_prediction[:, -1, :]

        if next_state_prediction.shape[-1] != self.neural_env.state_dim:
            raise RuntimeError(
                "Residual warm-start state must be an absolute full minimal state. "
                f"Expected last dim {self.neural_env.state_dim}, got "
                f"{next_state_prediction.shape[-1]}."
            )
        if next_lambda_prediction.shape[-1] != self._dims.num_constraints:
            raise RuntimeError(
                "Residual warm-start lambda dim mismatch: expected "
                f"{self._dims.num_constraints}, got {next_lambda_prediction.shape[-1]}."
            )

        return next_state_prediction, next_lambda_prediction

    def compute_loss(self, data, train):
        del train  # compatibility with SequenceModelTrainer.one_epoch
        prediction = self.neural_model(data)
        state_prediction, lambda_prediction = self._split_prediction_last_step(prediction)

        states = data["states"]
        lambdas = data["lambdas"]
        states_last = states[:, -1, :]
        lambdas_last = lambdas[:, -1, :]
        state_for_loss, lambda_for_loss = self._convert_predictions_to_absolute_for_loss(
            states_last,
            lambdas_last,
            state_prediction,
            lambda_prediction,
        )
        axion_contacts_last = self._extract_last_step_axion_contacts(data)
        body_vel_prediction = self._convert_minimal_state_to_maximal_vel(state_for_loss)

        batch_worlds = body_vel_prediction.shape[0]
        worlds = self._dims.num_worlds
        if batch_worlds != worlds:
            raise RuntimeError(
                "Residual training requires batch size to equal engine worlds: "
                f"batch={batch_worlds}, num_worlds={worlds}."
            )

        validate_warm_start_shapes(self._dims, body_vel_prediction, lambda_for_loss)
        self._load_engine_step_from_states_and_contacts(states_last, axion_contacts_last)
        residual_loss, residual_blocks_sq, residual_blocks_mse = compute_residual_diagnostics(
            self._engine,
            body_vel_prediction,
            lambda_for_loss,
        )
        supervised_loss, supervised_components = self._compute_supervised_losses(
            data,
            state_for_loss,
            lambda_for_loss,
        )
        total_loss = self._compose_total_loss(
            residual_loss=residual_loss,
            supervised_loss=supervised_loss,
        )

        with torch.no_grad():
            loss_itemized = {
                "residual_loss": residual_loss.detach(),
                "weighted_residual_loss": (self.residual_loss_weight * residual_loss).detach(),
                "total_loss": total_loss.detach(),
            }
            if supervised_loss is not None:
                loss_itemized["supervised_loss"] = supervised_loss.detach()
                loss_itemized["weighted_supervised_loss"] = (
                    self.supervised_loss_weight * supervised_loss
                ).detach()
            for key, value in supervised_components.items():
                loss_itemized[key] = value.detach()
                loss_itemized[f"weighted_{key}"] = (
                    self.supervised_loss_weight * value
                ).detach()
            for key, value in residual_blocks_sq.items():
                loss_itemized[key] = value.detach()
            for key, value in residual_blocks_mse.items():
                loss_itemized[key] = value.detach()
        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)