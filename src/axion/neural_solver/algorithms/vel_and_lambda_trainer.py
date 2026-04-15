import torch
import warp as wp

from axion.learning.residual_loss_utils import compute_residual_loss, validate_warm_start_shapes
from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
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

        if int(self.batch_size) != int(self._dims.num_worlds):
            raise ValueError(
                "VelAndLambdaTrainer requires algorithm.batch_size == env.num_envs "
                f"(engine num_worlds). Got batch_size={self.batch_size}, "
                f"num_worlds={self._dims.num_worlds}."
            )

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.residual_loss_weight = float(loss_cfg.get("residual_loss_weight", 1.0))
        self.supervised_loss_weight = float(loss_cfg.get("supervised_loss_weight", 1.0))
        self.use_supervised_loss = bool(loss_cfg.get("use_supervised_loss", False))

        # Residual-only default: disable rollout eval until residual path is stable.
        if bool(loss_cfg.get("disable_rollout_eval", True)) and self.eval_interval > 0:
            print_warning(
                "VelAndLambdaTrainer disables rollout eval by default for residual-only "
                "training. Set algorithm.loss.disable_rollout_eval=false to re-enable."
            )
            self.eval_interval = 0

        if self.use_supervised_loss:
            print_warning(
                "use_supervised_loss=true is reserved for a later hybrid milestone. "
                "Only residual loss is active right now."
            )

    def _compose_total_loss(self, residual_loss: torch.Tensor, supervised_loss: torch.Tensor = None):
        """Single loss aggregator to keep future hybrid mode easy to add."""
        total_loss = self.residual_loss_weight * residual_loss
        if self.use_supervised_loss:
            if supervised_loss is None:
                raise RuntimeError(
                    "use_supervised_loss=true but no supervised_loss was provided."
                )
            total_loss = total_loss + self.supervised_loss_weight * supervised_loss
        return total_loss

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
            axion_contacts_last[key] = value[:, -1, :]

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

    def compute_loss(self, data, train):
        del train  # compatibility with SequenceModelTrainer.one_epoch
        prediction = self.neural_model(data)
        vel_prediction, lambda_prediction = self._split_prediction_last_step(prediction)

        states = data["states"]
        states_last = states[:, -1, :]
        axion_contacts_last = self._extract_last_step_axion_contacts(data)

        batch_worlds = vel_prediction.shape[0]
        worlds = self._dims.num_worlds
        if batch_worlds != worlds:
            raise RuntimeError(
                "Residual training requires batch size to equal engine worlds: "
                f"batch={batch_worlds}, num_worlds={worlds}."
            )

        validate_warm_start_shapes(self._dims, vel_prediction, lambda_prediction)
        self._load_engine_step_from_states_and_contacts(states_last, axion_contacts_last)
        residual_loss = compute_residual_loss(self._engine, vel_prediction, lambda_prediction)
        total_loss = self._compose_total_loss(residual_loss=residual_loss, supervised_loss=None)

        with torch.no_grad():
            loss_itemized = {
                "residual_loss": residual_loss.detach(),
                "total_loss": total_loss.detach(),
            }
        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)
