import torch

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

    def _load_engine_step_from_states(self, states_bt: torch.Tensor) -> None:
        """Set simulator state, rebuild contacts, then load engine buffers."""
        if states_bt.ndim != 2:
            raise RuntimeError(
                f"Expected states chunk as (B, state_dim), got shape={tuple(states_bt.shape)}."
            )
        if states_bt.shape[0] != self._dims.num_worlds:
            raise RuntimeError(
                f"Residual path requires chunk size == num_worlds ({self._dims.num_worlds}), "
                f"got batch chunk {states_bt.shape[0]}."
            )
        expected_state_dim = self.neural_env.state_dim
        if states_bt.shape[1] != expected_state_dim:
            raise RuntimeError(
                f"State dim mismatch: expected {expected_state_dim}, got {states_bt.shape[1]}."
            )

        warp_utils.assign_states_from_torch(self._sim_wrapper, states_bt)
        warp_utils.eval_fk(self._sim_wrapper.model, self._sim_wrapper.state)

        contacts = self._sim_wrapper.model.collide(self._sim_wrapper.state)
        self._engine.load_data(self._sim_wrapper.state, self._control, contacts, self._dt)

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
        if states.ndim != 3:
            raise RuntimeError(
                f"Expected data['states'] shape (B,T,D), got shape={tuple(states.shape)}."
            )
        states_last = states[:, -1, :]

        batch_worlds = vel_prediction.shape[0]
        worlds = self._dims.num_worlds
        if batch_worlds != worlds:
            raise RuntimeError(
                "Residual training requires batch size to equal engine worlds: "
                f"batch={batch_worlds}, num_worlds={worlds}."
            )

        validate_warm_start_shapes(self._dims, vel_prediction, lambda_prediction)
        self._load_engine_step_from_states(states_last)
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
