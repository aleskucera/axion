import torch
import torch.nn.functional as F

from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer


class MSETrainer(SequenceModelTrainer):
    """Supervised MSE trainer for joint regression of [q, qd, lambda].

    Uses 1 − cos loss for angular q coordinates and MSE for qd and lambda.
    State and lambda prediction types (relative/absolute) are configured
    independently via env.utils_provider_cfg in the YAML.
    """

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        if not hasattr(self.neural_model, "regression_head"):
            raise ValueError("MSETrainer requires model attribute `regression_head`.")

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.state_loss_weight = float(loss_cfg.get("state_loss_weight", 1.0))
        self.lambda_loss_weight = float(loss_cfg.get("lambda_loss_weight", 1.0))
        self.lambda_log_space_loss = bool(loss_cfg.get("lambda_log_space_loss", False))
        self.lambda_log_space_eps = float(loss_cfg.get("lambda_log_space_eps", 1e-6))

        self._state_prediction_type = getattr(self.utils_provider, "state_prediction_type", "relative")
        self._lambda_prediction_type = getattr(self.utils_provider, "lambda_prediction_type", "relative")

        self._dof_q_per_env = int(self.neural_env.dof_q_per_env)

    def _convert_regression_to_absolute(
        self, data: dict, regression_prediction: torch.Tensor
    ) -> torch.Tensor:
        """Convert state and lambda slices to absolute values independently."""
        state_dim = self.utils_provider.state_prediction_dim
        state_pred = regression_prediction[..., :state_dim]
        lambda_pred = regression_prediction[..., state_dim:]

        if self._state_prediction_type == "relative":
            next_states = self.utils_provider.convert_prediction_to_next_states(
                data["states"], state_pred
            )
        else:
            next_states = state_pred

        if self._lambda_prediction_type == "relative":
            next_lambdas = self.utils_provider.convert_prediction_to_next_lambdas(
                data["lambdas"], lambda_pred
            )
        else:
            next_lambdas = lambda_pred

        return torch.cat([next_states, next_lambdas], dim=-1)

    def _build_regression_target(self, data: dict, prediction: torch.Tensor) -> torch.Tensor:
        regression_target = torch.cat([data["next_states"], data["next_lambdas"]], dim=-1)
        if regression_target.shape != prediction.shape:
            raise RuntimeError(
                "Regression target/prediction shape mismatch: "
                f"target={tuple(regression_target.shape)}, "
                f"prediction={tuple(prediction.shape)}."
            )
        return regression_target

    def _compute_state_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Angular q uses 1 − cos (wrap-safe); qd uses MSE."""
        state_dim = self.utils_provider.state_prediction_dim
        q_dim = self._dof_q_per_env
        pred_state = prediction[..., :state_dim]
        tgt_state = target[..., :state_dim]
        q_loss = torch.mean(1.0 - torch.cos(pred_state[..., :q_dim] - tgt_state[..., :q_dim]))
        qd_loss = F.mse_loss(pred_state[..., q_dim:], tgt_state[..., q_dim:])
        return q_loss + qd_loss

    def _signed_log(self, x: torch.Tensor) -> torch.Tensor:
        """sign(x) * ln(|x| + eps) — continuous at zero, preserves sign."""
        return x.sign() * (x.abs() + self.lambda_log_space_eps).log()

    def _compute_lambda_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        state_dim = self.utils_provider.state_prediction_dim
        pred_lambda = prediction[..., state_dim:]
        tgt_lambda = target[..., state_dim:]
        if self.lambda_log_space_loss:
            pred_lambda = self._signed_log(pred_lambda)
            tgt_lambda = self._signed_log(tgt_lambda)
        return F.mse_loss(pred_lambda, tgt_lambda)

    def compute_loss(self, data, train):
        del train
        regression_prediction = self.neural_model(data)

        regression_prediction = self._convert_regression_to_absolute(data, regression_prediction)
        regression_target = self._build_regression_target(data, regression_prediction)

        state_loss = self._compute_state_loss(regression_prediction, regression_target)
        lambda_loss = self._compute_lambda_loss(regression_prediction, regression_target)
        total_loss = self.state_loss_weight * state_loss + self.lambda_loss_weight * lambda_loss

        with torch.no_grad():
            loss_itemized = {
                "state_loss": state_loss.detach(),
                "lambda_loss": lambda_loss.detach(),
                "weighted_state_loss": (self.state_loss_weight * state_loss).detach(),
                "weighted_lambda_loss": (self.lambda_loss_weight * lambda_loss).detach(),
                "total_loss": total_loss.detach(),
            }
        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)
