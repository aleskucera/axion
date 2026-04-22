import torch
import torch.nn.functional as F

from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
from axion.neural_solver.utils.loss_utils import angular_prediction_loss


class MTLTrainer(SequenceModelTrainer):
    """Supervised MTL trainer for joint state regression, lambda regression, and jump classification."""

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        for attr in ("cls_head", "base_head", "jump_head", "state_head"):
            if not hasattr(self.neural_model, attr):
                raise ValueError(
                    f"MTLTrainer requires model attribute `{attr}`."
                )

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.state_head_loss_weight = float(loss_cfg.get("state_head_loss_weight", 1.0))
        self.base_head_loss_weight = float(loss_cfg.get("base_head_loss_weight", 1.0))
        self.cls_head_loss_weight = float(loss_cfg.get("cls_head_loss_weight", 1.0))
        self.jump_head_loss_weight = float(loss_cfg.get("jump_head_loss_weight", 1.0))
        self.classification_eps = float(loss_cfg.get("classification_eps", 1e-8))
        self.positive_class_weight = float(loss_cfg.get("positive_class_weight", 1.0))
        self.classification_prob_threshold = float(loss_cfg.get("classification_prob_threshold", 0.5))
        self.classification_loss_type = str(loss_cfg.get("classification_loss_type", "bce_logits")).lower()
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.angular_prediction_l2_weight = float(loss_cfg.get("angular_prediction_l2_weight", 0.5))
        self.jump_target_scale = float(loss_cfg.get("jump_target_scale", 100.0))
        self.neural_model.jump_target_scale = self.jump_target_scale

        self._bce_pos_weight = torch.tensor(self.positive_class_weight, device=self.device, dtype=torch.float32)

        self._dof_q_per_env = int(self.neural_env.dof_q_per_env)

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    def _convert_state_to_absolute(self, data: dict, state_pred: torch.Tensor) -> torch.Tensor:
        """Convert state head output to absolute next-state values."""
        if self.utils_provider.state_prediction_type == "relative":
            return self.utils_provider.convert_prediction_to_next_states(
                data["states"], state_pred
            )
        return state_pred

    def _convert_lambda_to_absolute(self, data: dict, lambda_pred: torch.Tensor) -> torch.Tensor:
        """Convert lambda head output to absolute next-lambda values."""
        if self.utils_provider.lambda_prediction_type == "relative":
            return self.utils_provider.convert_prediction_to_next_lambdas(
                data["lambdas"], lambda_pred
            )
        return lambda_pred

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _compute_state_regression_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Angular loss on generalised-coordinate q, MSE on the remaining dims (qd)."""
        q_dim = self._dof_q_per_env
        q_loss = angular_prediction_loss(
            prediction[..., :q_dim],
            target[..., :q_dim],
            angular_prediction_l2_weight=self.angular_prediction_l2_weight,
        )
        rest_loss = F.mse_loss(prediction[..., q_dim:], target[..., q_dim:])
        return q_loss + rest_loss

    def _build_classification_labels(self, data: dict, logits: torch.Tensor) -> torch.Tensor:
        if "lambda_activity" not in data:
            raise KeyError(
                "Batch is missing `lambda_activity`. "
                "MTLTrainer expects per-step class labels in the dataset."
            )
        labels = data["lambda_activity"].to(dtype=logits.dtype)
        if labels.shape != logits.shape:
            raise RuntimeError(
                "Label/logit shape mismatch: "
                f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}."
            )
        return labels

    def _classification_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        probs = torch.sigmoid(logits)
        preds = probs >= self.classification_prob_threshold
        targets = labels >= 0.5
        tp = (preds & targets).float().sum()
        fp = (preds & (~targets)).float().sum()
        fn = ((~preds) & targets).float().sum()
        eps = self.classification_eps
        accuracy = (preds == targets).float().mean()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        return {
            "classification_accuracy": accuracy.detach(),
            "classification_precision": precision.detach(),
            "classification_recall": recall.detach(),
            "classification_f1": f1.detach(),
        }

    def _focal_loss_with_logits(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Binary focal loss without alpha balancing."""
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        loss = focal_weight * bce
        return loss.mean()

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def compute_loss(self, data, train):
        del train
        prediction = self.neural_model(data)

        # 1. State regression loss
        state_abs = self._convert_state_to_absolute(data, prediction["state"])
        state_target = data["next_states"]
        if state_abs.shape != state_target.shape:
            raise RuntimeError(
                "State target/prediction shape mismatch: "
                f"target={tuple(state_target.shape)}, prediction={tuple(state_abs.shape)}."
            )
        state_loss = self._compute_state_regression_loss(state_abs, state_target)

        # 2. Classification loss (binary BCE or focal BCE)
        logits = prediction["logits"]
        cls_labels = self._build_classification_labels(data, logits)
        if self.classification_loss_type == "bce_logits":
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, cls_labels, pos_weight=self._bce_pos_weight
            )
        elif self.classification_loss_type == "focal_logits":
            cls_loss = self._focal_loss_with_logits(logits, cls_labels)

        # 3. Lambda regression loss — MSE on lambda_hat = base + p * jump
        lambda_abs = self._convert_lambda_to_absolute(data, prediction["lambda_hat"])
        lambda_target = data["next_lambdas"]
        if lambda_abs.shape != lambda_target.shape:
            raise RuntimeError(
                "Lambda target/prediction shape mismatch: "
                f"target={tuple(lambda_target.shape)}, prediction={tuple(lambda_abs.shape)}."
            )
        lambda_loss = F.mse_loss(lambda_abs, lambda_target)

        # 4. Jump loss — weighted SmoothL1:
        #    active labels use SmoothL1(prediction, jump_gt),
        #    inactive labels use 0.01 * SmoothL1(prediction, 0).
        jump_gt = (data["next_lambdas"] - data["lambdas"]) / self.jump_target_scale
        jump_pred = prediction["jump"]
        jump_active = F.smooth_l1_loss(jump_pred, jump_gt, reduction="none")
        jump_inactive = F.smooth_l1_loss(jump_pred, torch.zeros_like(jump_pred), reduction="none")
        jump_loss = (
            cls_labels * jump_active
            + (1.0 - cls_labels) * 0.01 * jump_inactive
        ).mean()

        total_loss = (
            self.state_head_loss_weight * state_loss
            + self.base_head_loss_weight * lambda_loss
            + self.cls_head_loss_weight * cls_loss
            + self.jump_head_loss_weight * jump_loss
        )

        with torch.no_grad():
            loss_itemized = {
                "state_loss": state_loss.detach(),
                "lambda_loss": lambda_loss.detach(),
                "classification_loss": cls_loss.detach(),
                "jump_loss": jump_loss.detach(),
                "weighted_state_loss": (self.state_head_loss_weight * state_loss).detach(),
                "weighted_lambda_loss": (self.base_head_loss_weight * lambda_loss).detach(),
                "weighted_classification_loss": (self.cls_head_loss_weight * cls_loss).detach(),
                "weighted_jump_loss": (self.jump_head_loss_weight * jump_loss).detach(),
                "total_loss": total_loss.detach(),
            }
            loss_itemized.update(self._classification_metrics(logits, cls_labels))

        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)
