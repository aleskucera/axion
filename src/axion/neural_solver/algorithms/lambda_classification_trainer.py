import torch
import torch.nn.functional as F

from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
from axion.neural_solver.utils.python_utils import print_warning


class LambdaClassificationTrainer(SequenceModelTrainer):
    """Training-focused lambda classifier built on top of SequenceModelTrainer."""

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        if not self.has_lambda_head:
            raise ValueError(
                "LambdaClassificationTrainer requires network.enable_lambda_head: true."
            )

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.label_key = str(loss_cfg.get("classification_label_key", "lambda_activity"))
        self.require_dataset_labels = bool(loss_cfg.get("require_dataset_labels", True))
        self.classification_threshold = float(loss_cfg.get("classification_threshold", 1e-3))
        self.classification_prob_threshold = float(
            loss_cfg.get("classification_prob_threshold", 0.5)
        )
        self.classification_eps = float(loss_cfg.get("classification_eps", 1e-8))
        self.positive_class_weight = float(loss_cfg.get("positive_class_weight", 1.0))

        if cfg["algorithm"].get("eval_interval", 1) > 0:
            print_warning(
                "LambdaClassificationTrainer disables rollout eval for this milestone. "
                "Forcing algorithm.eval_interval = 0."
            )
            self.eval_interval = 0

        self._bce_pos_weight = torch.tensor(
            self.positive_class_weight, device=self.device, dtype=torch.float32
        )

    def _build_lambda_labels(self, data: dict, logits: torch.Tensor) -> torch.Tensor:
        if self.label_key in data:
            labels = data[self.label_key].to(dtype=logits.dtype)
        else:
            raise KeyError(
                f"Batch is missing `{self.label_key}`. "
                "This trainer expects per-step class labels to be present in the dataset."
            )

        if labels.shape != logits.shape:
            raise RuntimeError(
                "Label/logit shape mismatch: "
                f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}. "
                "Classification head output dim must match the label tensor dim."
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
            "lambda_cls_accuracy": accuracy.detach(),
            "lambda_cls_precision": precision.detach(),
            "lambda_cls_recall": recall.detach(),
            "lambda_cls_f1": f1.detach(),
            "lambda_cls_target_positive_rate": targets.float().mean().detach(),
            "lambda_cls_pred_positive_rate": preds.float().mean().detach(),
        }

    def compute_loss(self, data, train):
        del train  # kept for compatibility with SequenceModelTrainer.one_epoch
        prediction = self.neural_model(data)
        lambda_logits = prediction["lambda"]
        if lambda_logits is None:
            raise RuntimeError("Lambda classification expects model to return lambda logits.")

        labels = self._build_lambda_labels(data, lambda_logits)
        loss = F.binary_cross_entropy_with_logits(
            lambda_logits,
            labels,
            pos_weight=self._bce_pos_weight,
        )

        with torch.no_grad():
            loss_itemized = {
                "lambda_cls_bce": loss.detach(),
            }
            loss_itemized.update(self._classification_metrics(lambda_logits, labels))
        return loss, loss_itemized

    def compute_test_loss_reference(self, data):
        raise NotImplementedError(
            "Test/eval branch is not implemented yet for LambdaClassificationTrainer."
        )
