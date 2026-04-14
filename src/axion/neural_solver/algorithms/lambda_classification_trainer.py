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
        self.classification_loss_type = str(loss_cfg.get("classification_loss_type", "bce_logits")).lower()
        self.classification_num_classes = int(loss_cfg.get("classification_num_classes", 2))

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
            labels = data[self.label_key]
        else:
            raise KeyError(
                f"Batch is missing `{self.label_key}`. "
                "This trainer expects per-step class labels to be present in the dataset."
            )

        if self.classification_num_classes == 2:
            labels = labels.to(dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise RuntimeError(
                    "Label/logit shape mismatch (binary): "
                    f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}. "
                    "For binary classification, labels must match logits shape."
                )
            return labels

        # Multiclass: logits expected (B,T,num_channels,num_classes) and labels (B,T,num_channels).
        if logits.ndim != 4 or logits.shape[-1] != self.classification_num_classes:
            raise RuntimeError(
                "Unexpected multiclass logits shape: "
                f"logits={tuple(logits.shape)}, expected (B,T,C,{self.classification_num_classes})."
            )

        if labels.ndim == 4 and labels.shape[-1] == 1:
            labels = labels[..., 0]
        if labels.ndim != 3:
            raise RuntimeError(
                "Unexpected multiclass label shape: "
                f"labels={tuple(labels.shape)}, expected (B,T,C)."
            )
        if labels.shape[0] != logits.shape[0] or labels.shape[1] != logits.shape[1] or labels.shape[2] != logits.shape[2]:
            raise RuntimeError(
                "Label/logit shape mismatch (multiclass): "
                f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}."
            )

        labels = labels.to(dtype=torch.long)
        with torch.no_grad():
            min_label = int(labels.min().item())
            max_label = int(labels.max().item())
        if min_label < 0 or max_label >= self.classification_num_classes:
            raise RuntimeError(
                "Multiclass labels out of range: "
                f"min={min_label}, max={max_label}, expected in [0,{self.classification_num_classes - 1}]."
            )
        return labels

    def _classification_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        # Multiclass classification
        if self.classification_num_classes > 2:
            # logits: (B,T,C,K), labels: (B,T,C)
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == labels).float().mean()

            out = {
                "lambda_cls_accuracy": accuracy.detach(),
            }

            # Macro precision/recall/F1 across classes.
            # Flatten over (B,T,C).
            preds_1d = preds.reshape(-1)
            labels_1d = labels.reshape(-1)
            eps = float(self.classification_eps)
            precision_per_class = []
            recall_per_class = []
            f1_per_class = []
            for k in range(self.classification_num_classes):
                pred_k = preds_1d == k
                label_k = labels_1d == k
                tp = (pred_k & label_k).float().sum()
                fp = (pred_k & (~label_k)).float().sum()
                fn = ((~pred_k) & label_k).float().sum()

                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2.0 * precision * recall / (precision + recall + eps)
                precision_per_class.append(precision)
                recall_per_class.append(recall)
                f1_per_class.append(f1)

                out[f"lambda_cls_precision_class_{k}"] = precision.detach()
                out[f"lambda_cls_recall_class_{k}"] = recall.detach()
                out[f"lambda_cls_f1_class_{k}"] = f1.detach()

            out["lambda_cls_precision_macro"] = torch.stack(precision_per_class).mean().detach()
            out["lambda_cls_recall_macro"] = torch.stack(recall_per_class).mean().detach()
            out["lambda_cls_f1_macro"] = torch.stack(f1_per_class).mean().detach()

            for k in range(self.classification_num_classes):
                out[f"lambda_cls_target_class_rate_{k}"] = (labels == k).float().mean().detach()
                out[f"lambda_cls_pred_class_rate_{k}"] = (preds == k).float().mean().detach()
            return out

        # Binary classification
        else:
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
        if self.classification_num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(
                lambda_logits,
                labels,
                pos_weight=self._bce_pos_weight,
            )
        else:
            # Cross entropy expects (N,K) logits and (N,) integer targets.
            # Flatten over (B,T,C).
            logits_2d = lambda_logits.reshape(-1, self.classification_num_classes)
            labels_1d = labels.reshape(-1)
            loss = F.cross_entropy(logits_2d, labels_1d)

        with torch.no_grad():
            loss_itemized = {}
            if self.classification_num_classes == 2:
                loss_itemized["lambda_cls_bce"] = loss.detach()
            else:
                loss_itemized["lambda_cls_ce"] = loss.detach()
            loss_itemized.update(self._classification_metrics(lambda_logits, labels))
        return loss, loss_itemized

    def compute_test_loss_reference(self, data):
        raise NotImplementedError(
            "Test/eval branch is not implemented yet for LambdaClassificationTrainer."
        )
