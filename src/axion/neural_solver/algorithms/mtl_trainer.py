import torch
import torch.nn.functional as F

from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer


class MTLTrainer(SequenceModelTrainer):
    """Supervised MTL trainer for joint regression + lambda classification."""

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        if not hasattr(self.neural_model, "regression_head"):
            raise ValueError(
                "MTLTrainer requires model attribute `regression_head`."
            )
        if not hasattr(self.neural_model, "classification_head"):
            raise ValueError(
                "MTLTrainer requires model attribute `classification_head`."
            )

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.regression_loss_weight = float(loss_cfg.get("regression_loss_weight", 1.0))
        self.classification_loss_weight = float(loss_cfg.get("classification_loss_weight", 1.0))
        self.classification_num_classes = int(loss_cfg.get("classification_num_classes", 2))
        self.classification_eps = float(loss_cfg.get("classification_eps", 1e-8))
        self.positive_class_weight = float(loss_cfg.get("positive_class_weight", 1.0))
        self.classification_prob_threshold = float(loss_cfg.get("classification_prob_threshold", 0.5))

        self._bce_pos_weight = torch.tensor(self.positive_class_weight, device=self.device, dtype=torch.float32)

    def _build_regression_target(self, data: dict, prediction: torch.Tensor) -> torch.Tensor:
        regression_target = torch.cat([data["next_states"], data["next_lambdas"]], dim=-1)
        if regression_target.shape != prediction.shape:
            raise RuntimeError(
                "Regression target/prediction shape mismatch: "
                f"target={tuple(regression_target.shape)}, "
                f"prediction={tuple(prediction.shape)}."
            )
        return regression_target

    def _build_classification_labels(self, data: dict, logits: torch.Tensor) -> torch.Tensor:
        if "lambda_activity" not in data:
            raise KeyError(
                "Batch is missing `lambda_activity`. "
                "MTLTrainer expects per-step class labels in the dataset."
            )
        labels = data["lambda_activity"]

        if self.classification_num_classes == 2:
            labels = labels.to(dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise RuntimeError(
                    "Label/logit shape mismatch (binary): "
                    f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}."
                )
            return labels

        if logits.ndim != 4 or logits.shape[-1] != self.classification_num_classes:
            raise RuntimeError(
                "Unexpected multiclass logits shape: "
                f"logits={tuple(logits.shape)}, "
                f"expected (B,T,C,{self.classification_num_classes})."
            )
        if labels.ndim == 4 and labels.shape[-1] == 1:
            labels = labels[..., 0]
        if labels.ndim != 3:
            raise RuntimeError(
                "Unexpected multiclass label shape: "
                f"labels={tuple(labels.shape)}, expected (B,T,C)."
            )
        if (
            labels.shape[0] != logits.shape[0]
            or labels.shape[1] != logits.shape[1]
            or labels.shape[2] != logits.shape[2]
        ):
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
                f"min={min_label}, max={max_label}, expected in "
                f"[0,{self.classification_num_classes - 1}]."
            )
        return labels

    def _classification_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        if self.classification_num_classes > 2:
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == labels).float().mean()
            out = {"classification_accuracy": accuracy.detach()}

            preds_1d = preds.reshape(-1)
            labels_1d = labels.reshape(-1)
            eps = self.classification_eps
            precision_per_class = []
            recall_per_class = []
            f1_per_class = []
            for cls_idx in range(self.classification_num_classes):
                pred_cls = preds_1d == cls_idx
                label_cls = labels_1d == cls_idx
                tp = (pred_cls & label_cls).float().sum()
                fp = (pred_cls & (~label_cls)).float().sum()
                fn = ((~pred_cls) & label_cls).float().sum()

                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2.0 * precision * recall / (precision + recall + eps)
                precision_per_class.append(precision)
                recall_per_class.append(recall)
                f1_per_class.append(f1)
                out[f"classification_precision_class_{cls_idx}"] = precision.detach()
                out[f"classification_recall_class_{cls_idx}"] = recall.detach()
                out[f"classification_f1_class_{cls_idx}"] = f1.detach()

            out["classification_precision_macro"] = torch.stack(precision_per_class).mean().detach()
            out["classification_recall_macro"] = torch.stack(recall_per_class).mean().detach()
            out["classification_f1_macro"] = torch.stack(f1_per_class).mean().detach()
            return out

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

    def compute_loss(self, data, train):
        del train
        prediction = self.neural_model(data)
        regression_prediction = prediction.get("regression")
        classification_logits = prediction.get("classification")
        if regression_prediction is None or classification_logits is None:
            raise RuntimeError(
                "MTLTrainer expects model outputs `regression` and `classification`."
            )

        regression_target = self._build_regression_target(data, regression_prediction)
        classification_labels = self._build_classification_labels(data, classification_logits)

        regression_loss = F.mse_loss(regression_prediction, regression_target)
        if self.classification_num_classes == 2:
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_logits,
                classification_labels,
                pos_weight=self._bce_pos_weight,
            )
        else:
            logits_2d = classification_logits.reshape(-1, self.classification_num_classes)
            labels_1d = classification_labels.reshape(-1)
            classification_loss = F.cross_entropy(logits_2d, labels_1d)

        total_loss = (
            self.regression_loss_weight * regression_loss
            + self.classification_loss_weight * classification_loss
        )

        with torch.no_grad():
            loss_itemized = {
                "regression_loss": regression_loss.detach(),
                "classification_loss": classification_loss.detach(),
                "weighted_regression_loss": (
                    self.regression_loss_weight * regression_loss
                ).detach(),
                "weighted_classification_loss": (
                    self.classification_loss_weight * classification_loss
                ).detach(),
                "total_loss": total_loss.detach(),
            }
            loss_itemized.update(
                self._classification_metrics(classification_logits, classification_labels)
            )
        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)
