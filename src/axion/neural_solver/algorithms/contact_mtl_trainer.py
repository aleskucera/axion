import torch
import torch.nn.functional as F

from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer


class ContactMTLTrainer(SequenceModelTrainer):
    """Supervised MTL trainer for joint contact classification and contact lambda regression.

    Target transform (training):
        z = asinh(next_lambdas[..., start:]) if use_asinh_transform else next_lambdas[..., start:]
        t = (z - μ) / σ   if normalize_output else z
    Loss: MSE(p, t)

    Inverse transform (inference, inside ContactMTLModel.evaluate):
        y = sinh(p * σ + μ) if use_asinh_transform else (p * σ + μ)   when normalize_output is enabled
        y = sinh(p) if use_asinh_transform else p                     when normalize_output is disabled
    """

    def __init__(self, neural_env, cfg, model_checkpoint_path=None, device="cuda:0"):
        # Must be set before super().__init__() because Python's dynamic dispatch
        # means preprocess_data_batch (overridden below) is called during
        # compute_dataset_statistics inside super().__init__().
        self.contact_lambda_start_index = int(cfg["network"].get("contact_lambda_start_index", 10))
        self.use_asinh_transform = bool(cfg["network"].get("use_asinh_transform", True))

        super().__init__(
            neural_env=neural_env,
            cfg=cfg,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

        for attr in ("cls_head", "contact_lambda_head"):
            if not hasattr(self.neural_model, attr):
                raise ValueError(
                    f"ContactMTLTrainer requires model attribute `{attr}`."
                )

        # Wire transformed-target RMS to the model (overwrites the base-class set_output_rms call,
        # which passed target/lambda_target RMS that are unused in this pipeline).
        if hasattr(self, "dataset_rms"):
            self.neural_model.set_output_rms(
                contact_lambda_transform_rms=self.dataset_rms.get("contact_lambda_regression_target")
            )

        loss_cfg = cfg["algorithm"].get("loss", {}) or {}
        self.cls_head_loss_weight = float(loss_cfg.get("cls_head_loss_weight", 1.0))
        self.lambda_head_loss_weight = float(loss_cfg.get("lambda_head_loss_weight", 1.0))
        self.classification_eps = float(loss_cfg.get("classification_eps", 1e-8))
        self.positive_class_weight = float(loss_cfg.get("positive_class_weight", 1.0))
        self.classification_prob_threshold = float(loss_cfg.get("classification_prob_threshold", 0.5))
        self.classification_loss_type = str(loss_cfg.get("classification_loss_type", "bce_logits")).lower()
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

        self._bce_pos_weight = torch.tensor(self.positive_class_weight, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def preprocess_data_batch(self, data):
        data = super().preprocess_data_batch(data)
        contact_lambdas = data["next_lambdas"][..., self.contact_lambda_start_index:]
        if self.use_asinh_transform:
            data["contact_lambda_regression_target"] = torch.asinh(contact_lambdas)
        else:
            data["contact_lambda_regression_target"] = contact_lambdas
        return data

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _build_classification_labels(self, data: dict, logits: torch.Tensor) -> torch.Tensor:
        if "lambda_activity" not in data:
            raise KeyError(
                "Batch is missing `lambda_activity`. "
                "ContactMTLTrainer expects per-step class labels in the dataset."
            )
        labels = data["lambda_activity"][..., self.contact_lambda_start_index:].to(dtype=logits.dtype)
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

        # 1. Classification loss (binary BCE or focal BCE)
        logits = prediction["logits"]
        cls_labels = self._build_classification_labels(data, logits)
        if self.classification_loss_type == "bce_logits":
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, cls_labels, pos_weight=self._bce_pos_weight
            )
        elif self.classification_loss_type == "focal_logits":
            cls_loss = self._focal_loss_with_logits(logits, cls_labels)
        else:
            raise ValueError(
                f"Unknown classification_loss_type: {self.classification_loss_type!r}. "
                "Expected 'bce_logits' or 'focal_logits'."
            )

        # 2. Contact lambda regression loss — MSE in transformed (optionally normalized) space.
        #    The model outputs raw predictions p; targets are transformed and optionally
        #    normalized to zero-mean/unit-variance per channel.
        regression_target = data["contact_lambda_regression_target"]
        if self.neural_model.normalize_output and self.neural_model.contact_lambda_transform_rms is not None:
            lambda_target = self.neural_model.contact_lambda_transform_rms.normalize(regression_target)
        else:
            lambda_target = regression_target

        contact_lambda_pred = prediction["contact_lambda_hat"]
        if contact_lambda_pred.shape != lambda_target.shape:
            raise RuntimeError(
                "Contact lambda target/prediction shape mismatch: "
                f"target={tuple(lambda_target.shape)}, prediction={tuple(contact_lambda_pred.shape)}."
            )
        lambda_loss = F.mse_loss(contact_lambda_pred, lambda_target)

        total_loss = (
            self.cls_head_loss_weight * cls_loss
            + self.lambda_head_loss_weight * lambda_loss
        )

        with torch.no_grad():
            loss_itemized = {
                "classification_loss": cls_loss.detach(),
                "lambda_loss": lambda_loss.detach(),
                "weighted_classification_loss": (self.cls_head_loss_weight * cls_loss).detach(),
                "weighted_lambda_loss": (self.lambda_head_loss_weight * lambda_loss).detach(),
                "total_loss": total_loss.detach(),
            }
            loss_itemized.update(self._classification_metrics(logits, cls_labels))

        return total_loss, loss_itemized

    def compute_test_loss_reference(self, data):
        return self.compute_loss(data, train=False)
