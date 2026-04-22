import torch


def angular_prediction_loss(
    pred_q: torch.Tensor,
    tgt_q: torch.Tensor,
    angular_prediction_l2_weight: float = 0.5,
) -> torch.Tensor:
    """Wrap-safe angular loss with prediction-magnitude regularization.

    Loss = mean(1 - cos(pred_q - tgt_q)) + w * mean(pred_q^2)
    """
    if pred_q.shape != tgt_q.shape:
        raise RuntimeError(
            "Angular loss shape mismatch: "
            f"pred_q={tuple(pred_q.shape)}, tgt_q={tuple(tgt_q.shape)}."
        )

    periodic_term = torch.mean(1.0 - torch.cos(pred_q - tgt_q))
    prediction_l2_term = torch.mean(pred_q.square())
    return periodic_term + float(angular_prediction_l2_weight) * prediction_l2_term
