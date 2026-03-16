from typing import Dict
import torch
import torch.nn.functional as F
from .dataset import EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS


class LossGNN(torch.nn.Module):
    def __init__(
        self,
        loss_name,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        if loss_name == "l1_loss":
            self.loss = self.l1_loss
        elif loss_name == "weighted_l1_loss":
            self.loss = self.weighted_l1_loss
        elif loss_name == "residue_loss":
            self.loss = self.residue_loss
        else:
            raise NotImplementedError(f"{loss_name} loss type not found")

    def forward(self, data, object_states, lambdas_dict):
        return self.loss(data, object_states, lambdas_dict)

    def l1_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        gt_values = torch.tensor([], device=self.device)
        pred_values = torch.tensor([], device=self.device)
        for key in OUTPUT_FEATURE_DIMS.keys():
            if isinstance(key, str) and key in data.node_types and key in object_states:
                gt_values = torch.cat([gt_values, data[key].y.flatten()])
                pred_values = torch.cat([pred_values, object_states[key].flatten()])
            elif key in data.edge_types and key in lambdas_dict:
                gt_values = torch.cat([gt_values, data[key].y.flatten()])
                pred_values = torch.cat([pred_values, lambdas_dict[key].flatten()])
        return {"total_loss": (F.l1_loss(pred_values, gt_values), gt_values.shape[0])}

    def weighted_l1_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        weight = torch.ones_like(pred_values)
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and data[edge_type].edge_index.shape[1] > 0:
                gt_edge = data[edge_type].y.flatten()
                pred_edge = lambdas_dict[edge_type].flatten()
                gt_values = torch.cat([gt_values, gt_edge])
                pred_values = torch.cat([pred_values, pred_edge])
                target_indices = data[edge_type].edge_index[1]
                target_masses = data["object"].x[target_indices, 6]
                dim = gt_edge.numel() // target_indices.numel()
                edge_weight = (1.0 / target_masses).repeat_interleave(dim)
                weight = torch.cat([weight, edge_weight])
        return {
            "total_loss": (F.l1_loss(pred_values, gt_values, weight=weight), gt_values.shape[0])
        }

    def residue_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"PINN loss not yet implemented")
