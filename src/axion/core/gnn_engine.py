from typing import Callable, Optional, Dict, Tuple

import warp as wp
import torch
from torch_geometric.data import HeteroData
import numpy as np

from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase
from .model import AxionModel
from axion.gnn.dataset import NODE_FEATURE_DIMS, EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions [x, y, z, w] to rotation matrices [N, 3, 3]."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)
    return R


def transform_points(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Applies a batch of SE(3) poses to a batch of 3D points."""
    R = quat_to_rot_matrix(pose[:, 3:])
    # R: [N, 3, 3], points: [N, 3, 1] -> bmm -> [N, 3, 1] -> squeeze -> [N, 3]
    rotated = torch.bmm(R, points.unsqueeze(2)).squeeze(2)
    return rotated + pose[:, :3]


class GNNEngine(SolverBase):
    def __init__(self, model: Model, model_path: str):
        super().__init__(model)
        self._torch_device = torch.device(wp.device_to_torch(model.device))
        self.gnn_model = torch.load(model_path, map_location=self._torch_device, weights_only=False)
        self.gnn_model.eval()

    def state_to_graph(
        self, state_in: State, contacts: Contacts
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor], Dict[Tuple, torch.Tensor]]:

        device = self._torch_device
        # --- 1. Object Node Features ---
        body_vel = wp.to_torch(state_in.body_qd)
        body_mass = wp.to_torch(self.model.body_mass).unsqueeze(1)
        ext_forces = wp.to_torch(state_in.body_f)
        body_pose = wp.to_torch(state_in.body_q)
        body_inertia = wp.to_torch(self.model.body_inertia)
        body_com = wp.to_torch(self.model.body_com)

        R = quat_to_rot_matrix(body_pose[:, 3:])
        rot_inertia = torch.bmm(torch.bmm(R, body_inertia), R.transpose(1, 2)).reshape(-1, 9)

        x_object = torch.cat([body_vel, body_mass, ext_forces, rot_inertia], dim=1).float()

        x_dict = {
            "object": x_object,
            "floor": torch.zeros((1, 0), dtype=torch.float32, device=device),
            "contact_point": torch.empty((0, 0), dtype=torch.float32, device=device),
        }

        # Initialize Edge Dicts
        edge_index_dict = {
            edge_type: torch.zeros((2, 0), dtype=torch.long, device=device)
            for edge_type in EDGE_FEATURE_DIMS
        }
        edge_attr_dict = {
            edge_type: torch.zeros((0, dim), dtype=torch.float32, device=device)
            for edge_type, dim in EDGE_FEATURE_DIMS.items()
        }

        # --- 2. Contact Processing ---
        # Safeguard to ensure contacts exist this step
        num_contacts = contacts.body_a.shape[0] if hasattr(contacts, "body_a") else 0

        if num_contacts > 0:
            # Note: We create exactly 2 * num_contacts nodes.
            # We skip the O(N^2) distance-based node merging from dataset.py to keep the engine fast.
            x_dict["contact_point"] = torch.zeros(
                (num_contacts * 2, 0), dtype=torch.float32, device=device
            )

            # Extract Warp arrays to Torch
            body_a = wp.to_torch(contacts.body_a).to(device).long()
            body_b = wp.to_torch(contacts.body_b).to(device).long()
            point_a_local = wp.to_torch(contacts.point_a).to(device)
            point_b_local = wp.to_torch(contacts.point_b).to(device)
            normal = wp.to_torch(contacts.normal).to(device)  # Assuming normal points A -> B

            # NOTE: If your Newton Contacts array doesn't explicitly store thickness or friction mu,
            # you may need to map these from shape indices or use defaults.
            thicc_a = torch.zeros(num_contacts, device=device)  # Replace with actual if available
            thicc_b = torch.zeros(num_contacts, device=device)
            mu = (
                torch.ones(num_contacts, device=device) * 0.5
            )  # Replace with actual model material mapping

            # Map poses (Handle floor index -1 gracefully)
            pose_a = torch.zeros((num_contacts, 7), device=device)
            pose_a[:, 6] = 1.0  # Default to identity quaternion for floor
            mask_a = body_a >= 0
            pose_a[mask_a] = body_pose[body_a[mask_a]]

            pose_b = torch.zeros((num_contacts, 7), device=device)
            pose_b[:, 6] = 1.0
            mask_b = body_b >= 0
            pose_b[mask_b] = body_pose[body_b[mask_b]]

            # Transform local contact points and CoM to global space
            point_global_a = transform_points(pose_a, point_a_local)
            point_global_b = transform_points(pose_b, point_b_local)

            com_global_a = torch.zeros((num_contacts, 3), device=device)
            com_global_a[mask_a] = transform_points(pose_a[mask_a], body_com[body_a[mask_a]])

            com_global_b = torch.zeros((num_contacts, 3), device=device)
            com_global_b[mask_b] = transform_points(pose_b[mask_b], body_com[body_b[mask_b]])

            # Calculate Levers
            normal_a = -normal
            normal_b = normal

            point_adj_a = point_global_a - (thicc_a.unsqueeze(1) * normal_a)
            point_adj_b = point_global_b - (thicc_b.unsqueeze(1) * normal_b)

            lever_a = point_adj_a - com_global_a
            lever_b = point_adj_b - com_global_b

            lever_norm_a = torch.norm(lever_a, dim=1, keepdim=True)
            lever_norm_b = torch.norm(lever_b, dim=1, keepdim=True)

            contact_dist = torch.sum(normal_a * (point_adj_a - point_adj_b), dim=1, keepdim=True)

            # --- 3. Build Edges ---
            cp_indices_a = torch.arange(0, num_contacts, device=device)
            cp_indices_b = torch.arange(num_contacts, num_contacts * 2, device=device)

            # Object <-> Contact edges
            if mask_a.any():
                obj_idx_a = body_a[mask_a]
                cp_idx_a = cp_indices_a[mask_a]

                # ("object", "inter_object", "contact_point")
                edge_index_dict[("object", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_index_dict[("object", "inter_object", "contact_point")],
                        torch.stack([obj_idx_a, cp_idx_a], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("object", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_attr_dict[("object", "inter_object", "contact_point")],
                        torch.cat([lever_a[mask_a], lever_norm_a[mask_a]], dim=1),
                    ],
                    dim=0,
                )

                # ("contact_point", "inter_object", "object") - Note negative lever based on dataset.py
                edge_index_dict[("contact_point", "inter_object", "object")] = torch.cat(
                    [
                        edge_index_dict[("contact_point", "inter_object", "object")],
                        torch.stack([cp_idx_a, obj_idx_a], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("contact_point", "inter_object", "object")] = torch.cat(
                    [
                        edge_attr_dict[("contact_point", "inter_object", "object")],
                        torch.cat([-lever_a[mask_a], lever_norm_a[mask_a]], dim=1),
                    ],
                    dim=0,
                )

            # Floor -> Contact edges
            floor_mask_a = ~mask_a
            if floor_mask_a.any():
                cp_idx_floor_a = cp_indices_a[floor_mask_a]
                floor_idx_a = torch.zeros_like(cp_idx_floor_a)

                edge_index_dict[("floor", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_index_dict[("floor", "inter_object", "contact_point")],
                        torch.stack([floor_idx_a, cp_idx_floor_a], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("floor", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_attr_dict[("floor", "inter_object", "contact_point")],
                        torch.cat([lever_a[floor_mask_a], lever_norm_a[floor_mask_a]], dim=1),
                    ],
                    dim=0,
                )

            # (Repeat analogous blocks for mask_b / floor_mask_b here ... omitted for brevity but mirrors A exactly)
            if mask_b.any():
                obj_idx_b = body_b[mask_b]
                cp_idx_b = cp_indices_b[mask_b]

                edge_index_dict[("object", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_index_dict[("object", "inter_object", "contact_point")],
                        torch.stack([obj_idx_b, cp_idx_b], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("object", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_attr_dict[("object", "inter_object", "contact_point")],
                        torch.cat([lever_b[mask_b], lever_norm_b[mask_b]], dim=1),
                    ],
                    dim=0,
                )

                edge_index_dict[("contact_point", "inter_object", "object")] = torch.cat(
                    [
                        edge_index_dict[("contact_point", "inter_object", "object")],
                        torch.stack([cp_idx_b, obj_idx_b], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("contact_point", "inter_object", "object")] = torch.cat(
                    [
                        edge_attr_dict[("contact_point", "inter_object", "object")],
                        torch.cat([-lever_b[mask_b], lever_norm_b[mask_b]], dim=1),
                    ],
                    dim=0,
                )

            floor_mask_b = ~mask_b
            if floor_mask_b.any():
                cp_idx_floor_b = cp_indices_b[floor_mask_b]
                floor_idx_b = torch.zeros_like(cp_idx_floor_b)

                edge_index_dict[("floor", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_index_dict[("floor", "inter_object", "contact_point")],
                        torch.stack([floor_idx_b, cp_idx_floor_b], dim=0),
                    ],
                    dim=1,
                )
                edge_attr_dict[("floor", "inter_object", "contact_point")] = torch.cat(
                    [
                        edge_attr_dict[("floor", "inter_object", "contact_point")],
                        torch.cat([lever_b[floor_mask_b], lever_norm_b[floor_mask_b]], dim=1),
                    ],
                    dim=0,
                )

            # Contact <-> Contact edges
            cc_index_1 = torch.stack([cp_indices_a, cp_indices_b], dim=0)
            cc_index_2 = torch.stack([cp_indices_b, cp_indices_a], dim=0)

            cc_attr_1 = torch.cat([normal_a, contact_dist, mu.unsqueeze(1)], dim=1)
            cc_attr_2 = torch.cat([normal_b, contact_dist, mu.unsqueeze(1)], dim=1)

            edge_index_dict[("contact_point", "contact", "contact_point")] = torch.cat(
                [cc_index_1, cc_index_2], dim=1
            )
            edge_attr_dict[("contact_point", "contact", "contact_point")] = torch.cat(
                [cc_attr_1, cc_attr_2], dim=0
            )

        return x_dict, edge_index_dict, edge_attr_dict

    # graph_to_state and step remain the same as the previous iteration
    def graph_to_state(
        self,
        gnn_node_outputs: Dict[str, torch.Tensor],
        state_in: State,
        state_out: State,
        dt: float,
    ):
        if "object" in gnn_node_outputs:
            pred_vel = gnn_node_outputs["object"]
            pred_vel_wp = wp.from_torch(pred_vel.contiguous())
            wp.copy(state_out.body_qd, pred_vel_wp)

            in_pose = wp.to_torch(state_in.body_q)
            out_pose = wp.to_torch(state_out.body_q)
            out_pose[:, :3] = in_pose[:, :3] + pred_vel[:, 3:6] * dt
            out_pose[:, 3:] = in_pose[:, 3:]

    def step(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
        x_dict, edge_index_dict, edge_attr_dict = self.state_to_graph(state_in, contacts)
        with torch.no_grad():
            gnn_node_outputs, _ = self.gnn_model(x_dict, edge_index_dict, edge_attr_dict)
        self.graph_to_state(gnn_node_outputs, state_in, state_out, dt)
