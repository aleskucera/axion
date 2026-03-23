from typing import Optional, Dict

import warp as wp
import torch
from torch_geometric.data import HeteroData

from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
from axion.math import integrate_body_pose_kernel
from .engine_config import GNNEngineConfig
from .logging_config import LoggingConfig
from axion.gnn.graph_builder import build_graph


class GNNEngine(AxionEngineBase):
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[GNNEngineConfig] = GNNEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)
        self._torch_device = torch.device(wp.device_to_torch(model.device))
        self.gnn_model = torch.load(
            config.model_path, map_location=self._torch_device, weights_only=False
        )
        self.gnn_model.eval()

    def step(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ) -> None:
        self.load_data(state_in, control, contacts, dt)
        graph = self.state_to_graph()
        with torch.no_grad():
            gnn_node_outputs, _ = self.gnn_model(
                graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict
            )
        self.graph_to_state(gnn_node_outputs, dt)
        self.compute_warm_start_forces()
        self._solve()
        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)

    def state_to_graph(self) -> HeteroData:
        device = self._torch_device
        num_bodies = self.dims.body_count

        body_vel = wp.to_torch(self.data.body_vel)
        body_vel_prev = wp.to_torch(self.data.body_vel_prev)
        body_pose_prev = wp.to_torch(self.data.body_pose_prev)
        ext_force = wp.to_torch(self.data.ext_force)
        body_mass = wp.to_torch(self.axion_model.body_mass).unsqueeze(2)
        body_inertia = wp.to_torch(self.axion_model.body_inertia)
        body_com = wp.to_torch(self.axion_model.body_com)

        contact_count = wp.to_torch(self.axion_contacts.contact_count).to(torch.long)
        contact_point0 = wp.to_torch(self.axion_contacts.contact_point0)
        contact_point1 = wp.to_torch(self.axion_contacts.contact_point1)
        contact_normal = wp.to_torch(self.axion_contacts.contact_normal)
        contact_shape0 = wp.to_torch(self.axion_contacts.contact_shape0).to(torch.long)
        contact_shape1 = wp.to_torch(self.axion_contacts.contact_shape1).to(torch.long)
        shape_material_mu = wp.to_torch(self.axion_model.shape_material_mu)
        contact_thickness0 = wp.to_torch(self.axion_contacts.contact_thickness0)
        contact_thickness1 = wp.to_torch(self.axion_contacts.contact_thickness1)
        shape_body = wp.to_torch(self.axion_model.shape_body).to(torch.long)

        graph = build_graph(
            body_vel,
            body_mass,
            ext_force,
            body_pose_prev,
            body_inertia,
            body_com,
            contact_count,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            shape_material_mu,
            contact_thickness0,
            contact_thickness1,
            num_bodies,
            device,
            shape_body,
        )

        return graph

    def graph_to_state(
        self,
        gnn_node_outputs: Dict[str, torch.Tensor],
        dt: float,
    ) -> None:
        if "object" in gnn_node_outputs:
            pred_vel = gnn_node_outputs["object"]
            num_bodies = self.dims.body_count
            num_worlds = self.dims.num_worlds
            pred_vel = pred_vel.reshape(num_worlds, num_bodies, 6)
            pred_vel = wp.from_torch(pred_vel.contiguous())
            wp.copy(dest=self.data.body_vel, src=pred_vel)

            wp.launch(
                kernel=integrate_body_pose_kernel,
                dim=(num_worlds, num_bodies),
                inputs=[
                    self.data.body_vel,
                    self.data.body_pose_prev,
                    self.axion_model.body_com,
                    dt,
                ],
                outputs=[
                    self.data.body_pose,
                ],
                device=self.device,
            )

    def _integrate_rotation(
        self, angular_vel: torch.Tensor, quat_prev: torch.Tensor, dt: float
    ) -> torch.Tensor:
        omega_magnitude = torch.norm(angular_vel, dim=-1, keepdim=True)
        theta = omega_magnitude * dt

        axis = torch.zeros_like(angular_vel)
        mask = omega_magnitude.squeeze(-1) > 1e-8
        axis[mask] = angular_vel[mask] / omega_magnitude[mask]

        half_theta = theta / 2
        dq_xyz = torch.sin(half_theta) * axis
        dq_w = torch.cos(half_theta)
        dq = torch.cat([dq_xyz, dq_w], dim=-1)

        quat_new = self._quaternion_multiply(dq, quat_prev)
        quat_new = quat_new / torch.linalg.norm(quat_new, dim=-1, keepdim=True)

        return quat_new

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        return torch.stack([x, y, z, w], dim=-1)
