from typing import Optional, Dict

import warp as wp
import torch
from torch_geometric.data import HeteroData

from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
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
            gnn_node_outputs, _ = self.gnn_model(graph)
        self.graph_to_state(gnn_node_outputs, state_in, state_out, dt)

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
            body_vel_prev,
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
        state_in: State,
        state_out: State,
        dt: float,
    ) -> None:
        if "object" in gnn_node_outputs:
            pred_vel = gnn_node_outputs["object"]
            pred_vel_wp = wp.from_torch(pred_vel.contiguous())
            wp.copy(dest=state_out.body_qd, src=pred_vel_wp)

            in_pose = wp.to_torch(state_in.body_q)
            out_pose = wp.to_torch(state_out.body_q)
            out_pose[:, :3] = in_pose[:, :3] + pred_vel[:, 3:6] * dt
            out_pose[:, 3:] = in_pose[:, 3:]
