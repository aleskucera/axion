import torch
import warp as wp

from axion.core.contacts import AxionContacts
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.learning.torch_residual_ad import AxionResidualAD
from axion.math import integrate_body_pose_kernel

from .graph_builder import build_graph
from .network import AxionGNN


class GNNResidualTrainer:
    """Trains AxionGNN online using residual loss via AxionResidualAD.

    Mirrors WarmStartTrainer but drives an AxionGNN instead of a dense MLP.
    Call train_step() after the engine's load_data() has been called so that
    body_pose_prev, body_vel_prev, and the contact arrays are populated for
    the current timestep.

    Args:
        engine: the AxionEngine instance (used for compute_warm_start_forces
            when warm_start_constr_forces=True).
        warm_start_constr_forces: if True, compute physically consistent
            constraint forces for the predicted state before evaluating the
            residual. The forces are detached from the autograd graph (the
            linear solve is not differentiable), so gradients still only flow
            through body_vel. Gives a more informative residual signal at the
            cost of two extra linear solves per step.
    """

    def __init__(
        self,
        net: AxionGNN,
        axion_model: AxionModel,
        axion_contacts: AxionContacts,
        data: EngineData,
        config: AxionEngineConfig,
        dims: EngineDimensions,
        lr: float = 1e-4,
        engine=None,
        warm_start_constr_forces: bool = False,
    ):
        self.net = net
        self.axion_model = axion_model
        self.axion_contacts = axion_contacts
        self.data = data
        self.config = config
        self.dims = dims
        self.torch_device = torch.device(wp.device_to_torch(axion_model.device))
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

        if warm_start_constr_forces and engine is None:
            raise ValueError("engine must be provided when warm_start_constr_forces=True")
        self.engine = engine
        self.warm_start_constr_forces = warm_start_constr_forces

    def set_engine(self, engine) -> None:
        """Swap in a new engine (e.g. after rebuilding the scene for a new episode).

        The optimizer and GNN weights are unaffected, so Adam momentum and
        learning-rate schedule carry over across episodes.
        """
        self.axion_model = engine.axion_model
        self.axion_contacts = engine.axion_contacts
        self.data = engine.data
        self.config = engine.config
        self.dims = engine.dims
        self.engine = engine
        self.torch_device = torch.device(wp.device_to_torch(engine.axion_model.device))

    def build_graph(self):
        """Build a PyG HeteroData graph from the current engine state.

        Requires that engine.load_data() has already been called so that
        body_pose_prev, body_vel_prev, ext_force, and contact arrays are valid.
        """
        device = self.torch_device

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

        joint_type = wp.to_torch(self.axion_model.joint_type).to(torch.long)
        joint_parent = wp.to_torch(self.axion_model.joint_parent).to(torch.long)
        joint_child = wp.to_torch(self.axion_model.joint_child).to(torch.long)
        joint_X_p = wp.to_torch(self.axion_model.joint_X_p)
        joint_X_c = wp.to_torch(self.axion_model.joint_X_c)
        joint_axis = wp.to_torch(self.axion_model.joint_axis)
        joint_qd_start = wp.to_torch(self.axion_model.joint_qd_start).to(torch.long)
        joint_enabled = wp.to_torch(self.axion_model.joint_enabled)
        joint_compliance = wp.to_torch(self.axion_model.joint_compliance)

        return build_graph(
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
            joint_type,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_axis,
            joint_qd_start,
            joint_enabled,
            joint_compliance,
            self.dims.joint_count,
            self.dims.body_count,
            device,
            shape_body,
            contact_dist_threshold=0.5,
        )

    def _compute_constr_force_for(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute converged constraint forces by running the Newton solver from the
        previous state. The returned forces satisfy the nonlinear residual exactly at
        the Newton solution, so r(v_gnn, λ_newton) ≈ M*(v_gnn − v*)/dt — a clean,
        bounded training signal regardless of how far the GNN prediction is from v*.

        The warm-start approach (linearised solve at the GNN-predicted state) fails
        because the linearisation is inaccurate for states with large penetration
        (e.g. random-init GNN), leaving the nonlinear residual huge.

        Args:
            body_vel: unused — forces come from the Newton solve, not the GNN state.

        Returns:
            (W, num_constraints) converged constraint forces, detached.
        """
        # Reset to the previous state — identical to the reset in run_training_step
        wp.copy(dest=self.data.body_pose, src=self.data.body_pose_prev)
        wp.copy(dest=self.data.body_vel, src=self.data.body_vel_prev)
        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        # Full Newton-Raphson solve → converged (v*, λ*)
        self.engine._solve()

        return wp.to_torch(self.data._constr_force).clone().detach()

    def train_step(self) -> float:
        """One training step: GNN forward → residual loss → backprop.

        Must be called after engine.load_data() and before the physics solve
        (engine._solve()). The engine buffers modified during this step
        (body_vel, body_pose, _constr_force) are reset by the caller in
        run_training_step() before _solve(), so no cleanup is needed here.

        Returns:
            Scalar loss value.
        """
        self.optimizer.zero_grad()

        graph = self.build_graph()

        self.net.train()
        node_outputs, _ = self.net(
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict
        )

        W = self.dims.num_worlds
        B = self.dims.body_count

        # Predicted acceleration (W*B, 6) → full velocity (W, N_u)
        accel = node_outputs["object"].reshape(W, B * 6)
        body_vel_prev = wp.to_torch(self.data.body_vel_prev).reshape(W, B * 6).detach()
        body_vel = body_vel_prev + accel

        if self.warm_start_constr_forces:
            # Converged Newton forces from the previous state (detached).
            # Using reference forces λ* independent of the GNN prediction keeps
            # the residual bounded: r(v_gnn, λ*) ≈ M*(v_gnn − v*)/dt.
            constr_force = self._compute_constr_force_for(body_vel)
        else:
            constr_force = torch.zeros(
                W, self.dims.num_constraints,
                device=self.torch_device,
                dtype=torch.float32,
            )

        residual = AxionResidualAD.apply(
            self.axion_model,
            self.axion_contacts,
            self.data,
            self.config,
            self.dims,
            body_vel,
            constr_force,
        )

        loss = residual.pow(2).sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()
