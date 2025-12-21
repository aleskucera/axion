from typing import Callable
from typing import Optional

import newton
import warp as wp
from axion.constraints import fill_contact_constraint_active_mask_kernel
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_active_mask_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_active_mask_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.optim import CRSolver
from axion.optim import JacobiPreconditioner
from axion.optim import SystemLinearData
from axion.optim import SystemOperator
from axion.types import compute_joint_constraint_offsets_batched
from axion.types import contact_interaction_kernel
from axion.types import world_spatial_inertia_kernel
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from .contacts import AxionContacts
from .control_utils import apply_control
from .engine_config import AxionEngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .engine_logger import EngineLogger
from .linear_utils import compute_dbody_qd_from_dbody_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch
from .linesearch_utils import update_body_q
from .model import AxionModel
from .pca_utils import copy_state_to_history


class AxionEngine(SolverBase):
    """
    The class implements a low-level physics solver.
    The engine implements a Non-Smooth Newton Method to solve
    the entire physics state—including dynamics, contacts,
    and joints—as a single, unified problem at each time step.
    This monolithic approach provides exceptional stability,
    especially for complex, highly-constrained systems like
    articulated robots.
    """

    def __init__(
        self,
        model: Model,
        init_state_fn: Callable[[State, State, Contacts, float], None],
        logger: EngineLogger,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
    ):
        """
        Initialize the physics engine for the given model and configuration.

        Args:
            model: The warp.sim.Model physics model containing bodies, joints, and other physics properties.
            config: Configuration parameters for the engine of type EngineConfig.
            logger: Optional HDF5Logger or NullLogger for recording simulation data.
        """
        super().__init__(model)

        self.init_state_fn = init_state_fn
        self.logger = logger
        self.config = config

        self.axion_model = AxionModel(model)
        self.axion_contacts = AxionContacts(
            model,
            max_contacts_per_world=self.config.max_contacts_per_world,
        )

        joint_constraint_offsets, num_constraints = compute_joint_constraint_offsets_batched(
            self.axion_model.joint_type,
        )

        self.dims = EngineDimensions(
            num_worlds=self.axion_model.num_worlds,
            body_count=self.axion_model.body_count,
            contact_count=self.axion_contacts.max_contacts,
            joint_count=self.axion_model.joint_count,
            linesearch_step_count=self.config.linesearch_step_count,
            joint_constraint_count=num_constraints,
        )

        self.data = EngineData.create(
            self.dims,
            self.config,
            joint_constraint_offsets,
            self.device,
            self.logger.uses_pca_arrays,
            self.logger.config.pca_grid_res,
        )

        self.A_op = SystemOperator(
            data=SystemLinearData.from_engine(self),
            device=self.device,
            regularization=self.config.regularization,
        )

        self.preconditioner = JacobiPreconditioner(self)

        self.cr_solver = CRSolver(
            num_worlds=self.dims.num_worlds,
            vec_dim=self.dims.N_c,
            dtype=wp.float32,
            device=self.device,
        )

        self.data.set_g_accel(model)

        self._timestep = 0

    def _apply_control(self, state: State, control: Control):
        newton.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        apply_control(self.model, state, self.data.dt, control)
        wp.copy(dest=self.data.body_f, src=state.body_f)

    def _initialize_variables(self, state_in: State, state_out: State, contacts: Contacts):
        self.init_state_fn(state_in, state_out, contacts, self.data.dt)

        wp.copy(dest=self.data.body_q, src=state_out.body_q)
        wp.copy(dest=self.data.body_u, src=state_out.body_qd)
        wp.copy(dest=self.data.body_q_prev, src=state_in.body_q)
        wp.copy(dest=self.data.body_u_prev, src=state_in.body_qd)

        self.data._body_lambda.zero_()
        self.data._body_lambda_prev.zero_()

    def _update_mass_matrix(self):
        wp.launch(
            kernel=world_spatial_inertia_kernel,
            dim=(self.axion_model.num_worlds, self.axion_model.body_count),
            inputs=[
                self.data.body_q,
                self.axion_model.body_mass,
                self.axion_model.body_inertia,
            ],
            outputs=[
                self.data.world_M,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=world_spatial_inertia_kernel,
            dim=(self.axion_model.num_worlds, self.axion_model.body_count),
            inputs=[
                self.data.body_q,
                self.axion_model.body_inv_mass,
                self.axion_model.body_inv_inertia,
            ],
            outputs=[
                self.data.world_M_inv,
            ],
            device=self.device,
        )

    def _initialize_constraints(self, contacts: Contacts):
        self.axion_contacts.load_contact_data(contacts)

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=(self.axion_model.num_worlds, self.axion_contacts.max_contacts),
            inputs=[
                self.data.body_q,
                self.axion_model.body_com,
                self.axion_model.shape_body,
                self.axion_model.shape_thickness,
                self.axion_model.shape_material_mu,
                self.axion_model.shape_material_restitution,
                self.axion_contacts.contact_count,
                self.axion_contacts.contact_point0,
                self.axion_contacts.contact_point1,
                self.axion_contacts.contact_normal,
                self.axion_contacts.contact_shape0,
                self.axion_contacts.contact_shape1,
                self.axion_contacts.contact_thickness0,
                self.axion_contacts.contact_thickness1,
            ],
            outputs=[
                self.data.contact_interaction,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_joint_constraint_body_idx_kernel,
            dim=(self.axion_model.num_worlds, self.axion_model.joint_count),
            inputs=[
                self.axion_model.joint_type,
                self.axion_model.joint_parent,
                self.axion_model.joint_child,
                self.data.joint_constraint_offsets,
            ],
            outputs=[
                self.data.constraint_body_idx.j,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_contact_constraint_body_idx_kernel,
            dim=(self.axion_model.num_worlds, self.dims.N_n),
            inputs=[
                self.data.contact_interaction,
            ],
            outputs=[
                self.data.constraint_body_idx.n,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_friction_constraint_body_idx_kernel,
            dim=(self.axion_model.num_worlds, self.dims.N_f),
            inputs=[
                self.data.contact_interaction,
            ],
            outputs=[
                self.data.constraint_body_idx.f,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_joint_constraint_active_mask_kernel,
            dim=(self.axion_model.num_worlds, self.axion_model.joint_count),
            inputs=[
                self.axion_model.joint_type,
                self.axion_model.joint_enabled,
                self.axion_model.joint_child,
                self.data.joint_constraint_offsets,
            ],
            outputs=[
                self.data.constraint_active_mask.j,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_contact_constraint_active_mask_kernel,
            dim=(self.axion_model.num_worlds, self.dims.N_n),
            inputs=[
                self.data.contact_interaction,
            ],
            outputs=[
                self.data.constraint_active_mask.n,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_friction_constraint_active_mask_kernel,
            dim=(self.axion_model.num_worlds, self.dims.N_f),
            inputs=[
                self.data.contact_interaction,
            ],
            outputs=[
                self.data.constraint_active_mask.f,
            ],
            device=self.device,
        )

    # def _update_constraint_positional_errors(self):
    #     wp.launch(
    #         kernel=update_penetration_depth_kernel,
    #         dim=(self.axion_model.num_worlds, self.axion_contacts.max_contacts),
    #         inputs=[
    #             self.data.body_q,
    #             self.data.contact_interaction,
    #         ],
    #         device=self.device,
    #     )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        step_events = self.logger.step_event_pairs[self.logger.current_step_in_segment]

        # if contacts.rigid_contact_count.numpy()[0] > 0:
        #     print(f"Contact count {contacts.rigid_contact_count} in timestep {self._timestep}.")

        self.data.set_dt(dt)

        # Control block
        with self.logger.timed_block(*step_events["control"]):
            self._apply_control(state_in, control)

        # Initial guess block
        with self.logger.timed_block(*step_events["initial_guess"]):
            self._initialize_variables(state_in, state_out, contacts)
            self._update_mass_matrix()
            self._initialize_constraints(contacts)

        for i in range(self.config.newton_iters):
            wp.copy(dest=self.data._body_lambda_prev, src=self.data._body_lambda)
            wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

            newton_iter_events = self.logger.engine_event_pairs[
                self.logger.current_step_in_segment
            ][i]

            # System linearization block
            with self.logger.timed_block(*newton_iter_events["system_linearization"]):
                compute_linear_system(self.axion_model, self.data, self.config, self.dims, dt)

            self.logger.log_newton_iteration_data(self, i)
            if self.logger.uses_pca_arrays:
                copy_state_to_history(i, self.data, self.config, self.dims)

            self.preconditioner.update()

            # Linear system solve block
            with self.logger.timed_block(*newton_iter_events["linear_system_solve"]):
                self.data._dbody_lambda.zero_()
                self.cr_solver.solve(
                    A=self.A_op,
                    b=self.data.b,
                    x=self.data.dbody_lambda.full,
                    iters=self.config.linear_iters,
                    M=self.preconditioner,
                )
                compute_dbody_qd_from_dbody_lambda(self.data, self.config, self.dims)
                # if np.max(self.data._dbody_lambda.numpy()) > 0:
                #     print("Here")

            # Linesearch block
            with self.logger.timed_block(*newton_iter_events["linesearch"]):
                perform_linesearch(self.axion_model, self.data, self.config, self.dims)
                update_body_q(self.axion_model, self.data, self.config, self.dims)
                self._update_mass_matrix()

        if self.logger.uses_pca_arrays:
            # Log the final state
            compute_linear_system(self.axion_model, self.data, self.config, self.dims, dt)
            copy_state_to_history(self.config.newton_iters, self.data, self.config, self.dims)

        self.logger.log_residual_norm_landscape(self)

        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)

        self._timestep += 1
