from typing import Callable
from typing import Optional

import newton
import warp as wp
from axion.optim import CRSolver
from axion.optim import JacobiPreconditioner
from axion.optim import SystemOperator
from axion.types import compute_joint_constraint_offsets_batched
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from .batched_model import BatchedContacts
from .batched_model import BatchedModel
from .control_utils import apply_control
from .engine_config import AxionEngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .engine_logger import EngineLogger
from .linear_utils import compute_dbody_qd_from_dbody_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch
from .linesearch_utils import update_body_q


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

        self.batched_model = BatchedModel(model)
        self.batched_contacts = BatchedContacts(
            model,
            max_contacts_per_world=self.config.max_contacts_per_world,
        )

        joint_constraint_offsets, num_constraints = compute_joint_constraint_offsets_batched(
            self.batched_model.joint_type,
        )

        self.dims = EngineDimensions(
            num_worlds=self.batched_model.num_worlds,
            body_count=self.batched_model.body_count,
            contact_count=self.batched_contacts.max_contacts,
            joint_count=self.batched_model.joint_count,
            linesearch_step_count=self.config.linesearch_step_count,
            joint_constraint_count=num_constraints,
        )

        self.data = create_engine_arrays(
            self.dims,
            self.config,
            joint_constraint_offsets,
            self.device,
            self.logger.uses_dense_matrices,
            self.logger.uses_pca_arrays,
            self.logger.config.pca_grid_res,
        )

        self.A_op = SystemOperator(
            engine=self,
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

    def _copy_computed_state_to_trajectory(self, iter: int):
        if self.logger.uses_pca_arrays:
            wp.copy(
                dest=self.data.optim_trajectory,
                src=self.data.body_u,
                dest_offset=(iter * (self.dims.N_u + self.dims.N_c)),
            )
            wp.copy(
                dest=self.data.optim_trajectory,
                src=self.data.body_lambda,
                dest_offset=(iter * (self.dims.N_u + self.dims.N_c) + self.dims.N_u),
            )
            wp.copy(
                dest=self.data.optim_h,
                src=self.data.h,
                dest_offset=(iter * (self.dims.N_u + self.dims.N_c)),
            )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        step_events = self.logger.step_event_pairs[self.logger.current_step_in_segment]

        self.batched_contacts.load_contact_data(contacts)

        # Control block
        with self.logger.timed_block(*step_events["control"]):
            newton.eval_ik(self.model, state_in, state_in.joint_q, state_in.joint_qd)
            apply_control(self.model, state_in, dt, control)

        # Initial guess block
        with self.logger.timed_block(*step_events["initial_guess"]):
            self.init_state_fn(state_in, state_out, contacts, dt)
            self.data.update_state_data(
                self.model,
                self.batched_model,
                state_in,
                state_out,
                contacts,
                self.batched_contacts,
                dt,
            )
            self.data._body_lambda.zero_()

        for i in range(self.config.newton_iters):
            wp.copy(dest=self.data._body_lambda_prev, src=self.data._body_lambda)
            wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

            newton_iter_events = self.logger.engine_event_pairs[
                self.logger.current_step_in_segment
            ][i]

            # System linearization block
            with self.logger.timed_block(*newton_iter_events["system_linearization"]):
                compute_linear_system(self.data, self.config, self.dims, dt)

            self.preconditioner.update()

            # Linear system solve block
            with self.logger.timed_block(*newton_iter_events["linear_system_solve"]):
                self.data.dbody_lambda.zero_()
                self.cr_solver.solve(
                    A=self.A_op,
                    b=self.data.b,
                    x=self.data.dbody_lambda.full,
                    iters=self.config.linear_iters,
                    M=self.preconditioner,
                )
                compute_dbody_qd_from_dbody_lambda(self.data, self.config, self.dims)

            # Linesearch block
            with self.logger.timed_block(*newton_iter_events["linesearch"]):
                perform_linesearch(self.data, self.config, self.dims)

            self.logger.log_newton_iteration_data(self, i)
            # self._copy_computed_state_to_trajectory(i)

        # self.logger.log_residual_norm_landscape(self)

        update_body_q(self.batched_model, self.data, self.config, self.dims)
        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)
