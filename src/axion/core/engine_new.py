from typing import Callable
from typing import Optional

import warp as wp
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.constraints.control_constraint import compute_control_constraint_offsets_batched
from axion.constraints.control_constraint import fill_control_constraint_body_idx_kernel
from axion.constraints.utils import compute_joint_constraint_offsets_batched
from axion.optim import CRSolver
from axion.optim import JacobiPreconditioner
from axion.optim import SystemLinearData
from axion.optim import SystemOperator
from axion.tiled.tiled_utils import TiledSqNorm
from axion.types import contact_interaction_kernel
from axion.types import world_spatial_inertia_kernel
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from .contacts import AxionContacts
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


@wp.kernel
def _check_newton_convergence(
    h_norm_sq: wp.array(dtype=float),
    atol_sq: float,
    iter_count: wp.array(dtype=int),
    max_iters: int,
    keep_running: wp.array(dtype=int),
):
    tid = wp.tid()

    # 1. Update Iteration Count (thread 0)
    if tid == 0:
        current_iter = iter_count[0] + 1
        iter_count[0] = current_iter
        if current_iter >= max_iters:
            keep_running[0] = 0
            return

    # 2. Check Convergence (All threads)
    current_iter = iter_count[0]
    if current_iter >= max_iters:
        return

    # Check bounds
    if tid >= h_norm_sq.shape[0]:
        return

    if h_norm_sq[tid] > atol_sq:
        keep_running[0] = 1


@wp.kernel
def _print_newton_stats(iter_count: wp.array(dtype=int)):
    wp.printf("Newton iterations: %d\n", iter_count[0])


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

        control_constraint_offsets, num_control_constraints = (
            compute_control_constraint_offsets_batched(
                self.axion_model.joint_type,
                self.axion_model.joint_dof_mode,
                self.axion_model.joint_qd_start,
            )
        )

        dof_count = self.axion_model.joint_dof_count

        self.dims = EngineDimensions(
            num_worlds=self.axion_model.num_worlds,
            body_count=self.axion_model.body_count,
            contact_count=self.axion_contacts.max_contacts,
            joint_count=self.axion_model.joint_count,
            linesearch_step_count=self.config.linesearch_step_count,
            joint_constraint_count=num_constraints,
            control_constraint_count=num_control_constraints,
        )

        self.data = EngineData.create(
            self.dims,
            self.config,
            joint_constraint_offsets,
            control_constraint_offsets,
            dof_count,
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

        # Loop control buffers
        self.keep_running = wp.zeros(shape=(1,), dtype=int, device=self.device)
        self.iter_count = wp.zeros(shape=(1,), dtype=int, device=self.device)
        self.h_norm_sq = wp.zeros(
            shape=(self.dims.num_worlds,), dtype=wp.float32, device=self.device
        )
        self.tiled_sq_norm = TiledSqNorm(
            shape=(self.dims.num_worlds, self.dims.N_u + self.dims.N_c),
            dtype=wp.float32,
            device=self.device,
        )

        self._timestep = 0

    def _load_control_inputs(self, state: State, control: Control):
        wp.copy(dest=self.data.body_f, src=state.body_f)
        wp.copy(dest=self.data.joint_target, src=control.joint_target)

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
            kernel=fill_control_constraint_body_idx_kernel,
            dim=(self.axion_model.num_worlds, self.axion_model.joint_count),
            inputs=[
                self.axion_model.joint_parent,
                self.axion_model.joint_child,
                self.axion_model.joint_type,
                self.axion_model.joint_dof_mode,
                self.axion_model.joint_qd_start,
                self.data.control_constraint_offsets,
            ],
            outputs=[
                self.data.constraint_body_idx.ctrl,
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

    def _step_linearize(self, dt: float):
        compute_linear_system(self.axion_model, self.data, self.config, self.dims, dt)
        self.data.h.sync_to_float()
        self.preconditioner.update()

    def _step_solve(self, linear_tol: float, linear_atol: float):
        self.data._dbody_lambda.zero_()
        self.cr_solver.solve(
            A=self.A_op,
            b=self.data.b,
            x=self.data.dbody_lambda.full,
            iters=self.config.max_linear_iters,
            tol=linear_tol,
            atol=linear_atol,
            M=self.preconditioner,
        )
        compute_dbody_qd_from_dbody_lambda(self.data, self.config, self.dims)

    def _step_linesearch(self):
        perform_linesearch(self.axion_model, self.data, self.config, self.dims)
        update_body_q(self.axion_model, self.data, self.config, self.dims)
        self._update_mass_matrix()

    def _solve_nonlinear_system(self, dt: float):
        """
        Solves the nonlinear system using Newton's method.
        Uses CUDA graph capture for performance if timing is disabled.
        Uses explicit Python loop with timing blocks if timing is enabled.
        """
        self.keep_running.fill_(1)
        self.iter_count.zero_()

        # Check if 0 iters requested (edge case)
        if self.config.max_newton_iters <= 0:
            self.keep_running.zero_()
            return

        # Newton convergence parameters
        newton_atol_sq = float(self.config.newton_atol**2)
        if self.config.newton_mode == "fixed":
            newton_atol_sq = -1.0

        # Linear solver parameters
        linear_tol = self.config.linear_tol
        linear_atol = self.config.linear_atol
        if self.config.linear_mode == "fixed":
            linear_tol = -1.0
            linear_atol = -1.0

        if not self.logger.config.enable_timing:
            # --- FAST PATH: CUDA Graph Capture ---
            def newton_step_graph():
                self.keep_running.zero_()

                wp.copy(dest=self.data._body_lambda_prev, src=self.data._body_lambda)
                wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

                self._step_linearize(dt)
                self._step_solve(linear_tol, linear_atol)
                self._step_linesearch()

                # Increment and Check
                self.tiled_sq_norm.compute(self.data.h.full, self.h_norm_sq)
                wp.launch(
                    kernel=_check_newton_convergence,
                    dim=(self.dims.num_worlds,),
                    device=self.device,
                    inputs=[
                        self.h_norm_sq,
                        newton_atol_sq,
                        self.iter_count,
                        self.config.max_newton_iters,
                        self.keep_running,
                    ],
                )

            wp.capture_while(self.keep_running, newton_step_graph)

            wp.launch(
                kernel=_print_newton_stats,
                dim=(1,),
                device=self.device,
                inputs=[self.iter_count],
            )

        else:
            # --- PROFILING PATH: Python Loop with Timing ---
            step_idx = self.logger.current_step_in_segment

            for current_iter in range(self.config.max_newton_iters):
                events = self.logger.engine_event_pairs[step_idx][current_iter]

                wp.copy(dest=self.data._body_lambda_prev, src=self.data._body_lambda)
                wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

                with self.logger.timed_block(*events["system_linearization"]):
                    self._step_linearize(dt)

                self.logger.log_newton_iteration_data(self, current_iter)
                if self.logger.uses_pca_arrays:
                    copy_state_to_history(current_iter, self.data, self.config, self.dims)

                with self.logger.timed_block(*events["linear_system_solve"]):
                    self._step_solve(linear_tol, linear_atol)

                with self.logger.timed_block(*events["linesearch"]):
                    self._step_linesearch()

                # Check convergence (Optional: update GPU count for consistency)
                self.tiled_sq_norm.compute(self.data.h.full, self.h_norm_sq)
                self.keep_running.zero_()
                wp.launch(
                    kernel=_check_newton_convergence,
                    dim=(self.dims.num_worlds,),
                    device=self.device,
                    inputs=[
                        self.h_norm_sq,
                        newton_atol_sq,
                        self.iter_count,
                        self.config.max_newton_iters,
                        self.keep_running,
                    ],
                )

        if self.logger.uses_pca_arrays:
            # Log the final state
            compute_linear_system(self.axion_model, self.data, self.config, self.dims, dt)
            copy_state_to_history(self.config.max_newton_iters, self.data, self.config, self.dims)

        self.logger.log_residual_norm_landscape(self)

    def _finalize_step(self, state_out: State):
        """
        Finalizes the step by copying the engine's internal state to the output State object.
        """
        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)

        self._timestep += 1

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        step_events = self.logger.step_event_pairs[self.logger.current_step_in_segment]

        self.data.set_dt(dt)

        # 1. Load Inputs
        with self.logger.timed_block(*step_events["control"]):
            self._load_control_inputs(state_in, control)

        # 2. Initialize Guess
        with self.logger.timed_block(*step_events["initial_guess"]):
            self._initialize_variables(state_in, state_out, contacts)
            self._update_mass_matrix()
            self._initialize_constraints(contacts)

        # 3. Solve
        self._solve_nonlinear_system(dt)

        # 4. Finalize
        self._finalize_step(state_out)
