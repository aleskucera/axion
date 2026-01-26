from typing import Callable
from typing import Optional

import warp as wp
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.constraints.control_constraint import compute_control_constraint_offsets_batched
from axion.constraints.control_constraint import fill_control_constraint_body_idx_kernel
from axion.constraints.utils import compute_joint_constraint_offsets_batched
from axion.core.contacts import AxionContacts
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.engine_logger import EngineEvents
from axion.core.engine_logger import EngineMode
from axion.core.engine_logger import HDF5Observer
from axion.core.linear_utils import compute_dbody_qd_from_dbody_lambda
from axion.core.linear_utils import compute_linear_system
from axion.core.linesearch_utils import perform_linesearch
from axion.core.linesearch_utils import update_body_q
from axion.core.model import AxionModel
from axion.core.history_utils import copy_state_to_history
from axion.optim import JacobiPreconditioner
from axion.optim import PCRSolver
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


@wp.kernel
def _check_newton_convergence(
    h_norm_sq: wp.array(dtype=float),
    atol_sq: float,
    iter_count: wp.array(dtype=int),
    max_iters: int,
    keep_running: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid == 0:
        current_iter = iter_count[0] + 1
        iter_count[0] = current_iter
        if current_iter >= max_iters:
            keep_running[0] = 0
            return

    current_iter = iter_count[0]
    if current_iter >= max_iters:
        return

    if tid < h_norm_sq.shape[0]:
        if h_norm_sq[tid] > atol_sq:
            keep_running[0] = 1


class AxionEngine(SolverBase):
    """
    The class implements a low-level physics solver.
    The engine implements a Non-Smooth Newton Method to solve
    the entire physics state—including dynamics, contacts,
    and joints—as a single, unified problem at each time step.
    """

    def __init__(
        self,
        model: Model,
        init_state_fn: Callable[[State, State, Contacts, float], None],
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
    ):
        super().__init__(model)
        self.init_state_fn = init_state_fn
        self.config = config

        # --- 1. Event System Setup ---
        self.events = EngineEvents()

        # Pre-allocate timing events if timing might be used
        if self.config.enable_timing:
            self.events.allocate_timing_events(self.config.max_newton_iters)

        # Attach Data Observer (Debug)
        # Note: We pass config here so it only activates if enable_hdf5_logging is True
        self.data_observer = HDF5Observer(self.events, self.config)

        # --- 2. Model & Data Setup ---
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
        linesearch_step_count = (
            self.config.linesearch_conservative_step_count
            + self.config.linesearch_optimistic_step_count
        )

        self.dims = EngineDimensions(
            num_worlds=self.axion_model.num_worlds,
            body_count=self.axion_model.body_count,
            contact_count=self.axion_contacts.max_contacts,
            joint_count=self.axion_model.joint_count,
            linesearch_step_count=linesearch_step_count,
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
            allocate_history=self.config.enable_hdf5_logging,
        )

        self.A_op = SystemOperator(
            data=SystemLinearData.from_engine(self),
            regularization=self.config.regularization,
            device=self.device,
        )

        self.preconditioner = JacobiPreconditioner(self, self.config.regularization)

        self.cr_solver = PCRSolver(
            max_iters=self.config.max_linear_iters,
            batch_dim=self.dims.num_worlds,
            vec_dim=self.dims.N_c,
            device=self.device,
        )

        self.data.set_g_accel(model)

        # Loop control
        self.keep_running = wp.zeros(shape=(1,), dtype=int, device=self.device)
        self.iter_count = wp.zeros(shape=(1,), dtype=int, device=self.device)
        self.h_norm_sq = wp.zeros(
            shape=(self.dims.num_worlds,), dtype=wp.float32, device=self.device
        )
        self.tiled_sq_norm = TiledSqNorm(
            shape=(self.dims.num_worlds, self.dims.N_u + self.dims.N_c),
            dtype=wp.float32,
            tile_size=256,
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

        if self.dims.N_n > 0:
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
                    self.data.contact_body_a,
                    self.data.contact_body_b,
                    self.data.contact_point_a,
                    self.data.contact_point_b,
                    self.data.contact_thickness_a,
                    self.data.contact_thickness_b,
                    self.data.contact_dist,
                    self.data.contact_friction_coeff,
                    self.data.contact_restitution_coeff,
                    self.data.contact_basis_n_a,
                    self.data.contact_basis_t1_a,
                    self.data.contact_basis_t2_a,
                    self.data.contact_basis_n_b,
                    self.data.contact_basis_t1_b,
                    self.data.contact_basis_t2_b,
                ],
                device=self.device,
            )

        if self.dims.N_j > 0:
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

        if self.dims.N_ctrl > 0:
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

        if self.dims.N_n > 0:
            wp.launch(
                kernel=fill_contact_constraint_body_idx_kernel,
                dim=(self.axion_model.num_worlds, self.dims.N_n),
                inputs=[
                    self.data.contact_body_a,
                    self.data.contact_body_b,
                ],
                outputs=[
                    self.data.constraint_body_idx.n,
                ],
                device=self.device,
            )

        if self.dims.N_f > 0:
            wp.launch(
                kernel=fill_friction_constraint_body_idx_kernel,
                dim=(self.axion_model.num_worlds, self.dims.N_f),
                inputs=[
                    self.data.contact_body_a,
                    self.data.contact_body_b,
                ],
                outputs=[
                    self.data.constraint_body_idx.f,
                ],
                device=self.device,
            )

    def _execute_newton_step_math(
        self, dt: float, iter_idx: int = 0, log_linear_solver: bool = False
    ):
        """
        The pure physics logic.
        Uses PolymorphicScope to handle mode-specific instrumentation.
        """
        # Maintain history for friction
        wp.copy(dest=self.data._body_lambda_prev, src=self.data._body_lambda)
        wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

        # Linearize
        with self.events.linearization.scope(iter_idx=iter_idx):
            compute_linear_system(self.axion_model, self.data, self.config, self.dims, dt)
            self.data.h.sync_to_float()
            self.preconditioner.update()

        # Solve
        solver_stats = None
        with self.events.linear_solve.scope(iter_idx=iter_idx):
            self.data._dbody_lambda.zero_()
            solver_stats = self.cr_solver.solve(
                A=self.A_op,
                b=self.data.b,
                x=self.data.dbody_lambda.full,
                preconditioner=self.preconditioner,
                iters=self.config.max_linear_iters,
                tol=self.config.linear_tol,
                atol=self.config.linear_atol,
                log=log_linear_solver,
            )
            compute_dbody_qd_from_dbody_lambda(self.axion_model, self.data, self.config, self.dims)

        # Linesearch
        with self.events.linesearch.scope(iter_idx=iter_idx):
            perform_linesearch(self.axion_model, self.data, self.config, self.dims)
            update_body_q(self.axion_model, self.data, self.config, self.dims)
            self._update_mass_matrix()

        return solver_stats

    def _check_convergence_kernel_launch(self):
        """Helper to launch the convergence check kernel."""
        self.tiled_sq_norm.compute(self.data.h.full, self.h_norm_sq)
        wp.launch(
            kernel=_check_newton_convergence,
            dim=(self.dims.num_worlds,),
            device=self.device,
            inputs=[
                self.h_norm_sq,
                self.config.newton_atol**2,
                self.iter_count,
                self.config.max_newton_iters,
                self.keep_running,
            ],
        )

    def _solve_production(self, dt: float):
        def loop_body():
            # The scope() calls inside here resolve to 'pass'
            self._execute_newton_step_math(dt, iter_idx=0)
            self._check_convergence_kernel_launch()

        # If we are already capturing (e.g. AbstractSimulator), we insert the while loop node.
        if self.device.is_capturing:
            wp.capture_while(self.keep_running, loop_body)
        else:
            # Fallback for eager execution (no graph)
            # This is slower but functional for debugging or legacy pipelines
            while True:
                loop_body()
                # Must sync to CPU to check condition
                if self.keep_running.numpy()[0] == 0:
                    break

    def _solve_timing(self, dt: float):
        # Unroll the loop to bake discrete events
        # Works in both graph (unrolled nodes) and eager (iterative execution) modes
        for i in range(self.config.max_newton_iters):
            self._execute_newton_step_math(dt, iter_idx=i)

    def _solve_debug(self, dt: float):
        # 0. Capture Initial State (Iter 0)
        copy_state_to_history(0, self.data, self.config, self.dims)

        for i in range(self.config.max_newton_iters):
            # 1. Run Math (Signals fire automatically for start/end)
            solver_stats = self._execute_newton_step_math(dt, iter_idx=i, log_linear_solver=True)

            # 2. Capture Updated State (Iter i+1)
            copy_state_to_history(i + 1, self.data, self.config, self.dims)

            # 3. Log Data Snapshot
            # (In debug mode, we assume HDF5 is enabled)
            if self.config.enable_hdf5_logging:
                snapshot = self.data.get_snapshot()
                if solver_stats:
                    snapshot["linear_solver_stats"] = solver_stats
                self.events.newton_iteration_end.emit(iter_idx=i, snapshot=snapshot)

            # 4. Check Convergence (CPU Sync required)
            self._check_convergence_kernel_launch()
            if self.keep_running.numpy()[0] == 0:
                break

    def _solve_nonlinear_system(self, dt: float):
        """Orchestrator."""
        self.keep_running.fill_(1)
        self.iter_count.zero_()

        # Decide Mode
        if self.config.enable_hdf5_logging:
            self.events.current_mode = EngineMode.DEBUG
            self._solve_debug(dt)
        elif self.config.enable_timing:
            self.events.current_mode = EngineMode.TIMING
            self._solve_timing(dt)
        else:
            self.events.current_mode = EngineMode.PRODUCTION
            self._solve_production(dt)

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
        with self.events.step.scope(iter_idx=self._timestep):

            self.data.set_dt(dt)

            with self.events.control.scope():
                self._load_control_inputs(state_in, control)

            with self.events.initial_guess.scope():
                self._initialize_variables(state_in, state_out, contacts)
                self._update_mass_matrix()
                self._initialize_constraints(contacts)

            self._solve_nonlinear_system(dt)
            self._finalize_step(state_out)

        # # After step, if timing, we might want to print
        # if self.events.current_mode == EngineMode.TIMING:
        #     # Note: This requires synchronizing/reading back events
        #     # Usually you'd do this once per second or at end of sim
        #     pass
