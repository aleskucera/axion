from typing import Any
from typing import Callable
from typing import Optional

import warp as wp
from axion.core.contacts import AxionContacts
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.linear_utils import compute_dbody_qd_from_dbody_lambda
from axion.core.linear_utils import compute_linear_system
from axion.core.linesearch_utils import perform_linesearch
from axion.core.logging_config import LoggingConfig
from axion.core.model import AxionModel
from axion.optim import JacobiPreconditioner
from axion.optim import PCRSolver
from axion.optim import SystemLinearData
from axion.optim import SystemOperator
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from .adjoint_utils import compute_adjoint_rhs_kernel
from .adjoint_utils import compute_body_adjoint_init_kernel
from .adjoint_utils import subtract_constraint_feedback_kernel
from .residual_utils import compute_residual
from .sim_logger import SimulationHDF5Logger


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


@wp.kernel
def increment_timestep_kernel(step_count: wp.array(dtype=int)):
    step_count[0] = step_count[0] + 1


@wp.kernel
def copy_best_sample_kernel(
    x_buffer: wp.array(dtype=Any, ndim=3),
    best_idx: wp.array(dtype=wp.int32),
    # Outputs
    x: wp.array(dtype=Any, ndim=2),
):
    world_idx, x_idx = wp.tid()
    x[world_idx, x_idx] = x_buffer[best_idx[world_idx], world_idx, x_idx]


@wp.kernel
def copy_best_sample_kernel_1d(
    x_buffer: wp.array(dtype=Any, ndim=2),
    best_idx: wp.array(dtype=wp.int32),
    # Outputs
    x: wp.array(dtype=Any, ndim=1),
):
    world_idx = wp.tid()
    x[world_idx] = x_buffer[best_idx[world_idx], world_idx]


@wp.kernel
def find_minimal_residual_index_kernel(
    batch_h_norm_sq: wp.array(dtype=wp.float32, ndim=2),
    iter_count: wp.array(dtype=wp.int32),
    start_idx: wp.int32,
    # Outputs
    minimal_index: wp.array(dtype=wp.int32),
):
    world_idx = wp.tid()
    count = iter_count[0]

    if count <= 0:
        minimal_index[world_idx] = wp.int32(0)
        return

    if count <= start_idx:
        # This means we exited early, so we take the value from the last iteration
        minimal_index[world_idx] = count - wp.int32(1)
        return

    min_idx = wp.int32(start_idx)
    min_value = batch_h_norm_sq[min_idx, world_idx]

    for i in range(start_idx + 1, count):
        value = batch_h_norm_sq[i, world_idx]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value

    minimal_index[world_idx] = min_idx


class AxionEngine(SolverBase):
    def __init__(
        self,
        model: Model,
        init_state_fn: Callable[[State, State, Contacts, float], None],
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
    ):
        super().__init__(model)
        self.init_state_fn = init_state_fn
        self.config = config
        self.logging_config = logging_config

        # --- 2. Model & Data Setup ---
        self.axion_model = AxionModel(model)
        self.axion_contacts = AxionContacts(model, self.config.max_contacts_per_world)

        self.dims = EngineDimensions(
            num_worlds=self.axion_model.num_worlds,
            body_count=self.axion_model.body_count,
            contact_count=self.axion_contacts.max_contacts,
            joint_count=self.axion_model.joint_count,
            joint_dof_count=self.axion_model.joint_dof_count,
            linesearch_step_count=self.config.num_linesearch_steps,
            joint_constraint_count=self.axion_model.num_joint_constraints,
            control_constraint_count=self.axion_model.num_control_constraints,
        )

        self.data = EngineData(
            model=self.axion_model,
            dims=self.dims,
            config=self.config,
            device=self.device,
            alloc_history_arrays=self.logging_config.enable_hdf5_logging,
            alloc_grad_arrays=self.config.differentiable_simulation,
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

        self.logger = None
        if self.logging_config.enable_hdf5_logging:
            self.logger = SimulationHDF5Logger(
                num_steps=self.logging_config.max_simulation_steps,
                data=self.data,
                config=self.config,
                dims=self.dims,
                device=self.device,
            )

        self.timestep = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _save_iter_to_history(self):
        if not self.logging_config.enable_hdf5_logging:
            return

        wp.copy(dest=self.data.pcr_iter_count, src=self.cr_solver.iter_count)
        wp.copy(dest=self.data.pcr_final_res_norm_sq, src=self.cr_solver.r_sq)
        wp.copy(dest=self.data.pcr_res_norm_sq_history, src=self.cr_solver.history_r_sq)

        self.data.save_iter_to_history()

    def _check_convergence(self):
        wp.launch(
            kernel=_check_newton_convergence,
            dim=(self.dims.num_worlds,),
            inputs=[
                self.data.res_norm_sq,
                self.config.newton_atol**2,
                self.data.iter_count,
                self.config.max_newton_iters,
            ],
            outputs=[self.data.keep_running],
            device=self.device,
        )

    def _restore_best_newton_candidate(self):
        wp.launch(
            kernel=find_minimal_residual_index_kernel,
            dim=(self.dims.num_worlds),
            inputs=[
                self.data.candidates_res_norm_sq,
                self.data.iter_count,
                self.config.backtrack_min_iter,
            ],
            outputs=[
                self.data.candidates_best_idx,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=copy_best_sample_kernel,
            dim=(self.dims.num_worlds, self.dims.body_count),
            inputs=[
                self.data.candidates_body_pose,
                self.data.candidates_best_idx,
            ],
            outputs=[
                self.data.body_pose,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=copy_best_sample_kernel,
            dim=(self.dims.num_worlds, self.dims.body_count),
            inputs=[
                self.data.candidates_body_vel,
                self.data.candidates_best_idx,
            ],
            outputs=[
                self.data.body_vel,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=copy_best_sample_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data._candidates_constr_force,
                self.data.candidates_best_idx,
            ],
            outputs=[
                self.data._constr_force,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=copy_best_sample_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data._candidates_res,
                self.data.candidates_best_idx,
            ],
            outputs=[
                self.data._res,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=copy_best_sample_kernel_1d,
            dim=(self.dims.num_worlds),
            inputs=[
                self.data.candidates_res_norm_sq,
                self.data.candidates_best_idx,
            ],
            outputs=[
                self.data.res_norm_sq,
            ],
            device=self.device,
        )
        # compute_residual(self.axion_model, self.data, self.config, self.dims)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        self.data.dt = dt

        # =========================================================================
        # Load the data from the arguments
        # =========================================================================

        # Load the actuation data
        wp.copy(dest=self.data.ext_force, src=state_in.body_f)
        wp.copy(dest=self.data.joint_target_pos, src=control.joint_target_pos)
        wp.copy(dest=self.data.joint_target_vel, src=control.joint_target_vel)

        # Initialize the optimization with init_state_fn heuristic
        self.init_state_fn(state_in, state_out, contacts, self.data.dt)

        wp.copy(dest=self.data.body_pose, src=state_out.body_q)
        wp.copy(dest=self.data.body_vel, src=state_out.body_qd)
        wp.copy(dest=self.data.body_pose_prev, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel_prev, src=state_in.body_qd)

        # Constraint impulses are currently initialized as zeros
        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        self.axion_contacts.load_contact_data(
            contacts,
            self.axion_model,
            self.data,
            self.dims,
        )

        # =========================================================================
        # Solve non-linear system with Newton-Raphson (NR) method
        # =========================================================================
        def nr_loop():
            # Linearize
            compute_linear_system(
                self.axion_model, self.axion_contacts, self.data, self.config, self.dims
            )
            self.preconditioner.update()

            # Linear Solve
            self.data._dconstr_force.zero_()
            self.cr_solver.solve(
                A=self.A_op,
                b=self.data.rhs,
                x=self.data.dconstr_force.full,
                preconditioner=self.preconditioner,
                iters=self.config.max_linear_iters,
                tol=self.config.linear_tol,
                atol=self.config.linear_atol,
                log=self.logging_config.enable_hdf5_logging,
            )
            compute_dbody_qd_from_dbody_lambda(self.axion_model, self.data, self.config, self.dims)

            # Linesearch
            wp.copy(dest=self.data._constr_force_prev_iter, src=self.data._constr_force)
            perform_linesearch(
                self.axion_model, self.axion_contacts, self.data, self.config, self.dims
            )

            self.data.save_state_to_candidates()
            self._save_iter_to_history()
            self._check_convergence()

        # Run the NR loop
        self.data.keep_running.fill_(1)
        self.data.iter_count.zero_()
        if self.device.is_capturing:
            wp.capture_while(self.data.keep_running, nr_loop)
        else:
            # Fallback for eager execution (no graph)
            while True:
                nr_loop()
                if self.data.keep_running.numpy()[0] == 0:
                    break

        self._restore_best_newton_candidate()
        if self.logger:
            self.logger.capture_step(self.timestep, self.data)

        # =========================================================================
        # Copy the computed state into the output state
        # =========================================================================
        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)

        wp.launch(
            kernel=increment_timestep_kernel, dim=(1,), inputs=[self.timestep], device=self.device
        )

    def step_backward(self):
        compute_linear_system(
            self.axion_model, self.axion_contacts, self.data, self.config, self.dims
        )

        wp.launch(
            kernel=compute_body_adjoint_init_kernel,
            dim=(self.dims.num_worlds, self.dims.body_count),
            inputs=[
                self.data.body_pose_grad,
                self.data.body_vel_grad,
                self.data.body_pose,
                self.axion_model.body_inv_mass,
                self.axion_model.body_inv_inertia,
                self.data.dt,
            ],
            outputs=[
                self.data.w.d_spatial,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=compute_adjoint_rhs_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data.J_values.full,
                self.data.constr_body_idx.full,
                self.data.constr_active_mask.full,
                self.data.w.d_spatial,
            ],
            outputs=[
                self.data.adjoint_rhs,
            ],
            device=self.device,
        )
        self.preconditioner.update()

        self._update_mass_matrix()
        self.data.w.c.full.zero_()
        _ = self.cr_solver.solve(
            A=self.A_op,
            b=self.data.adjoint_rhs,
            x=self.data.w.c.full,
            preconditioner=self.preconditioner,
            iters=self.config.max_linear_iters,
            tol=self.config.linear_tol,
            atol=self.config.linear_atol,
            log=False,
        )

        wp.launch(
            kernel=subtract_constraint_feedback_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data.w.c.full,
                self.data.J_values.full,
                self.data.constr_body_idx.full,
                self.data.constr_active_mask.full,
                self.data.body_pose,
                self.axion_model.body_inv_mass,
                self.axion_model.body_inv_inertia,
            ],
            outputs=[
                self.data.w.d_spatial,
            ],
            device=self.device,
        )

        self.data.w.sync_to_float()

        self.data.zero_gradients()

        # Initialize with explicit part BEFORE backward
        # This ensures tape.backward accumulates (adds) the implicit part to the explicit part
        # wp.copy(dest=self.data.body_pose_prev.grad, src=self.data.body_pose_grad)
        # wp.copy(dest=self.data.body_vel_prev.grad, src=self.data.body_vel_grad)

        tape = wp.Tape()
        with tape:
            compute_residual(self.axion_model, self.data, self.config, self.dims)

        # This adds the implicit gradient (-w^T * dh/d_theta) to the arrays
        tape.backward(grads={self.data._res: self.data._w})

    def save_logs(self):
        if self.logger:
            self.logger.save_to_hdf5(self.logging_config.hdf5_log_file)
