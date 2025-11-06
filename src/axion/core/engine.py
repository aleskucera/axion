from typing import Callable
from typing import Optional

import newton
import numpy as np
import scipy
import warp as wp
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from axion.types import compute_joint_constraint_offsets
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from .control_utils import apply_control
from .engine_config import AxionEngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .engine_logger import EngineLogger
from .linear_utils import compute_dbody_qd_from_dbody_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch
from .linesearch_utils import update_body_q


@wp.kernel
def print_linear_solve_result(
    final_iteration: wp.array(dtype=wp.int32),
    residual_norm_squared: wp.array(dtype=wp.float32),
    absolute_tolerance_squared: wp.array(dtype=wp.float32),
):

    if wp.tid() > 0:
        return

    # if final_iteration[0] > 4:
    #     wp.printf("Final iteration: %d\n", final_iteration[0])
    if residual_norm_squared[0] > 1e-4:
        wp.printf("Residual norm squared: %f\n", residual_norm_squared[0])


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

        joint_constraint_offsets, num_constraints = compute_joint_constraint_offsets(
            model.joint_type,
        )

        self.dims = EngineDimensions(
            body_count=self.model.body_count,
            contact_count=self.model.rigid_contact_max,
            joint_count=self.model.joint_count,
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

        if self.config.matrixfree_representation:
            self.A_op = MatrixFreeSystemOperator(
                engine=self,
                regularization=self.config.regularization,
            )
        else:
            self.A_op = MatrixSystemOperator(
                engine=self,
                regularization=self.config.regularization,
            )

        self.preconditioner = JacobiPreconditioner(self)

        self.data.set_body_M(model)
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
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        step_events = self.logger.step_event_pairs[self.logger.current_step_in_segment]

        # Control block
        with self.logger.timed_block(*step_events["control"]):
            newton.eval_ik(self.model, state_in, state_in.joint_q, state_in.joint_qd)
            apply_control(self.model, state_in, state_out, dt, control)

        # Initial guess block
        with self.logger.timed_block(*step_events["initial_guess"]):
            self.init_state_fn(state_in, state_out, contacts, dt)
            self.data.update_state_data(self.model, state_in, state_out, contacts, dt)
            self.data.body_lambda.zero_()

        for i in range(self.config.newton_iters):
            wp.copy(dest=self.data.body_lambda_prev, src=self.data.body_lambda)
            wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

            newton_iter_events = self.logger.engine_event_pairs[
                self.logger.current_step_in_segment
            ][i]

            # System linearization block
            with self.logger.timed_block(*newton_iter_events["system_linearization"]):
                compute_linear_system(self.model, self.data, self.config, self.dims, dt)

            if not self.config.matrixfree_representation:
                self.A_op.update()
            self.preconditioner.update()

            # Linear system solve block
            with self.logger.timed_block(*newton_iter_events["linear_system_solve"]):
                self.data.dbody_lambda.zero_()
                ret = wp.optim.linear.cr(
                    A=self.A_op,
                    b=self.data.b,
                    x=self.data.dbody_lambda,
                    atol=2e-5,
                    maxiter=20,
                    M=self.preconditioner,
                    check_every=0,
                    use_cuda_graph=True,
                )
                final_iteration = ret[0]
                residual_norm_squared = ret[1]
                absolute_tolerance_squared = ret[2]

                compute_dbody_qd_from_dbody_lambda(self.data, self.config, self.dims)

            # Linesearch block
            with self.logger.timed_block(*newton_iter_events["linesearch"]):
                perform_linesearch(self.model, self.data, self.config, self.dims)

            self.logger.log_newton_iteration_data(self, i)
            self._copy_computed_state_to_trajectory(i)

        self.logger.log_residual_norm_landscape(self)

        update_body_q(self.model, self.data, self.config, self.dims)
        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)

    def simulate_scipy(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
        method: str = "hybr",
        tolerance: float = 1e-10,
        max_iterations: int = 5000,
    ):
        apply_control(model, state_in, state_out, dt, control)
        self.init_state_fn(model, state_in, state_out, dt)
        self.data.update_state_data(model, state_in, state_out)

        def residual_function(x: np.ndarray) -> np.ndarray:
            wp.copy(dest=self.data.body_lambda_prev, src=self.data.body_lambda)

            # x contains both lambda and body_qd
            n_lambda = self.dims.N_c
            lambda_vals = x[:n_lambda]
            body_qd_vals = x[n_lambda:]

            # Store current state
            lambda_backup = wp.clone(self.data.body_lambda)
            body_qd_backup = wp.clone(self.data.body_u)

            try:
                # Set state from input vector
                self.data.body_lambda.assign(lambda_vals)
                self.data.body_u.assign(body_qd_vals)

                # Compute residuals (right hand side of the linear system)
                compute_linear_system(self.data, self.config, self.dims, dt)

                # Residual is concatenation of g and h vector
                return self.data.h_c.numpy()

            finally:
                # Restore original state
                wp.copy(dest=self.data.body_lambda, src=lambda_backup)
                wp.copy(dest=self.data.body_u, src=body_qd_backup)

        # Initial guess from current state
        x0 = np.concatenate([self.data.body_lambda.numpy(), self.data.body_u.numpy().flatten()])

        # Solve
        result = scipy.optimize.root(
            residual_function,
            x0,
            method=method,
            options={"xtol": tolerance, "maxfev": max_iterations},
        )

        n_lambda = self.dims.con_dim
        self.data.body_lambda.assign(wp.from_numpy(result.x[:n_lambda].astype(np.float32)))
        body_qd_solution = result.x[n_lambda:][np.newaxis, :]
        self.data.body_u.assign(wp.from_numpy(body_qd_solution.astype(np.float32)))

        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)
