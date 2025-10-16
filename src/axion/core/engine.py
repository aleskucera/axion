from typing import Optional

import numpy as np
import scipy
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import NullLogger
from axion.optim import cr_solver
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

from .control_utils import apply_control
from .dense_utils import get_system_matrix_numpy
from .dense_utils import update_dense_matrices
from .engine_config import EngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .general_utils import update_body_q
from .general_utils import update_variables
from .linear_utils import compute_delta_body_qd_from_delta_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch


class AxionEngine(Integrator):
    def __init__(
        self,
        model: Model,
        config: Optional[EngineConfig],
        logger: Optional[HDF5Logger | NullLogger],
    ):
        super().__init__()
        self.device = model.device

        self.model = model
        self.logger = logger
        self.config = config

        self.dims = EngineDimensions(
            N_b=self.model.body_count,
            N_c=self.model.rigid_contact_max,
            N_j=self.model.joint_count,
            N_alpha=self.config.linesearch_steps,
        )

        allocate_dense_matrices = isinstance(self.logger, HDF5Logger)
        self.data = create_engine_arrays(self.dims, self.device, allocate_dense_matrices)

        if self.config.matrixfree_representation:
            self.A_op = MatrixFreeSystemOperator(self)
        else:
            self.A_op = MatrixSystemOperator(self)

        self.preconditioner = JacobiPreconditioner(self)

        self.data.set_generalized_mass(model)
        self.data.set_gravitational_acceleration(model)

        self.events = [
            {
                "iter_start": wp.Event(enable_timing=True),
                "linearize": wp.Event(enable_timing=True),
                "lin_solve": wp.Event(enable_timing=True),
                "linesearch": wp.Event(enable_timing=True),
            }
            for _ in range(self.config.newton_iters)
        ]

    def _log_newton_iteration_data(self):
        if isinstance(self.logger, NullLogger):
            return

        self.logger.log_wp_dataset("res", self.data.res)
        self.logger.log_wp_dataset("J_values", self.data.J_values)
        self.logger.log_wp_dataset("C_values", self.data.C_values)
        self.logger.log_wp_dataset("constraint_body_idx", self.data.constraint_body_idx)

        self.logger.log_wp_dataset("body_f", self.data.body_f)
        self.logger.log_wp_dataset("body_q", self.data.body_q)
        self.logger.log_wp_dataset("body_qd", self.data.body_qd)
        self.logger.log_wp_dataset("body_qd_prev", self.data.body_qd_prev)

        self.logger.log_wp_dataset("lambda", self.data._lambda)
        self.logger.log_wp_dataset("lambda_prev", self.data.lambda_prev)

        self.logger.log_wp_dataset("delta_body_qd", self.data.delta_body_qd)
        self.logger.log_wp_dataset("delta_lambda", self.data.delta_lambda)

        self.logger.log_wp_dataset("b", self.data.b)

        self.logger.log_struct_array("gen_mass", self.data.gen_mass)
        self.logger.log_struct_array("gen_inv_mass", self.data.gen_inv_mass)

        self.logger.log_struct_array("joint_interaction", self.data.joint_interaction)
        self.logger.log_struct_array("contact_interaction", self.data.contact_interaction)

        update_dense_matrices(self.data, self.config, self.dims)

        self.logger.log_wp_dataset("Minv_dense", self.data.Minv_dense)
        self.logger.log_wp_dataset("J_dense", self.data.J_dense)
        self.logger.log_wp_dataset("C_dense", self.data.C_dense)

        if not self.config.matrixfree_representation:
            self.logger.log_wp_dataset("A_dense", self.A_op._A)
        else:
            A_np = get_system_matrix_numpy(self.data, self.config, self.dims)
            cond_number = np.linalg.cond(A_np)
            self.logger.log_np_dataset("A_np", A_np)
            self.logger.log_scalar("cond_number", cond_number)

    def _log_static_data(self):
        if isinstance(self.logger, NullLogger):
            return

        self.logger.log_wp_dataset("gen_mass", self.data.gen_mass)
        self.logger.log_wp_dataset("gen_inv_mass", self.data.gen_inv_mass)

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
    ):
        apply_control(model, state_in, state_out, dt, control)
        self.integrate_bodies(model, state_in, state_out, dt)
        self.data.update_state_data(model, state_in, state_out)

        # TODO: Check the warm startup
        # self._lambda.zero_()

        for i in range(self.config.newton_iters):
            wp.record_event(self.events[i]["iter_start"])

            with self.logger.scope(f"newton_iteration_{i:02d}"):
                wp.copy(dest=self.data.lambda_prev, src=self.data._lambda)
                wp.copy(dest=self.data.lambda_n_scale_prev, src=self.data.lambda_n_scale)

                # --- Linearize the system of equations ---
                compute_linear_system(self.model, self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linearize"])

                if not self.config.matrixfree_representation:
                    self.A_op.update()
                self.preconditioner.update()

                # --- Solve linear system of equations ---
                cr_solver(
                    A=self.A_op,
                    b=self.data.b,
                    x=self.data.delta_lambda,
                    iters=self.config.linear_iters,
                    preconditioner=self.preconditioner,
                    logger=self.logger,
                )

                compute_delta_body_qd_from_delta_lambda(self.data, self.config, self.dims)
                wp.record_event(self.events[i]["lin_solve"])

                if self.config.linesearch_steps > 0:
                    perform_linesearch(self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linesearch"])

                update_variables(self.model, self.data, self.config, self.dims, dt)

                self._log_newton_iteration_data()

        update_body_q(self.model, self.data, self.config, self.dims, dt)
        wp.copy(dest=state_out.body_qd, src=self.data.body_qd)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)

        return self.events

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
        self.integrate_bodies(model, state_in, state_out, dt)
        self.data.update_state_data(model, state_in, state_out)

        def residual_function(x: np.ndarray) -> np.ndarray:
            wp.copy(dest=self.data.lambda_prev, src=self.data._lambda)

            # x contains both lambda and body_qd
            n_lambda = self.dims.con_dim
            lambda_vals = x[:n_lambda]
            body_qd_vals = x[n_lambda:]

            # Store current state
            lambda_backup = wp.clone(self.data._lambda)
            body_qd_backup = wp.clone(self.data.body_qd)

            try:
                # Set state from input vector
                self.data._lambda.assign(lambda_vals)
                self.data.body_qd.assign(body_qd_vals)

                # Compute residuals (right hand side of the linear system)
                compute_linear_system(self.data, self.config, self.dims, dt)

                # Residual is concatenation of g and h vector
                return self.data.res.numpy()

            finally:
                # Restore original state
                wp.copy(dest=self.data._lambda, src=lambda_backup)
                wp.copy(dest=self.data.body_qd, src=body_qd_backup)

        # Initial guess from current state
        x0 = np.concatenate([self.data._lambda.numpy(), self.data.body_qd.numpy().flatten()])

        # Solve
        result = scipy.optimize.root(
            residual_function,
            x0,
            method=method,
            options={"xtol": tolerance, "maxfev": max_iterations},
        )

        n_lambda = self.dims.con_dim
        self.data._lambda.assign(wp.from_numpy(result.x[:n_lambda].astype(np.float32)))
        body_qd_solution = result.x[n_lambda:][np.newaxis, :]
        self.data.body_qd.assign(wp.from_numpy(body_qd_solution.astype(np.float32)))

        wp.copy(dest=state_out.body_qd, src=self.data.body_qd)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)
