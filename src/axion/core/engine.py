from typing import Optional

import numpy as np
import scipy
import warp as wp
from axion.optim import cr_solver
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

from .control_utils import apply_control
from .engine_config import EngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .general_utils import update_variables
from .linear_utils import compute_delta_body_qd_from_delta_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch
from .logging_utils import HDF5Logger


class AxionEngine(Integrator):
    def __init__(
        self,
        model: Model,
        config: Optional[EngineConfig] = None,
        logger: Optional[HDF5Logger] = None,
    ):
        super().__init__()
        self.device = model.device

        self.model = model
        self.logger = logger
        self.config = config if config is not None else EngineConfig()

        self.dims = EngineDimensions(
            N_b=self.model.body_count,
            N_c=self.model.rigid_contact_max,
            N_j=self.model.joint_count,
            N_alpha=self.config.linesearch_steps,
        )

        self.data = create_engine_arrays(self.dims, self.device)

        if self.config.matrixfree_representation:
            self.A_op = MatrixFreeSystemOperator(self)
        else:
            self.A_op = MatrixSystemOperator(self)

        self.preconditioner = JacobiPreconditioner(self)

        self.data.set_generalized_mass(model)
        self.data.set_gravitational_acceleration(model)

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
            wp.copy(dest=self.data.lambda_prev, src=self.data._lambda)
            compute_linear_system(self.data, self.config, self.dims, dt)

            if not self.config.matrixfree_representation:
                self.A_op.update()

            self.preconditioner.update()

            cr_solver(
                A=self.A_op,
                b=self.data.b,
                x=self.data.delta_lambda,
                iters=self.config.linear_iters,
                preconditioner=self.preconditioner,
            )

            compute_delta_body_qd_from_delta_lambda(self.data, self.config, self.dims)

            if self.config.linesearch_steps > 0:
                perform_linesearch(self.data, self.config, self.dims, dt)

            update_variables(self.data, self.config, self.dims)

        wp.copy(dest=state_out.body_qd, src=self.data.body_qd)
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
