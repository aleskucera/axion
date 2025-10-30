from typing import Optional
from typing import Callable

import newton
import numpy as np
import scipy
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import NullLogger
from axion.optim import cr_solver
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from axion.types import compute_joint_constraint_offsets
from newton import Control
from newton import Model
from newton import State
from newton import Contacts
from newton.solvers import SolverBase

from .control_utils import apply_control
from .dense_utils import get_system_matrix_numpy
from .dense_utils import update_dense_matrices
from .engine_config import AxionEngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .general_utils import update_body_q
from .general_utils import update_variables
from .linear_utils import compute_dbody_qd_from_dbody_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch


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
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logger: Optional[HDF5Logger | NullLogger] = NullLogger(),
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
            linesearch_steps=self.config.linesearch_steps,
            joint_constraint_count=num_constraints,
        )

        allocate_dense_matrices = isinstance(self.logger, HDF5Logger)
        self.data = create_engine_arrays(
            self.dims,
            joint_constraint_offsets,
            self.device,
            allocate_dense_matrices,
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
        # self.data.set_joint_constraint_body_idx(model)

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

        self.logger.log_wp_dataset("h", self.data.h)
        self.logger.log_wp_dataset("J_values", self.data.J_values)
        self.logger.log_wp_dataset("C_values", self.data.C_values)
        self.logger.log_wp_dataset("constraint_body_idx", self.data.constraint_body_idx)

        self.logger.log_wp_dataset("body_f", self.data.body_f)
        self.logger.log_wp_dataset("body_q", self.data.body_q)
        self.logger.log_wp_dataset("body_u", self.data.body_u)
        self.logger.log_wp_dataset("body_u_prev", self.data.body_u_prev)

        self.logger.log_wp_dataset("body_lambda", self.data.body_lambda)
        self.logger.log_wp_dataset("body_lambda_prev", self.data.body_lambda_prev)

        self.logger.log_wp_dataset("dbody_qd", self.data.dbody_u)
        self.logger.log_wp_dataset("dbody_lambda", self.data.dbody_lambda)

        self.logger.log_wp_dataset("b", self.data.b)

        self.logger.log_struct_array("body_M", self.data.body_M)
        self.logger.log_struct_array("body_M_inv", self.data.body_M_inv)

        self.logger.log_struct_array("joint_constraint_data", self.data.joint_constraint_data)
        self.logger.log_struct_array("contact_interaction", self.data.contact_interaction)

        update_dense_matrices(self.data, self.config, self.dims)

        self.logger.log_wp_dataset("M_inv_dense", self.data.M_inv_dense)
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

        self.logger.log_wp_dataset("gen_mass", self.data.body_M)
        self.logger.log_wp_dataset("gen_inv_mass", self.data.body_M_inv)

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        """
        The primary method for running the physics simulation for a single time step.
        This method is an implementation of the abstract method from the base `Integrator` class in Warp.

        Args:
            model: The physics model containing bodies, joints, and other physics properties.
            state_in: The input state at the beginning of the time step.
            state_out: The output state at the end of the time step. This will be modified by the engine.
            dt: The time step duration.
            control: Optional control inputs to be applied during the simulation step.
        """
        newton.eval_ik(self.model, state_in, state_in.joint_q, state_in.joint_qd)
        apply_control(self.model, state_in, state_out, dt, control)
        self.init_state_fn(state_in, state_out, contacts, dt)
        self.data.update_state_data(self.model, state_in, state_out, contacts)

        self.data.body_lambda.zero_()

        for i in range(self.config.newton_iters):
            wp.record_event(self.events[i]["iter_start"])

            with self.logger.scope(f"newton_iteration_{i:02d}"):
                wp.copy(dest=self.data.body_lambda_prev, src=self.data.body_lambda)
                wp.copy(dest=self.data.s_n_prev, src=self.data.s_n)

                # --- Linearize the system of equations ---
                compute_linear_system(self.model, self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linearize"])

                if not self.config.matrixfree_representation:
                    self.A_op.update()
                self.preconditioner.update()

                # --- Solve linear system of equations ---
                self.data.dbody_lambda.zero_()
                wp.optim.linear.cg(
                    A=self.A_op,
                    b=self.data.b,
                    x=self.data.dbody_lambda,
                    atol=1e-5,
                    maxiter=self.config.linear_iters,
                    M=self.preconditioner,
                    check_every=0,
                    use_cuda_graph=True,
                )

                compute_dbody_qd_from_dbody_lambda(self.data, self.config, self.dims)
                wp.record_event(self.events[i]["lin_solve"])

                if self.config.linesearch_steps > 0:
                    perform_linesearch(self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linesearch"])

                update_variables(self.model, self.data, self.config, self.dims, dt)

                self._log_newton_iteration_data()

        update_body_q(self.model, self.data, self.config, self.dims, dt)
        wp.copy(dest=state_out.body_qd, src=self.data.body_u)
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
        """
        Apply the SciPy root-finding algorithm to the simulation.

        Args:
            model: The physics model containing bodies, joints, and other physics properties.
            state_in: The input state at the beginning of the time step.
            state_out: The output state at the end of the time step. This will be modified by the engine.
            dt: The time step duration.
            control: Optional control inputs to be applied during the simulation step.
            method: The scipy root-finding method to use (default is 'hybr').
            tolerance: The tolerance for convergence (default is 1e-10).
            max_iterations: The maximum number of iterations for the solver (default is 5000).
        """
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
