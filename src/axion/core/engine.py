"""
Implements the core Axion physics engine, a high-performance solver based on Warp.

This module contains the `AxionEngine`, a low-level physics integrator that solves the equations
of motion for rigid body systems with contacts and joints. This approach formulates all
physical laws—including dynamics, collision response, joints, and friction—as a single,
large-scale Differential Variational Inequality (DVI).

After time discretization, the DVI becomes a root-finding problem for a non-smooth system of
equations. The engine solves this problem directly using a custom Non-Smooth Newton method,
which is the key to its stability, accuracy, and performance, especially for highly-coupled
systems like articulated robots.
"""
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
from .engine_config import EngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .general_utils import update_body_q
from .general_utils import update_variables
from .linear_utils import compute_delta_body_qd_from_delta_lambda
from .linear_utils import compute_linear_system
from .linesearch_utils import perform_linesearch


class AxionEngine(Integrator):
    """
    A physics engine implementing a non-smooth Newton method for rigid body dynamics.

    This class serves as the core solver, implementing the `warp.sim.Integrator` interface.
    It takes the full description of a physics scene (`warp.sim.Model`) and advances it in time.

    The engine formulates the entire physics state—including body dynamics, contacts, and joints—as a
    single, unified problem at each time step. This monolithic approach, solved with a robust
    non-smooth Newton method, provides exceptional stability and precision, particularly for
    complex, highly-constrained systems where traditional, sequential solvers might struggle.

    Attributes:
        model (Model): The Warp simulation model defining the scene.
        config (EngineConfig): Configuration parameters for the solver.
        logger (HDF5Logger | NullLogger): Logger for recording simulation data.
        dims (EngineDimensions): Dataclass holding the dimensions of all vectors and matrices.
        data (EngineArrays): A structure holding all GPU arrays used by the solver.
        A_op (MatrixSystemOperator | MatrixFreeSystemOperator): The linear system operator for `Ax=b`.
        preconditioner (JacobiPreconditioner): Preconditioner for the linear solver.
        events (list): A list of Warp events used for fine-grained performance profiling.
    """

    def __init__(
        self,
        model: Model,
        config: Optional[EngineConfig],
        logger: Optional[HDF5Logger | NullLogger],
    ):
        """
        Initializes the Axion physics engine.

        This constructor sets up all necessary data structures on the GPU for the given model
        and solver configuration. It allocates memory for state vectors, Jacobians, residuals,
        and other intermediate quantities used within the non-smooth Newton iterations.

        Args:
            model: The `warp.sim.Model` containing bodies, joints, and other physics properties.
            config: An `EngineConfig` object with parameters for the solver, such as the number
                of Newton iterations and linear solver settings.
            logger: An optional `HDF5Logger` or `NullLogger` for recording simulation data.
        """
        super().__init__()
        self.device = model.device

        self.model = model
        self.logger = logger if logger else NullLogger()
        self.config = config if config else EngineConfig()

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
        """
        Helper method to log internal solver data for a single Newton iteration.

        If the logger is an HDF5Logger, this function records the state of key solver
        variables, such as residuals, Jacobians, constraint impulses, and body states.
        This is invaluable for debugging and analyzing the solver's convergence behavior.
        """
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

    def _log_static_data(self):
        """
        Helper method to log data that is static throughout the simulation.

        If the logger is an HDF5Logger, this function records data that is initialized
        once and does not change, such as the generalized mass matrix.
        """
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
        """
        Advances the physics simulation by a single time step using the Non-Smooth Newton method.

        This is the primary entry point for the solver. It implements the abstract method from the
        base `warp.sim.Integrator` class. The method takes the system state at time `t` (`state_in`)
        and computes the state at `t + dt` (`state_out`), respecting all physical constraints.

        The core logic is a loop of Newton iterations:

        1.  **Linearize**: Formulate a linear system `Ax = b` that approximates the non-linear,
            non-smooth dynamics and constraint equations.
        2.  **Solve**: Solve the linear system for the search direction `x = [delta_lambda, delta_qd]`,
            typically using a preconditioned iterative solver like Conjugate Residual.
        3.  **Line Search**: Find an appropriate step size `alpha` along the search direction that
            ensures progress towards the solution and respects complementarity constraints.
        4.  **Update**: Update the current estimates for constraint impulses `lambda` and velocities `qd`.

        After the Newton iterations converge, the final velocities are integrated to update positions.

        Args:
            model: The physics model specifying the scene's bodies, joints, and contacts.
            state_in: The input state at the beginning of the time step.
            state_out: The output state structure to be populated with the results at `t + dt`.
            dt: The time step duration in seconds.
            control: Optional control inputs (forces/torques) to be applied.

        Returns:
            A list of Warp events used for detailed performance profiling of solver stages.
        """
        apply_control(model, state_in, state_out, dt, control)
        # Note: 'integrate_bodies' is a legacy name; here it primarily applies gravity
        # and initializes the 'body_f' force accumulators.
        self.integrate_bodies(model, state_in, state_out, dt)
        self.data.update_state_data(model, state_in)

        # TODO: Implement and evaluate warm-starting strategies
        # self.data._lambda.zero_()

        for i in range(self.config.newton_iters):
            wp.record_event(self.events[i]["iter_start"])

            with self.logger.scope(f"newton_iteration_{i:02d}"):
                wp.copy(dest=self.data.lambda_prev, src=self.data._lambda)

                # --- 1. Linearize the non-smooth system of equations ---
                compute_linear_system(self.model, self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linearize"])

                # Update operators with newly computed matrix/preconditioner data
                if not self.config.matrixfree_representation:
                    self.A_op.update()
                self.preconditioner.update()

                # --- 2. Solve the linear system Ax=b for the Newton step ---
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

                # --- 3. Perform line search to find a suitable step size ---
                if self.config.linesearch_steps > 0:
                    perform_linesearch(self.model, self.data, self.config, self.dims, dt)
                wp.record_event(self.events[i]["linesearch"])

                # --- 4. Update the state variables with the scaled Newton step ---
                update_variables(self.model, self.data, self.config, self.dims, dt)

                self._log_newton_iteration_data()

        # Final integration of velocity to get new positions
        update_body_q(self.model, self.data, dt)
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
        """
        (For Debugging) Solves the simulation step using a SciPy root-finding algorithm.

        This method replaces the custom GPU-based non-smooth Newton solver with a standard
        numerical solver from the SciPy library. It is intended for validation and debugging
        purposes to verify the correctness of the residual function (`compute_linear_system`).

        This function is significantly slower than the primary `simulate` method due to:

        - Data transfers between GPU and CPU on every function evaluation.
        - The solver itself running on the CPU.
        - Python overhead.

        It should not be used for performance-critical simulations.

        Args:
            model: The physics model.
            state_in: The input state at the beginning of the time step.
            state_out: The output state at the end of the time step.
            dt: The time step duration.
            control: Optional control inputs.
            method: The SciPy root-finding method to use (e.g., 'hybr', 'lm').
            tolerance: The convergence tolerance for the solver.
            max_iterations: The maximum number of iterations for the SciPy solver.
        """
        apply_control(model, state_in, state_out, dt, control)
        self.integrate_bodies(model, state_in, state_out, dt)
        self.data.update_state_data(model, state_in)

        def residual_function(x: np.ndarray) -> np.ndarray:
            """Computes the full system residual F(x) for the SciPy solver."""
            wp.copy(dest=self.data.lambda_prev, src=self.data._lambda)

            # Deconstruct the input vector x into lambda and body_qd
            n_lambda = self.dims.con_dim
            lambda_vals = x[:n_lambda]
            body_qd_vals = np.reshape(x[n_lambda:], (self.dims.N_b, 6))

            # Store current GPU state to restore it later
            lambda_backup = wp.clone(self.data._lambda)
            body_qd_backup = wp.clone(self.data.body_qd)

            try:
                # Set GPU kernel inputs from the solver's guess vector x
                self.data._lambda.assign(lambda_vals)
                self.data.body_qd.assign(body_qd_vals)

                # Compute the system residual `res` based on the current guess
                compute_linear_system(self.model, self.data, self.config, self.dims, dt)

                # Return the residual F(x) back to the SciPy solver
                return self.data.res.numpy()

            finally:
                # Restore original GPU state to not interfere with solver's internal state
                wp.copy(dest=self.data._lambda, src=lambda_backup)
                wp.copy(dest=self.data.body_qd, src=body_qd_backup)

        # Initial guess from the current simulation state
        x0 = np.concatenate([self.data._lambda.numpy(), self.data.body_qd.numpy().flatten()])

        # Call SciPy to find the root of the residual function
        result = scipy.optimize.root(
            residual_function,
            x0,
            method=method,
            options={"xtol": tolerance, "maxfev": max_iterations},
        )

        if not result.success:
            print(f"Warning: SciPy solver did not converge. Message: {result.message}")

        # Update the engine's state with the solution found by SciPy
        n_lambda = self.dims.con_dim
        self.data._lambda.assign(wp.from_numpy(result.x[:n_lambda].astype(np.float32)))
        body_qd_solution = np.reshape(result.x[n_lambda:], (self.dims.N_b, 6))
        self.data.body_qd.assign(wp.from_numpy(body_qd_solution.astype(np.float32)))

        # Finalize the state update
        update_body_q(self.model, self.data, dt)
        wp.copy(dest=state_out.body_qd, src=self.data.body_qd)
        wp.copy(dest=state_out.body_q, src=self.data.body_q)
