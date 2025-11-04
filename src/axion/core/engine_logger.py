from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import NullLogger

from .dense_utils import get_system_matrix_numpy
from .dense_utils import update_dense_matrices
from .pca_utils import compute_pca_batch_h_norm
from .pca_utils import create_pca_grid_batch
from .pca_utils import perform_pca

if TYPE_CHECKING:
    from .engine import AxionEngine


@dataclass(frozen=True)
class LoggingConfig:
    enable_timing: bool = False
    enable_hdf5_logging: bool = False

    # Timing options
    time_whole_step: bool = True
    time_collision_detection: bool = True
    time_engine_step: bool = True
    time_newton_iteration: bool = True
    time_linearization: bool = True
    time_linear_solve: bool = True
    time_linesearch: bool = True

    # HDF5 logging options
    log_dynamics_state: bool = True
    log_linear_system_data: bool = True

    log_joint_constraint_data: bool = True
    log_contact_constraint_data: bool = True
    log_friction_constraint_data: bool = True

    log_residual_norm_landscape: bool = True
    pca_grid_res: int = 100
    pca_plot_range_scale: float = 3.0

    hdf5_log_file: str = "simulation_log.h5"

    def __post_init__(self):
        """Validate all configuration parameters."""

        def _validate_positive_int(value: int, name: str, min_value: int = 1) -> None:
            """Validate that a value is a positive integer >= min_value."""
            if value < min_value:
                raise ValueError(f"{name} must be >= {min_value}, got {value}")

        def _validate_non_negative_float(value: float, name: str) -> None:
            """Validate that a value is a non-negative float."""
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        def _validate_unit_interval(value: float, name: str) -> None:
            """Validate that a value is in the unit interval [0, 1]."""
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        def _validate_non_negative_int(value: int, name: str) -> None:
            """Validate that a value is a non-negative integer."""
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        _validate_positive_int(self.pca_grid_res, "pca_grid_res")
        _validate_non_negative_float(self.pca_plot_range_scale, "pca_plot_range_scale")


class EngineLogger:
    def __init__(self, config: LoggingConfig) -> None:
        self.config = config

        self.hdf5_logger = NullLogger()
        if self.config.enable_hdf5_logging:
            self.hdf5_logger = HDF5Logger(filepath=config.hdf5_log_file)

        self.simulator_event_pairs = []
        self.engine_event_pairs = []
        self._current_step_in_segment = 0

    @property
    def can_cuda_graph_be_used(self):
        if self.config.enable_hdf5_logging:
            return False
        return True

    @property
    def current_step_in_segment(self) -> int:
        """The current physics substep index within the segment."""
        return self._current_step_in_segment

    @property
    def uses_dense_matrices(self):
        return self.config.enable_hdf5_logging and self.config.log_linear_system_data

    @property
    def uses_pca_arrays(self):
        return self.config.enable_hdf5_logging and self.config.log_residual_norm_landscape

    # --- NEW: Setter method ---
    def set_current_step_in_segment(self, step_num: int):
        """
        Sets the current step index. Called by the simulator before each physics step.
        """
        self._current_step_in_segment = step_num

    def initialize_events(self, steps_per_segment: int, newton_iters: int):
        """Creates pairs of (start, end) events for each timed block."""

        # Events for the main simulator loop (collision, integration)
        self.simulator_event_pairs = [
            {
                "collision": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                "integration": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
            }
            for _ in range(steps_per_segment)
        ]

        # Events for the engine's internal Newton solver
        self.engine_event_pairs = [
            [
                {
                    "linearize": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                    "lin_solve": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                    "linesearch": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                }
                for _ in range(newton_iters)
            ]
            for _ in range(steps_per_segment)
        ]

    @contextmanager
    def timed_block(self, start_event: wp.Event, end_event: wp.Event):
        """A context manager to record start/end events for a code block."""
        if self.config.enable_timing:
            wp.record_event(start_event)
            yield
            wp.record_event(end_event)
        else:
            yield

    def open(self):
        if not self.config.enable_hdf5_logging:
            return

        self.hdf5_logger.open()

    def close(self):
        if not self.config.enable_hdf5_logging:
            return

        self.hdf5_logger.close()

    def log_newton_iteration_data(self, engine: AxionEngine, iteration: int):
        if not self.config.enable_hdf5_logging:
            return

        with self.hdf5_logger.scope(f"newton_iteration_{iteration:02d}"):
            # Log dynamics state
            if self.config.log_dynamics_state:
                with self.hdf5_logger.scope("Dynamics state"):
                    self.hdf5_logger.log_wp_dataset("h_d", engine.data.h_d)
                    self.hdf5_logger.log_wp_dataset("body_q", engine.data.body_q)
                    self.hdf5_logger.log_wp_dataset("body_u", engine.data.body_u)
                    self.hdf5_logger.log_wp_dataset("body_u_prev", engine.data.body_u_prev)
                    self.hdf5_logger.log_wp_dataset("body_f", engine.data.body_f)
                    self.hdf5_logger.log_struct_array("body_M", engine.data.body_M)
                    self.hdf5_logger.log_struct_array("body_M_inv", engine.data.body_M_inv)
                    self.hdf5_logger.log_struct_array("world_M", engine.data.world_M)
                    self.hdf5_logger.log_struct_array("world_M_inv", engine.data.world_M_inv)

            # Log joint constraints data
            if self.config.log_joint_constraint_data:
                with self.hdf5_logger.scope("Joint constraint data"):
                    self.hdf5_logger.log_wp_dataset("h_j", engine.data.h_j)
                    self.hdf5_logger.log_wp_dataset("J_j_values", engine.data.J_j_values)
                    self.hdf5_logger.log_wp_dataset("C_j_values", engine.data.C_j_values)
                    self.hdf5_logger.log_wp_dataset("body_lambda_j", engine.data.body_lambda_j)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_j_prev", engine.data.body_lambda_j_prev
                    )
                    self.hdf5_logger.log_struct_array(
                        "joint_constraint_data", engine.data.joint_constraint_data
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "joint_constraint_body_idx", engine.data.joint_constraint_body_idx
                    )

            # Log contact constraints data
            if self.config.log_contact_constraint_data:
                with self.hdf5_logger.scope("Contact constraint data"):
                    self.hdf5_logger.log_wp_dataset("h_n", engine.data.h_n)
                    self.hdf5_logger.log_wp_dataset("s_n", engine.data.s_n)
                    self.hdf5_logger.log_wp_dataset("J_n_values", engine.data.J_n_values)
                    self.hdf5_logger.log_wp_dataset("C_n_values", engine.data.C_n_values)
                    self.hdf5_logger.log_wp_dataset("body_lambda_n", engine.data.body_lambda_n)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_n_prev", engine.data.body_lambda_n_prev
                    )
                    self.hdf5_logger.log_struct_array(
                        "contact_interaction", engine.data.contact_interaction
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "contact_constraint_body_idx", engine.data.contact_constraint_body_idx
                    )

            # Log friction constraints data
            if self.config.log_friction_constraint_data:
                with self.hdf5_logger.scope("Friction constraint data"):
                    self.hdf5_logger.log_wp_dataset("h_f", engine.data.h_f)
                    self.hdf5_logger.log_wp_dataset("J_f_values", engine.data.J_f_values)
                    self.hdf5_logger.log_wp_dataset("C_f_values", engine.data.C_f_values)
                    self.hdf5_logger.log_wp_dataset("body_lambda_f", engine.data.body_lambda_f)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_f_prev", engine.data.body_lambda_f_prev
                    )
                    self.hdf5_logger.log_struct_array(
                        "contact_interaction", engine.data.contact_interaction
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "friction_constraint_body_idx", engine.data.friction_constraint_body_idx
                    )

            # Log linear system data
            if self.config.log_linear_system_data:
                with self.hdf5_logger.scope("Linear system data"):
                    self.hdf5_logger.log_wp_dataset("b", engine.data.b)
                    self.hdf5_logger.log_wp_dataset("dbody_qd", engine.data.dbody_u)
                    self.hdf5_logger.log_wp_dataset("dbody_lambda", engine.data.dbody_lambda)

                    update_dense_matrices(engine.data, engine.config, engine.dims)

                    self.hdf5_logger.log_wp_dataset("M_inv_dense", engine.data.M_inv_dense)
                    self.hdf5_logger.log_wp_dataset("J_dense", engine.data.J_dense)
                    self.hdf5_logger.log_wp_dataset("C_dense", engine.data.C_dense)

                    if not engine.config.matrixfree_representation:
                        self.hdf5_logger.log_wp_dataset("A", self.A_op._A)
                    else:
                        A_np = get_system_matrix_numpy(engine.data, engine.config, engine.dims)
                        self.hdf5_logger.log_np_dataset("A", A_np)
                        self.hdf5_logger.log_scalar("cond_number", np.linalg.cond(A_np))

    def log_residual_norm_landscape(self, engine: AxionEngine):
        if not self.config.enable_hdf5_logging or not self.config.log_residual_norm_landscape:
            return

        trajectory = wp.to_torch(engine.data.optim_trajectory)
        trajectory_residuals = wp.to_torch(engine.data.optim_h)
        trajectory_residual_norms = torch.norm(trajectory_residuals, dim=1)
        if len(trajectory) < 2:
            raise ValueError(
                f"Trajectory has {len(trajectory)} points. PCA requires at least 2 points to find a direction. "
                "Ensure 'newton_iters' is >= 2."
            )

        # 1. Get parameters and data from the engine
        grid_res = engine.config.pca_grid_res
        plot_scale = engine.config.pca_plot_range_scale

        # Center the visualization plane on the final solution point
        x_center = trajectory[-1]

        # 2. Perform PCA on the trajectory to find the two most important directions
        v1, v2, S = perform_pca(trajectory)

        # 3. Project the high-dimensional trajectory into the 2D PCA space for plotting
        vecs_from_center = trajectory - x_center
        alpha_coords = vecs_from_center @ v1  # Project onto v1
        beta_coords = vecs_from_center @ v2  # Project onto v2
        trajectory_2d = torch.stack([alpha_coords, beta_coords], dim=1)

        # 4. Create the batch of grid points in the high-dimensional space
        pca_u, pca_lambda, alphas, betas, alpha_grid, beta_grid = create_pca_grid_batch(
            x_center,
            v1,
            v2,
            S,
            grid_res,
            plot_scale,
            engine.dims.N_u,
        )

        # Copy the grid points to the Warp arrays for computation
        wp.copy(dest=engine.data.pca_batch_body_u, src=wp.from_torch(pca_u.contiguous()))
        wp.copy(dest=engine.data.pca_batch_body_lambda, src=wp.from_torch(pca_lambda.contiguous()))

        # 5. Compute the loss (residual norm) for every point on the grid
        pca_batch_h_norm = compute_pca_batch_h_norm(
            engine.model,
            engine.data,
            engine.config,
            engine.dims,
        )
        residual_norm_landscape = pca_batch_h_norm.reshape(grid_res, grid_res)

        # 6. Store all data to HDF5 file
        with self.hdf5_logger.scope("residual_norm_landscape_data"):
            # Store the core visualization data
            self.hdf5_logger.log_np_dataset(
                "residual_norm_grid", residual_norm_landscape.cpu().numpy()
            )
            self.hdf5_logger.log_np_dataset("pca_alphas", alphas.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_betas", betas.cpu().numpy())
            self.hdf5_logger.log_np_dataset("trajectory_2d_projected", trajectory_2d.cpu().numpy())

            # Store PCA components and metadata
            self.hdf5_logger.log_np_dataset("pca_v1", v1.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_v2", v2.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_singular_values", S.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_center_point", x_center.cpu().numpy())

            # Store complete trajectory for additional analysis
            self.hdf5_logger.log_np_dataset("optimization_trajectory", trajectory.cpu().numpy())
            self.hdf5_logger.log_np_dataset(
                "trajectory_residuals", trajectory_residuals.cpu().numpy()
            )
            self.hdf5_logger.log_np_dataset(
                "trajectory_residual_norms", trajectory_residual_norms.cpu().numpy()
            )

            # Store metadata
            metadata = np.array(
                [
                    grid_res,  # Grid resolution
                    plot_scale,  # Plot range scale
                    len(trajectory),  # Number of Newton iterations
                ]
            )
            self.hdf5_logger.log_np_dataset("pca_metadata", metadata)

    def timestep_start(self, timestep: int, current_time: float):
        """Start logging for a timestep and log the current time."""
        if not self.config.enable_hdf5_logging:
            return

        self._timestep_scope = self.hdf5_logger.scope(f"timestep_{timestep:04d}")
        self._timestep_scope.__enter__()
        self.hdf5_logger.log_scalar("time", current_time)

    def timestep_end(self):
        """End logging for the current timestep."""
        if not self.config.enable_hdf5_logging or not hasattr(self, "_timestep_scope"):
            return

        self._timestep_scope.__exit__(None, None, None)
        del self._timestep_scope

    def log_contact_count(self, contacts):
        """Log rigid contact count."""
        if not self.config.enable_hdf5_logging:
            return

        self.hdf5_logger.log_wp_dataset("rigid_contact_count", contacts.rigid_contact_count)

    def log_segment_timings(self, steps_per_segment: int, newton_iters: int):
        """Calculates, prints, and logs timing info from the recorded events."""
        if not self.config.enable_timing:
            return

        for step in range(steps_per_segment):
            sim_events = self.simulator_event_pairs[step]

            collision_time = wp.get_event_elapsed_time(*sim_events["collision"])
            integration_time = wp.get_event_elapsed_time(*sim_events["integration"])

            print(
                f"\t- SUBSTEP {step}: collision detection took {collision_time:.03f} ms "
                f"and integration step took {integration_time:0.3f} ms."
            )

            if not self.config.time_newton_iteration:
                continue

            for newton_iter in range(newton_iters):
                # Access the events using both the step and iteration index
                engine_events = self.engine_event_pairs[step][newton_iter]

                linearize_time = wp.get_event_elapsed_time(*engine_events["linearize"])
                lin_solve_time = wp.get_event_elapsed_time(*engine_events["lin_solve"])
                linesearch_time = wp.get_event_elapsed_time(*engine_events["linesearch"])

                print(
                    f"\t\t- NEWTON ITERATION {newton_iter}: Linearization took {linearize_time:.03f} ms, "
                    f"linear solve took {lin_solve_time:.03f} ms and linesearch took {linesearch_time:.03f} ms."
                )
