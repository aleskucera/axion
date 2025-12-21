from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import NullLogger

from .pca_utils import compute_pca_batch_h_norm
from .pca_utils import copy_grid_lambda_kernel
from .pca_utils import copy_grid_q_kernel
from .pca_utils import copy_grid_u_kernel
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
    time_control: bool = True
    time_initial_guess: bool = True
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

        # Events for the main simulator loop (collision detection and step)
        self.simulator_event_pairs = [
            {
                "collision_detection": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                "step": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
            }
            for _ in range(steps_per_segment)
        ]

        # Events for step-level timing blocks (control and initial guess)
        self.step_event_pairs = [
            {
                "control": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                "initial_guess": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
            }
            for _ in range(steps_per_segment)
        ]

        # Events for the engine's internal Newton solver
        self.engine_event_pairs = [
            [
                {
                    "system_linearization": (
                        wp.Event(enable_timing=True),
                        wp.Event(enable_timing=True),
                    ),
                    "linear_system_solve": (
                        wp.Event(enable_timing=True),
                        wp.Event(enable_timing=True),
                    ),
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
                    self.hdf5_logger.log_wp_dataset("h_d", engine.data.h.d)
                    self.hdf5_logger.log_wp_dataset("body_q", engine.data.body_q)
                    self.hdf5_logger.log_wp_dataset("body_u", engine.data.body_u)
                    self.hdf5_logger.log_wp_dataset("body_u_prev", engine.data.body_u_prev)
                    self.hdf5_logger.log_wp_dataset("body_f", engine.data.body_f)
                    self.hdf5_logger.log_struct_array("world_M", engine.data.world_M)
                    self.hdf5_logger.log_struct_array("world_M_inv", engine.data.world_M_inv)

            # Log joint constraints data
            if self.config.log_joint_constraint_data:
                with self.hdf5_logger.scope("Joint constraint data"):
                    self.hdf5_logger.log_wp_dataset("h", engine.data.h.c.j)
                    self.hdf5_logger.log_wp_dataset("J_values", engine.data.J_values.j)
                    self.hdf5_logger.log_wp_dataset("C_values", engine.data.C_values.j)
                    self.hdf5_logger.log_wp_dataset("body_lambda", engine.data.body_lambda.j)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_prev", engine.data.body_lambda_prev.j
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "constraint_body_idx", engine.data.constraint_body_idx.j
                    )
                    # self.hdf5_logger.log_struct_array(
                    #     "joint_constraint_data", engine.data.joint_constraint_data
                    # )

            # Log contact constraints data
            if self.config.log_contact_constraint_data:
                with self.hdf5_logger.scope("Contact constraint data"):
                    self.hdf5_logger.log_wp_dataset("h", engine.data.h.c.n)
                    self.hdf5_logger.log_wp_dataset("s", engine.data.s_n)
                    self.hdf5_logger.log_wp_dataset("J_values", engine.data.J_values.n)
                    self.hdf5_logger.log_wp_dataset("C_values", engine.data.C_values.n)
                    self.hdf5_logger.log_wp_dataset("body_lambda", engine.data.body_lambda.n)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_prev", engine.data.body_lambda_prev.n
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "constraint_body_idx", engine.data.constraint_body_idx.n
                    )
                    self.hdf5_logger.log_struct_array(
                        "contact_interaction", engine.data.contact_interaction
                    )

            # Log friction constraints data
            if self.config.log_friction_constraint_data:
                with self.hdf5_logger.scope("Friction constraint data"):
                    self.hdf5_logger.log_wp_dataset("h", engine.data.h.c.f)
                    self.hdf5_logger.log_wp_dataset("J_values", engine.data.J_values.f)
                    self.hdf5_logger.log_wp_dataset("C_values", engine.data.C_values.f)
                    self.hdf5_logger.log_wp_dataset("body_lambda", engine.data.body_lambda.f)
                    self.hdf5_logger.log_wp_dataset(
                        "body_lambda_prev", engine.data.body_lambda_prev.f
                    )
                    self.hdf5_logger.log_wp_dataset(
                        "constraint_body_idx", engine.data.constraint_body_idx.f
                    )
                    self.hdf5_logger.log_struct_array(
                        "contact_interaction", engine.data.contact_interaction
                    )

            # Log linear system data
            if self.config.log_linear_system_data:
                with self.hdf5_logger.scope("Linear system data"):
                    self.hdf5_logger.log_wp_dataset("b", engine.data.b)
                    self.hdf5_logger.log_wp_dataset("dbody_qd", engine.data.dbody_u)
                    self.hdf5_logger.log_wp_dataset("dbody_lambda", engine.data.dbody_lambda.full)

    def log_residual_norm_landscape(self, engine: AxionEngine):
        if not self.config.enable_hdf5_logging or not self.config.log_residual_norm_landscape:
            return

        # 1. Reconstruct Trajectory from History Arrays
        # State vector x = [q, u, lambda]

        # --- A. Body Positions (q) ---
        # shape: (Steps, Worlds, BodyCount) of wp.transform (7 floats)
        q_hist = wp.to_torch(engine.data.history.body_q_history)
        steps, n_w, n_b, _ = q_hist.shape
        # Flatten: (Steps, Worlds, N_b * 7)
        q_flat = q_hist.reshape(steps, n_w, n_b * 7)

        # --- B. Body Velocities (u) ---
        # shape: (Steps, Worlds, BodyCount) of wp.spatial_vector (6 floats)
        u_hist = wp.to_torch(engine.data.history.body_u_history)
        # Flatten: (Steps, Worlds, N_b * 6)
        u_flat = u_hist.reshape(steps, n_w, n_b * 6)

        # --- C. Constraint Impulses (lambda) ---
        # shape: (Steps, Worlds, N_c)
        lambda_hist = wp.to_torch(engine.data.history._body_lambda_history)

        # --- D. Full Trajectory Concatenation ---
        # shape: (Steps, Worlds, Dofs) where Dofs = (N_b*7 + N_b*6 + N_c)
        trajectory_all = torch.cat([q_flat, u_flat, lambda_hist], dim=2)

        # Residuals for norm calculation
        residuals_all = wp.to_torch(engine.data.history._h_history)

        if steps < 2:
            raise ValueError(
                f"Trajectory has {steps} points. PCA requires at least 2 points. "
                "Ensure 'newton_iters' is >= 2."
            )

        # 2. Compute Norms for Visualization
        trajectory_residual_norms = torch.norm(residuals_all, dim=2)

        # Warning threshold logic
        last_step_norms = trajectory_residual_norms[-1, :]
        max_residual = torch.max(last_step_norms)
        threshold = 1e-1
        if max_residual > threshold:
            print(
                f"Warning: Max residual norm ({max_residual.item():.6f}) exceeds limit ({threshold}) in timestep {engine._timestep}"
            )

        # 3. Perform PCA (Batched over Worlds)
        # v1, v2: (Worlds, Dofs) | S: (Worlds, 2)
        v1, v2, S, mean, std = perform_pca(trajectory_all)
        x_center = trajectory_all[-1]

        # 4. Project Trajectory to 2D
        # Project normalized deviations onto normalized eigenvectors
        vecs_from_center = trajectory_all - x_center.unsqueeze(0)
        vecs_normalized = vecs_from_center / std.unsqueeze(0)

        alpha_coords = (vecs_normalized * v1.unsqueeze(0)).sum(dim=2)
        beta_coords = (vecs_normalized * v2.unsqueeze(0)).sum(dim=2)
        trajectory_2d = torch.stack([alpha_coords, beta_coords], dim=2)

        # --- Auto-Scale Logic ---
        max_alpha = torch.max(torch.abs(alpha_coords), dim=0).values
        max_beta = torch.max(torch.abs(beta_coords), dim=0).values

        # Safety for zero movement
        max_alpha = torch.maximum(max_alpha, torch.tensor(1e-6, device=max_alpha.device))
        max_beta = torch.maximum(max_beta, torch.tensor(1e-6, device=max_beta.device))

        trajectory_bounds = torch.zeros_like(S)
        trajectory_bounds[:, 0] = max_alpha
        trajectory_bounds[:, 1] = max_beta
        auto_scale_margin = 1.2

        # 5. Create Grid Batch (Now includes Q)
        grid_res = self.config.pca_grid_res

        # Note: We pass the full dimensions object now so the function knows how to split N_b*7 vs N_b*6
        # Returns: pca_q, pca_u, pca_lambda, alphas, betas, ...
        pca_q, pca_u, pca_lambda, alphas, betas, _, _ = create_pca_grid_batch(
            x_center,
            v1,
            v2,
            trajectory_bounds,  # passing bounds as S
            mean,
            std,
            grid_res,
            auto_scale_margin,
            engine.dims,  # passing dims to handle q/u splitting
            engine.data.dt,
        )

        # 6. Copy Grid Points to Engine Data
        src_q = wp.from_torch(pca_q, dtype=wp.float32)
        src_u = wp.from_torch(pca_u, dtype=wp.float32)
        src_lambda = wp.from_torch(pca_lambda, dtype=wp.float32)

        grid_size = pca_q.shape[0]
        n_w = pca_q.shape[1]
        n_b = engine.dims.N_b
        n_c = engine.dims.N_c

        wp.launch(
            kernel=copy_grid_q_kernel,
            dim=(grid_size, n_w, n_b),
            inputs=[src_u],
            outputs=[engine.data.history.pca_batch_body_q],
            device=engine.data.device,
        )

        wp.launch(
            kernel=copy_grid_u_kernel,
            dim=(grid_size, n_w, n_b),
            inputs=[src_q],
            outputs=[engine.data.history.pca_batch_body_u],
            device=engine.data.device,
        )

        wp.launch(
            kernel=copy_grid_lambda_kernel,
            dim=(grid_size, n_w, n_c),
            inputs=[src_lambda],
            outputs=[engine.data.history.pca_batch_body_lambda.full],
            device=engine.data.device,
        )
        # 7. Compute Norms with Q-aware kernels
        # This will now trigger the dynamic mass matrix re-computation
        norm_results = compute_pca_batch_h_norm(
            engine.axion_model, engine.data, engine.config, engine.dims
        )

        residual_norm_grid = norm_results.reshape(grid_res, grid_res, n_w)

        # 8. Log Data
        with self.hdf5_logger.scope("residual_norm_landscape_data"):
            self.hdf5_logger.log_np_dataset("residual_norm_grid", residual_norm_grid.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_alphas", alphas.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_betas", betas.cpu().numpy())
            self.hdf5_logger.log_np_dataset("trajectory_2d_projected", trajectory_2d.cpu().numpy())

            self.hdf5_logger.log_np_dataset("pca_v1", v1.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_v2", v2.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_singular_values", S.cpu().numpy())
            self.hdf5_logger.log_np_dataset("pca_center_point", x_center.cpu().numpy())

            # Log history arrays including Q
            self.hdf5_logger.log_np_dataset("optimization_trajectory", trajectory_all.cpu().numpy())
            self.hdf5_logger.log_np_dataset(
                "body_q_history", engine.data.history.body_q_history.numpy()
            )
            self.hdf5_logger.log_np_dataset(
                "body_u_history", engine.data.history.body_u_history.numpy()
            )
            self.hdf5_logger.log_np_dataset(
                "body_lambda_history", engine.data.history._body_lambda_history.numpy()
            )

            self.hdf5_logger.log_np_dataset("trajectory_residuals", residuals_all.cpu().numpy())
            self.hdf5_logger.log_np_dataset(
                "trajectory_residual_norms", trajectory_residual_norms.cpu().numpy()
            )

            # Metadata
            dims = engine.dims
            # Add N_q (7 per body) to metadata if needed, otherwise dashboard assumes N_u
            dims_metadata = np.array([dims.N_u, dims.N_j, dims.N_n, dims.N_f], dtype=np.int32)
            self.hdf5_logger.log_np_dataset("simulation_dims", dims_metadata)

            metadata = np.array([grid_res, self.config.pca_plot_range_scale, steps, n_w])
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
        """
        Calculates, aggregates, and prints a statistical summary of timing info
        from the recorded events for the last segment.
        """
        if not self.config.enable_timing:
            return

        # 1. Aggregate all raw timing values
        timings: dict[str, list[float]] = {
            "collision_detection": [],
            "step": [],
            "control": [],
            "initial_guess": [],
            "system_linearization": [],
            "linear_system_solve": [],
            "linesearch": [],
        }

        for step in range(steps_per_segment):
            sim_events = self.simulator_event_pairs[step]
            timings["collision_detection"].append(
                wp.get_event_elapsed_time(*sim_events["collision_detection"])
            )
            timings["step"].append(wp.get_event_elapsed_time(*sim_events["step"]))

            step_events = self.step_event_pairs[step]
            timings["control"].append(wp.get_event_elapsed_time(*step_events["control"]))
            timings["initial_guess"].append(
                wp.get_event_elapsed_time(*step_events["initial_guess"])
            )

            for newton_iter in range(newton_iters):
                engine_events = self.engine_event_pairs[step][newton_iter]
                for key in ["system_linearization", "linear_system_solve", "linesearch"]:
                    timings[key].append(wp.get_event_elapsed_time(*engine_events[key]))

        # 2. Compute statistics for each timed operation
        stats: dict[str, dict] = {}
        for name, data in timings.items():
            if not data:
                continue
            data_np = np.array(data)
            stats[name] = {
                "mean": np.mean(data_np),
                "std": np.std(data_np),
                "v_min": np.min(data_np),
                "v_max": np.max(data_np),
                "count": len(data_np),
            }

        if not stats:
            return

        # 3. Calculate Derived Totals
        total_physics_step_time = stats.get("collision_detection", {}).get("mean", 0.0) + stats.get(
            "step", {}
        ).get("mean", 0.0)
        physics_update_total = stats.get("step", {}).get("mean", 0.0)
        avg_newton_iters = newton_iters
        per_newton_iter_total = sum(
            stats.get(op, {}).get("mean", 0.0)
            for op in ["system_linearization", "linear_system_solve", "linesearch"]
        )
        total_newton_time = per_newton_iter_total * avg_newton_iters

        def format_stat(stat_data, parent_total, is_summary=False):
            if not stat_data:
                return {
                    "mean_s": "N/A",
                    "std_s": "N/A",
                    "range_s": "N/A",
                    "perc_s": "N/A",
                    "count_s": "N/A",
                }

            mean = stat_data.get("mean", 0)
            std = stat_data.get("std", 0)
            v_min = stat_data.get("v_min", 0)
            v_max = stat_data.get("v_max", 0)
            count = stat_data.get("count", 0)

            mean_s = f"{mean:.3f} ms"
            std_s = f"± {std:.3f}" if not is_summary else "N/A"
            range_s = f"{v_min:.3f} - {v_max:.3f}" if not is_summary else "N/A"
            perc_s = f"{(mean / parent_total * 100):.1f}%" if parent_total > 0 else "0.0%"
            count_s = str(count)
            return {
                "mean_s": mean_s,
                "std_s": std_s,
                "range_s": range_s,
                "perc_s": perc_s,
                "count_s": count_s,
            }

        # Collect data for DataFrame
        data = []

        # Total
        s = format_stat(
            {"mean": total_physics_step_time, "count": stats["step"]["count"]},
            total_physics_step_time,
            is_summary=True,
        )
        data.append(
            {
                "Operation": "TOTAL PHYSICS STEP",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Collision
        s = format_stat(stats.get("collision_detection"), total_physics_step_time)
        data.append(
            {
                "Operation": "├─ collision_detection",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Physics update
        s = format_stat(stats.get("step"), total_physics_step_time)
        data.append(
            {
                "Operation": "└─ physics_update",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Control
        s = format_stat(stats.get("control"), physics_update_total)
        data.append(
            {
                "Operation": "   ├─ control",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Initial guess
        s = format_stat(stats.get("initial_guess"), physics_update_total)
        data.append(
            {
                "Operation": "   ├─ initial_guess",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Newton iterations
        s = format_stat(
            {
                "mean": total_newton_time,
                "count": stats.get("system_linearization", {}).get("count", 0),
            },
            physics_update_total,
            is_summary=True,
        )
        data.append(
            {
                "Operation": f"   └─ newton_iterations ({avg_newton_iters})",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Linearization
        s = format_stat(stats.get("system_linearization"), per_newton_iter_total)
        data.append(
            {
                "Operation": "      ├─ system_linearization",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Solve
        s = format_stat(stats.get("linear_system_solve"), per_newton_iter_total)
        data.append(
            {
                "Operation": "      ├─ linear_system_solve",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        # Linesearch
        s = format_stat(stats.get("linesearch"), per_newton_iter_total)
        data.append(
            {
                "Operation": "      └─ linesearch",
                "Mean": s["mean_s"],
                "Std Dev": s["std_s"],
                "Min - Max": s["range_s"],
                "% Total": s["perc_s"],
                "Samples": s["count_s"],
            }
        )

        df = pd.DataFrame(data)

        # Format the table with better spacing and alignment
        print("\nTIMING PERFORMANCE REPORT\n")

        # Custom formatting for better readability
        col_widths = {
            "Operation": 35,
            "Mean": 10,
            "Std Dev": 12,
            "Min - Max": 15,
            "% Total": 8,
            "Samples": 8,
        }

        # Print header
        header = ""
        for col, width in col_widths.items():
            header += f"{col:<{width}}"
        print(header)
        print("-" * sum(col_widths.values()))

        # Print data rows
        for _, row in df.iterrows():
            line = ""
            for col, width in col_widths.items():
                line += f"{str(row[col]):<{width}}"
            print(line)

        print()
