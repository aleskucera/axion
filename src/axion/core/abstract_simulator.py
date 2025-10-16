"""
An abstract base class orchestrating a Warp-based physics simulation using the Axion engine.

This module provides the `AbstractSimulator` class, which serves as the main driver for running physics simulations.
It handles the main simulation loop, timing, rendering, performance profiling, and execution strategy.
The class is designed to be subclassed, with the user only needing to implement the `build_model` method
to define the specific physics scene.

The simulator's architecture is built around the concept of "segments." A segment is a block of multiple
physics timesteps. When rendering is enabled, a segment corresponds to the number of simulation steps
that fit into a single render frame. In headless mode, it's a user-defined chunk of steps. This segmented
approach allows for performance optimizations like CUDA graphs, where an entire segment of operations can be
captured and replayed with minimal CPU overhead.
"""
import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import warp as wp
import warp.sim.render
from axion.logging import HDF5Logger
from axion.logging import NullLogger
from tqdm import tqdm

from .engine import AxionEngine
from .engine_config import EngineConfig


# --- Configuration Data Classes ---


@dataclass
class SimulationConfig:
    """
    Parameters defining the simulation's timeline and core physics rate.

    Attributes:
        duration_seconds: The total desired duration of the simulation in seconds. The actual
            duration may be slightly adjusted to align with segment boundaries.
        target_timestep_seconds: The desired physics time step (dt) in seconds. If rendering is
            enabled, this value may be adjusted to ensure an integer number of simulation
            steps occur per render frame.
    """

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3


@dataclass
class RenderingConfig:
    """
    Parameters for rendering the simulation to a USD file for visualization.

    Attributes:
        enable: If `True`, enables rendering and saves the output to a USD file.
        fps: The frame rate of the output video or animation. The simulation timestep will
            be adjusted to align with this rate.
        scaling: A scaling factor applied to all objects in the rendered USD scene.
        usd_file: The path to the output USD file.
    """

    enable: bool = True
    fps: int = 30
    scaling: float = 100.0
    usd_file: str = "sim.usd"


@dataclass
class ExecutionConfig:
    """
    Parameters controlling the performance and execution strategy of the simulation.

    Attributes:
        use_cuda_graph: If `True` and running on a CUDA-capable device, captures the simulation
            segment into a CUDA graph. This significantly reduces CPU overhead and improves
            performance by replaying the graph instead of launching kernels individually.
            This is automatically disabled if HDF5 logging is on.
        headless_steps_per_segment: When rendering is disabled, this determines how many
            simulation steps are grouped into a single execution segment.
    """

    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10


@dataclass
class ProfilingConfig:
    """
    Parameters for debugging, performance timing, and detailed data logging.

    Attributes:
        enable_timing: If `True`, enables `wp.ScopedTimer` and prints detailed performance
            timings for each simulation segment to the console.
        enable_hdf5_logging: If `True`, logs detailed simulation state data at each
            timestep to an HDF5 file. Enabling this disables `use_cuda_graph` optimization
            as it requires CPU-side intervention at every step.
        hdf5_log_file: The path to the output HDF5 log file.
    """

    enable_timing: bool = False
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"


# --- Timing Calculation Helpers ---


def _calculate_render_aligned_timestep(
    target_timestep_seconds: float, fps: int
) -> tuple[float, int]:
    """
    Calculates an effective timestep that aligns perfectly with render frame duration.

    To avoid visual stutter or aliasing, the simulation timestep (`dt`) should be a clean divisor
    of the render frame duration (1 / fps). This function finds the number of integer simulation
    steps that best fits into one render frame and calculates the required `dt` to make them match.

    Args:
        target_timestep_seconds: The user-requested simulation timestep.
        fps: The target rendering frames per second.

    Returns:
        A tuple containing:
            - The adjusted, effective timestep in seconds.
            - The number of simulation steps per render frame.
    """
    frame_duration = 1.0 / fps
    ideal_steps_per_frame = frame_duration / target_timestep_seconds
    steps_per_frame = round(ideal_steps_per_frame) or 1
    effective_timestep = frame_duration / steps_per_frame

    adj_ratio = abs(effective_timestep - target_timestep_seconds) / target_timestep_seconds
    if adj_ratio > 0.01:
        print(
            f"\nINFO: Target timestep adjusted to {1000 * effective_timestep:.3f}ms"
            f" for rendering. ({100 * adj_ratio:.3}% change)"
        )

    return effective_timestep, steps_per_frame


def _align_duration_to_segment(
    target_duration: float, timestep: float, steps_per_segment: int
) -> tuple[float, int]:
    """
    Adjusts total simulation duration to be a whole multiple of the segment duration.

    The main simulation loop processes time in "segments" (a fixed number of steps). This
    function ensures the total simulation duration is an integer number of these segments,
    simplifying the main loop logic.

    Args:
        target_duration: The user-requested total simulation duration.
        timestep: The effective simulation timestep (dt).
        steps_per_segment: The number of simulation steps in one segment.

    Returns:
        A tuple containing:
            - The adjusted, effective total simulation duration.
            - The total number of segments the simulation will run for.
    """
    segment_duration = timestep * steps_per_segment
    num_segments = math.ceil(target_duration / segment_duration)
    total_sim_steps = num_segments * steps_per_segment
    effective_duration = total_sim_steps * timestep

    adj_ratio = abs(effective_duration - target_duration) / target_duration
    if adj_ratio > 0.01:
        print(
            f"\nINFO: Simulation duration adjusted to {effective_duration:.4f}s "
            f"to align with segment size. ({100 * adj_ratio:.3}% change)"
        )
    return effective_duration, num_segments


# --- Main Simulator Class ---


class AbstractSimulator(ABC):
    """
    An abstract base class for running a Warp-based physics simulation with the Axion engine.

    This class provides the complete orchestration for a physics simulation, including the
    main loop, state management, timing alignment, rendering, profiling, and execution strategy
    (iterative-step vs. CUDA graph).

    To create a concrete simulation, a user must inherit from this class and implement the
    `build_model` method, which is responsible for defining the bodies, joints, and initial
    conditions of the physics scene.

    The core of the simulation loop is the `_single_physics_step` method, which first performs
    collision detection and then invokes the `AxionEngine`. The engine solves the unified
    constraint problem for all contacts, joints, and friction simultaneously using a non-smooth
    Newton method, advancing the system state by one time step.

    Attributes:
        simulation_config: Configuration for the simulation timeline.
        rendering_config: Configuration for USD rendering.
        execution_config: Configuration for performance and execution strategy.
        profiling_config: Configuration for debugging and logging.
        engine_config: Configuration for the low-level Axion physics solver.
        logger: An instance of a logger (`HDF5Logger` or `NullLogger`) for data logging.
        steps_per_segment: The number of simulation steps grouped into one execution block.
        num_segments: The total number of segments to run for the full simulation duration.
        effective_timestep: The actual timestep (dt) used, possibly adjusted for rendering.
        effective_duration: The actual total simulation duration, adjusted to fit segments.
        model: The `wp.sim.Model` instance representing the physics scene.
        integrator: The `AxionEngine` instance that performs the physics integration.
        current_state: The `wp.sim.State` at the beginning of a timestep.
        next_state: The `wp.sim.State` computed at the end of a timestep.
        control: The `wp.sim.Control` structure for applying external forces/torques.
        renderer: An optional `wp.sim.render.SimRenderer` instance for USD output.
        cuda_graph: An optional `wp.Graph` object holding the captured simulation segment.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        profiling_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        """
        Initializes the simulator.

        This sets up all configurations, resolves timing parameters, builds the physics model
        by calling the abstract `build_model` method, and initializes the Axion engine, renderer,
        and profiling tools.

        Args:
            simulation_config: Configuration for the simulation timeline.
            rendering_config: Configuration for USD rendering.
            execution_config: Configuration for performance and execution strategy.
            profiling_config: Configuration for debugging and logging.
            engine_config: Configuration for the low-level Axion physics solver.
        """
        self.simulation_config = simulation_config
        self.rendering_config = rendering_config
        self.execution_config = execution_config
        self.profiling_config = profiling_config
        self.engine_config = engine_config

        self.logger = (
            HDF5Logger(self.profiling_config.hdf5_log_file)
            if self.profiling_config.enable_hdf5_logging
            else NullLogger()
        )

        self._current_step = 0
        self._current_time = 0.0

        # Calculated by _resolve_timing_parameters
        self.steps_per_segment: int = 0
        self.num_segments: int = 0
        self.effective_timestep: float = 0.0
        self.effective_duration: float = 0.0
        self._resolve_timing_parameters()

        self.model = self.build_model()
        self.integrator = AxionEngine(self.model, self.engine_config, self.logger)

        self.current_state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()

        self.renderer: Optional[wp.sim.render.SimRenderer] = None
        if self.rendering_config.enable:
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                self.rendering_config.usd_file,
                scaling=self.rendering_config.scaling,
                fps=self.rendering_config.fps,
            )

        self.cuda_graph: Optional[wp.Graph] = None

        # --- Initialize profiling events ---
        num_substeps = self.steps_per_segment
        self.events = [
            {
                "step_start": wp.Event(enable_timing=True),
                "collision": wp.Event(enable_timing=True),
                "integration": wp.Event(enable_timing=True),
                # Populated by the engine if profiling is active
                "integration_parts": None,
            }
            for _ in range(num_substeps)
        ]

    def run(self):
        """
        Main entry point to start the simulation.

        This method executes the main simulation loop. It iterates through each segment,
        runs the physics simulation for that segment, and, if enabled, renders the state
        at the end of the segment. After the loop completes, it finalizes and saves the
        rendered output.
        """
        disable_progress = self.profiling_config.enable_timing
        with self.logger:
            for i in tqdm(
                range(self.num_segments),
                desc="Simulating",
                disable=disable_progress,
            ):
                self._run_simulation_segment(i)

                if self.rendering_config.enable:
                    wp.synchronize()
                    # The time for the renderer is the time at the end of the frame
                    time = (i + 1) * self.steps_per_segment * self.effective_timestep
                    self.renderer.begin_frame(time)
                    self.renderer.render(self.current_state)
                    self.renderer.end_frame()

        if self.rendering_config.enable and self.renderer:
            self.renderer.save()
            print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")
        else:
            wp.synchronize()
            print("Headless simulation complete.")

    def _run_simulation_segment(self, segment_num: int):
        """
        Executes a single simulation segment using the chosen execution path.

        This method acts as a dispatcher, calling the high-performance CUDA graph-based
        method or the more flexible iterative method based on the configuration.

        Args:
            segment_num: The index of the current segment being executed.
        """
        if self.use_cuda_graph:
            self._run_segment_with_graph(segment_num)
        else:
            self._run_segment_without_graph(segment_num)

    def _run_segment_without_graph(self, segment_num: int):
        """
        Runs a segment by iterating and launching each step's kernels individually.

        This execution mode is more flexible than the graphed version and is used when detailed
        profiling or HDF5 logging is enabled, as it allows for CPU-side operations and
        synchronization between steps.

        Args:
            segment_num: The index of the current segment being executed.
        """
        n_steps = self.steps_per_segment
        timer_msg = f"SEGMENT {segment_num}/{self.num_segments}: Simulation of {n_steps} time steps"
        with wp.ScopedTimer(
            timer_msg,
            active=self.profiling_config.enable_timing,
            synchronize=True,
        ):
            for step in range(n_steps):
                self._single_physics_step(step)

                # Update attributes for logging. This happens outside the kernel launch.
                self._current_step += 1
                self._current_time += self.effective_timestep

        if self.profiling_config.enable_timing:
            self._log_segment_timings()

    def _run_segment_with_graph(self, segment_num: int):
        """
        Runs a segment by launching a pre-captured CUDA graph.

        This is the high-performance execution path. If a CUDA graph has not yet been
        captured, it first calls `_capture_cuda_graph`. It then launches the entire
        segment's worth of computation in a single GPU call, minimizing CPU-GPU
        communication overhead.

        Args:
            segment_num: The index of the current segment being executed.
        """
        if self.cuda_graph is None:
            self._capture_cuda_graph()
            if self.cuda_graph is None:
                # Fallback if capture failed for some reason
                self._run_segment_without_graph(segment_num)
                return

        n_steps = self.steps_per_segment
        timer_msg = f"SEGMENT {segment_num}/{self.num_segments}: Simulation of {n_steps} time steps (with CUDA graph)"
        with wp.ScopedTimer(
            timer_msg,
            active=self.profiling_config.enable_timing,
            synchronize=True,
        ):
            wp.capture_launch(self.cuda_graph)

        if self.profiling_config.enable_timing:
            self._log_segment_timings()

    def _log_segment_timings(self):
        """Logs the detailed timing information for the most recent segment."""
        for step in range(self.steps_per_segment):
            collision_time = wp.get_event_elapsed_time(
                self.events[step]["step_start"],
                self.events[step]["collision"],
            )
            integration_time = wp.get_event_elapsed_time(
                self.events[step]["collision"],
                self.events[step]["integration"],
            )

            print(
                f"\t- SUBSTEP {step}: collision detection took {collision_time:.03f} ms "
                f"and simulation step took {integration_time:0.3f} ms."
            )

            # Check if detailed integrator events were captured
            if self.events[step]["integration_parts"] is None:
                continue

            for newton_iter in range(self.engine_config.newton_iters):
                events = self.events[step]["integration_parts"][newton_iter]
                linearize_time = wp.get_event_elapsed_time(
                    events["iter_start"], events["linearize"]
                )
                lin_solve_time = wp.get_event_elapsed_time(events["linearize"], events["lin_solve"])
                linesearch_time = wp.get_event_elapsed_time(
                    events["lin_solve"], events["linesearch"]
                )

                print(
                    f"\t\t- NEWTON ITERATION {newton_iter}: Linearization took {linearize_time:.03f} ms, "
                    f"solving of linear system took {lin_solve_time:.03f} ms and linesearch took {linesearch_time:.03f} ms."
                )

    def _capture_cuda_graph(self):
        """Records the sequence of operations for one segment into a CUDA graph."""
        n_steps = self.steps_per_segment
        try:
            with wp.ScopedCapture() as capture:
                for i in range(n_steps):
                    self._single_physics_step(i)
            self.cuda_graph = capture.graph
        except Exception as e:
            print(
                f"Warning: Failed to capture CUDA graph. Falling back to iterative mode. Error: {e}"
            )
            self.execution_config.use_cuda_graph = False  # Disable for future segments

    def _single_physics_step(self, step_num: int):
        """
        Performs one fundamental integration step of the simulation.

        This method encapsulates the work for a single timestep `dt`. It follows a sequence of:
        1.  (Optional) Logging the pre-step state.
        2.  Recording a start event for profiling.
        3.  Running broad-phase and narrow-phase collision detection.
        4.  Recording a collision event for profiling.
        5.  Invoking the `AxionEngine` to solve the unified constraint dynamics. The engine uses a
            non-smooth Newton method to find impulses that satisfy all contact, friction, and
            joint constraints simultaneously, producing the next state.
        6.  Recording an integration event for profiling.
        7.  Copying the computed next state back to the current state for the next step.

        Args:
            step_num: The index of the step within the current segment. Used for profiling.
        """
        with self.logger.scope(f"timestep_{self._current_step + step_num:06d}"):
            current_time = self._current_time + (step_num * self.effective_timestep)
            self.logger.log_scalar("time", current_time)

            # Record that step started
            wp.record_event(self.events[step_num]["step_start"])

            # Detect collisions
            wp.sim.collide(self.model, self.current_state)

            # Record that collision detection finished
            wp.record_event(self.events[step_num]["collision"])
            self.logger.log_wp_dataset("rigid_contact_count", self.model.rigid_contact_count)

            # Compute simulation step
            self.events[step_num]["integration_parts"] = self.integrator.simulate(
                model=self.model,
                state_in=self.current_state,
                state_out=self.next_state,
                dt=self.effective_timestep,
                control=self.control,
            )

            # Record that simulation step finished
            wp.record_event(self.events[step_num]["integration"])

            # Update state for the next step
            wp.copy(dest=self.current_state.body_q, src=self.next_state.body_q)
            wp.copy(dest=self.current_state.body_qd, src=self.next_state.body_qd)

    def _resolve_timing_parameters(self):
        """
        Calculates all operational timing parameters based on user configuration.

        This internal method reconciles the desired simulation parameters with the constraints
        imposed by rendering and execution segmentation. It calculates and sets the
        `effective_timestep`, `steps_per_segment`, `effective_duration`, and `num_segments`
        attributes.
        """
        if self.rendering_config.enable:
            self.effective_timestep, self.steps_per_segment = _calculate_render_aligned_timestep(
                self.simulation_config.target_timestep_seconds, self.rendering_config.fps
            )
        else:
            self.effective_timestep = self.simulation_config.target_timestep_seconds
            self.steps_per_segment = self.execution_config.headless_steps_per_segment

        self.effective_duration, self.num_segments = _align_duration_to_segment(
            self.simulation_config.duration_seconds, self.effective_timestep, self.steps_per_segment
        )

    @property
    def use_cuda_graph(self) -> bool:
        """
        Determines if conditions are met to use CUDA graph optimization.

        CUDA graphs provide a significant performance boost by minimizing CPU-GPU interaction,
        but they have certain requirements.

        Returns:
            `True` if CUDA graphs should be used, which requires that:
            - `execution_config.use_cuda_graph` is `True`.
            - The current Warp device is a CUDA GPU.
            - `profiling_config.enable_hdf5_logging` is `False` (logging is incompatible).
        """
        return (
            self.execution_config.use_cuda_graph
            and wp.get_device().is_cuda
            and not self.profiling_config.enable_hdf5_logging
        )

    @abstractmethod
    def build_model(self) -> wp.sim.Model:
        """
        Builds the physics model for the simulation.

        This abstract method MUST be implemented by any subclass. It is responsible for
        programmatically defining all the physical elements of the scene, including:
        - Creating a `wp.sim.ModelBuilder`.
        - Adding rigid bodies with their shapes, mass, and initial poses.
        - Defining joints to connect bodies.
        - Setting other physical properties.

        Returns:
            A fully constructed `wp.sim.Model` instance representing the simulation scene.
        """
        pass
