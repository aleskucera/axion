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
    """Parameters defining the simulation's timeline."""

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3


@dataclass
class RenderingConfig:
    """Parameters for rendering the simulation to a USD file."""

    enable: bool = True
    fps: int = 30
    scaling: float = 100.0
    usd_file: str = "sim.usd"


@dataclass
class ExecutionConfig:
    """Parameters controlling the performance and execution strategy."""

    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10


@dataclass
class ProfilingConfig:
    """Parameters for debugging, timing, and logging."""

    enable_timing: bool = False

    # Enables HDF5 logging (disables CUDA graph optimization).
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"


# --- Timing Calculation Helpers ---


def calculate_render_aligned_timestep(target_timestep_seconds: float, fps: int):
    """Calculates an effective timestep that aligns perfectly with render frame duration."""
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


def align_duration_to_segment(target_duration: float, timestep: float, steps_per_segment: int):
    """Adjusts total simulation duration to be a whole multiple of the segment duration."""
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
    """An abstract base class for running a Warp-based physics simulation."""

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        profiling_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
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

        num_substeps = self.steps_per_segment
        self.start_events = [wp.Event(enable_timing=True) for _ in range(num_substeps)]
        self.collision_events = [wp.Event(enable_timing=True) for _ in range(num_substeps)]
        self.integration_events = [wp.Event(enable_timing=True) for _ in range(num_substeps)]

        self.events = [
            {
                "step_start": wp.Event(enable_timing=True),
                "collision": wp.Event(enable_timing=True),
                "integration": wp.Event(enable_timing=True),
                "integration_parts": None,
            }
            for _ in range(num_substeps)
        ]

        self.simulation_events = [[] for _ in range(num_substeps)]

    def run(self):
        """Main entry point to start the simulation."""
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
                    time = (i + 1) * (1.0 / self.rendering_config.fps)
                    self.renderer.begin_frame(time)
                    self.renderer.render(self.current_state)
                    self.renderer.end_frame()

        if self.rendering_config.enable:
            self.renderer.save()
            print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")
        else:
            wp.synchronize()
            print("Headless simulation complete.")

    def _run_simulation_segment(self, segment_num: int):
        """Executes a single simulation segment, using the chosen execution path."""
        if self.use_cuda_graph:
            self._run_segment_with_graph(segment_num)
        else:
            self._run_segment_without_graph(segment_num)

    def _run_segment_without_graph(self, segment_num: int):
        """Runs a segment by iterating and launching each step's kernels individually."""
        n_steps = self.steps_per_segment
        timer_msg = f"SEGMENT {segment_num}/{self.num_segments}: Simulation of {n_steps} time steps"
        with wp.ScopedTimer(
            timer_msg,
            active=self.profiling_config.enable_timing,
            synchronize=True,
        ):
            for step in range(n_steps):
                self._single_physics_step(step)

                # Update attributes for logging
                self._current_step += 1
                self._current_time += self.effective_timestep

        if self.profiling_config.enable_timing:
            self._log_segment_timings()

    def _run_segment_with_graph(self, segment_num: int):
        """Runs a segment by launching a pre-captured CUDA graph."""
        if self.cuda_graph is None:
            self._capture_cuda_graph()

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
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph = capture.graph

    def _single_physics_step(self, step_num: int):
        """Performs one fundamental integration step of the simulation."""
        with self.logger.scope(f"timestep_{self._current_step:04d}"):
            self.logger.log_scalar("time", self._current_time)

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

            wp.copy(dest=self.current_state.body_q, src=self.next_state.body_q)
            wp.copy(dest=self.current_state.body_qd, src=self.next_state.body_qd)

    def _resolve_timing_parameters(self):
        """
        Calculates all operational timing parameters based on user configuration,
        ensuring alignment between simulation, rendering, and segmentation.
        """
        if self.rendering_config.enable:
            self.effective_timestep, self.steps_per_segment = calculate_render_aligned_timestep(
                self.simulation_config.target_timestep_seconds, self.rendering_config.fps
            )
        else:
            self.effective_timestep = self.simulation_config.target_timestep_seconds
            self.steps_per_segment = self.execution_config.headless_steps_per_segment

        self.effective_duration, self.num_segments = align_duration_to_segment(
            self.simulation_config.duration_seconds, self.effective_timestep, self.steps_per_segment
        )

    @property
    def use_cuda_graph(self) -> bool:
        """Determines if conditions are met to use CUDA graph optimization."""
        return (
            self.execution_config.use_cuda_graph
            and wp.get_device().is_cuda
            and not self.profiling_config.enable_hdf5_logging
        )

    @abstractmethod
    def build_model(self) -> wp.sim.Model:
        """
        Builds the physics model for the simulation.

        This method MUST be implemented by any subclass. It should define all the
        rigid bodies, joints, and other physical properties of the scene.
        """
        pass
