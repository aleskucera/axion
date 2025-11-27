import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal
from typing import Optional

import newton
import warp as wp
from newton import Model
from newton.solvers import SolverFeatherstone
from newton.solvers import SolverMuJoCo
from newton.solvers import SolverXPBD
from tqdm import tqdm

from .control_utils import JointMode
from .engine import AxionEngine
from .engine_config import AxionEngineConfig
from .engine_config import EngineConfig
from .engine_config import FeatherstoneEngineConfig
from .engine_config import MuJoCoEngineConfig
from .engine_config import XPBDEngineConfig
from .engine_logger import EngineLogger
from .engine_logger import LoggingConfig


@dataclass
class SimulationConfig:
    """Parameters defining the simulation's timeline."""

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3
    num_worlds: int = 1


@dataclass
class RenderingConfig:
    """Parameters for rendering the simulation to a USD file."""

    vis_type: Literal["gl", "usd", "null", None] = "gl"
    target_fps: int | None = 30
    usd_file: str | None = "sim.usd"
    usd_scaling: float | None = 100.0


@dataclass
class ExecutionConfig:
    """Parameters controlling the performance and execution strategy."""

    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10


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
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        self.simulation_config = simulation_config
        self.rendering_config = rendering_config
        self.execution_config = execution_config
        self.engine_config = engine_config
        self.logging_config = logging_config

        self.logger = EngineLogger(self.logging_config)

        self._current_step = 0
        self._current_time = 0.0

        # Calculated by _resolve_timing_parameters
        self.steps_per_segment: int = 0
        self.num_segments: int = 0
        self.effective_timestep: float = 0.0
        self.effective_duration: float = 0.0
        self._resolve_timing_parameters()

        self.builder = self._create_builder_with_custom_attributes()
        self.model = self.build_model()

        self.current_state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.current_state)

        if isinstance(self.engine_config, AxionEngineConfig):
            self.solver = AxionEngine(
                self.model, self.init_state_fn, self.logger, self.engine_config
            )
        elif isinstance(self.engine_config, FeatherstoneEngineConfig):
            self.solver = SolverFeatherstone(self.model, **vars(self.engine_config))
        elif isinstance(self.engine_config, MuJoCoEngineConfig):
            self.solver = SolverMuJoCo(self.model, **vars(self.engine_config))
        elif isinstance(self.engine_config, XPBDEngineConfig):
            self.solver = SolverXPBD(self.model, **vars(self.engine_config))
        else:
            raise ValueError(f"Unsupported engine configuration type: {type(self.engine_config)}")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.current_state)

        if self.rendering_config.vis_type == "usd":
            self.viewer = newton.viewer.ViewerUSD(
                output_path=self.rendering_config.usd_file,
                fps=self.rendering_config.target_fps,
                up_axis="Z",
                num_frames=self.num_segments,
            )
        elif self.rendering_config.vis_type == "gl":
            self.viewer = newton.viewer.ViewerGL()
        elif self.rendering_config.vis_type == "null" or self.rendering_config.vis_type is None:
            self.viewer = newton.viewer.ViewerNull(self.num_segments)
        else:
            raise ValueError(f"Unsupported rendering type: {self.rendering_config.vis_type}")

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((20.0, 20.0, 0.0))

        self.cuda_graph: Optional[wp.Graph] = None

        self.logger.initialize_events(self.steps_per_segment, self._get_newton_iters())

    def run(self):
        """Main entry point to start the simulation."""
        self.logger.open()
        pbar = tqdm(
            total=self.num_segments,
            desc="Simulating",
        )

        try:
            segment_num = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self._run_simulation_segment(segment_num)
                    segment_num += 1
                    pbar.update(1)
                self._render(segment_num)
        finally:
            pbar.close()
            self.logger.close()

            if self.rendering_config.vis_type == "usd":
                self.viewer.close()
                print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")

    def _render(self, segment_num: int):
        """Renders the current state to the appropriate viewers."""
        sim_time = segment_num * self.steps_per_segment * self.effective_timestep
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        self.viewer.end_frame()

    def _run_simulation_segment(self, segment_num: int):
        """Executes a single simulation segment, using the chosen execution path."""
        if self.use_cuda_graph:
            self._run_segment_with_graph(segment_num)
        else:
            self._run_segment_without_graph(segment_num)

    def _run_segment_without_graph(self, segment_num: int):
        """Runs a segment by iterating and launching each step's kernels individually."""
        n_steps = self.steps_per_segment
        for step in range(n_steps):
            self._single_physics_step(step)

            # Update attributes for logging
            self._current_step += 1
            self._current_time += self.effective_timestep

        self.logger.log_segment_timings(self.steps_per_segment, self._get_newton_iters())

    def _run_segment_with_graph(self, segment_num: int):
        """Runs a segment by launching a pre-captured CUDA graph."""
        if self.cuda_graph is None:
            self._capture_cuda_graph()

        wp.capture_launch(self.cuda_graph)

        self.logger.log_segment_timings(self.steps_per_segment, self._get_newton_iters())

    def _log_segment_timings(self):
        """Logs the detailed timing information for the most recent segment."""
        for step in range(self.steps_per_segment):
            collision_time = wp.get_event_elapsed_time(
                self.events[step]["step_start"],
                self.events[step]["collision_detection"],
            )
            integration_time = wp.get_event_elapsed_time(
                self.events[step]["collision_detection"],
                self.events[step]["step"],
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
                    events["iter_start"], events["system_linearization"]
                )
                lin_solve_time = wp.get_event_elapsed_time(
                    events["system_linearization"], events["linear_system_solve"]
                )
                linesearch_time = wp.get_event_elapsed_time(
                    events["linear_system_solve"], events["linesearch"]
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
        self.current_state.clear_forces()

        self.logger.set_current_step_in_segment(step_num)
        self.logger.timestep_start(self._current_step, self._current_time)
        sim_events = self.logger.simulator_event_pairs[step_num]

        # Detect collisions
        with self.logger.timed_block(*sim_events["collision_detection"]):
            self.contacts = self.model.collide(self.current_state)

        # Record that collision detection finished
        self.logger.log_contact_count(self.contacts)

        self.control_policy(self.current_state)
        self.viewer.apply_forces(self.current_state)

        # Compute simulation step
        with self.logger.timed_block(*sim_events["step"]):
            self.solver.step(
                state_in=self.current_state,
                state_out=self.next_state,
                control=self.control,
                contacts=self.contacts,
                dt=self.effective_timestep,
            )

        self.current_state, self.next_state = self.next_state, self.current_state

        self.logger.timestep_end()

    def _resolve_timing_parameters(self):
        """
        Calculates all operational timing parameters based on user configuration,
        ensuring alignment between simulation, rendering, and segmentation.
        """
        if self.rendering_config.vis_type == "usd":
            self.effective_timestep, self.steps_per_segment = calculate_render_aligned_timestep(
                self.simulation_config.target_timestep_seconds, self.rendering_config.target_fps
            )
        else:
            self.effective_timestep = self.simulation_config.target_timestep_seconds
            self.steps_per_segment = self.execution_config.headless_steps_per_segment

        self.effective_duration, self.num_segments = align_duration_to_segment(
            self.simulation_config.duration_seconds, self.effective_timestep, self.steps_per_segment
        )
        if self.rendering_config.vis_type == "gl":
            self.num_segments = None

    def _create_builder_with_custom_attributes(self) -> newton.ModelBuilder:
        """
        Adds the custom attributes to the ModelBuilder and adds the instance of the builder
        to self attributes of AbstractSimulator class.
        """
        builder = newton.ModelBuilder()

        # --- Add custom attributes to the model class ---

        # integral constant (PID control)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_target_ki",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,  # Explicit default value
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        # previous instance of the control error (PID control)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_err_prev",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

        # cummulative error of the integral part (PID control)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_err_i",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_dof_mode",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.int32,
                default=JointMode.NONE,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_target",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

        return builder

    @property
    def use_cuda_graph(self) -> bool:
        """Determines if conditions are met to use CUDA graph optimization."""
        return (
            self.execution_config.use_cuda_graph
            and wp.get_device().is_cuda
            and self.logger.can_cuda_graph_be_used
        )

    def _get_newton_iters(self) -> int:
        """Get the number of Newton iterations, or 0 if not using AxionEngine."""
        return self.engine_config.newton_iters if isinstance(self.solver, AxionEngine) else 0

    @abstractmethod
    def build_model(self) -> Model:
        """
        Builds the physics model for the simulation.

        This method MUST be implemented by any subclass. It should define all the
        rigid bodies, joints, and other physical properties of the scene.

        It HAS TO use the self.builder instance of creating its own.
        """
        pass

    def control_policy(self, current_state: newton.State):
        """
        Implements the control policy for the simulation.

        This method may be optionally overridden by any subclass.
        It is called at each simulation step. It can be used to update control inputs
        (self.control.joint_targets and self.control.joint_f).
        By default, it does nothing.
        """
        pass

    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        """
        This function initializes the simulation state before the first step for the Axion engine.

        It is not necessary to use every available argument. Control is applied before this call.
        This method may be optionally overridden by any subclass.
        For one time initializations use the subclass constructor, which can use all the AbstractSimulator self attributes.
        It is called before each simulation step.
        """

        self.solver.integrate_bodies(self.model, current_state, next_state, dt)
