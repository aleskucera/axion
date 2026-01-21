import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
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
from .model_builder import AxionModelBuilder


class SyncMode(Enum):
    """Defines how the simulator synchronizes physics time with render time."""

    ALIGN_DT_TO_FPS = 0
    """
    Priority: Rendering FPS.
    The physics timestep is adjusted (shortened) so that an integer number of 
    steps fit exactly into one render frame.
    """

    ALIGN_FPS_TO_DT = 1
    """
    Priority: Physics Timestep.
    The rendering FPS is adjusted so that one frame equals exactly one physics 
    timestep (or the configured number of steps per segment).
    """


@dataclass
class SimulationConfig:
    """Parameters defining the simulation's timeline."""

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3
    num_worlds: int = 1
    sync_mode: SyncMode = SyncMode.ALIGN_FPS_TO_DT


@dataclass
class RenderingConfig:
    """Parameters for rendering the simulation to a USD file."""

    vis_type: Literal["gl", "usd", "null", None] = "gl"
    target_fps: int | None = 30
    usd_file: str | None = "sim.usd"
    usd_scaling: float | None = 100.0
    start_paused: bool = True


@dataclass
class ExecutionConfig:
    """Parameters controlling the performance and execution strategy."""

    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10


# --- Timing Calculation Helpers ---


def calculate_render_aligned_timestep(
    target_timestep_seconds: float, fps: int, force_even: bool = True
):
    """Calculates an effective timestep that aligns perfectly with render frame duration."""
    frame_duration = 1.0 / fps
    ideal_steps_per_frame = frame_duration / target_timestep_seconds
    steps_per_frame = round(ideal_steps_per_frame) or 1

    if force_even and steps_per_frame % 2 != 0:
        steps_per_frame += 1

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
    ):
        self.simulation_config = simulation_config
        self.rendering_config = rendering_config
        self.execution_config = execution_config
        self.engine_config = engine_config

        self._current_step = 0
        self._current_time = 0.0

        # Calculated by _resolve_timing_parameters
        self.steps_per_segment: int = 0
        self.num_segments: int = 0
        self.effective_timestep: float = 0.0
        self.effective_duration: float = 0.0
        self._resolve_timing_parameters()

        self.builder = AxionModelBuilder()
        self.model = self.build_model()

        self.current_state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.current_state)

        # Prepare kwargs for third-party solvers (non-Axion)
        solver_kwargs = vars(self.engine_config).copy()
        axion_logging_keys = [
            "enable_timing",
            "enable_hdf5_logging",
            "hdf5_log_file",
            "log_dynamics_state",
            "log_linear_system_data",
            "log_constraint_data",
        ]
        for k in axion_logging_keys:
            solver_kwargs.pop(k, None)

        if isinstance(self.engine_config, AxionEngineConfig):
            self.solver = AxionEngine(self.model, self.init_state_fn, self.engine_config)
        elif isinstance(self.engine_config, FeatherstoneEngineConfig):
            self.solver = SolverFeatherstone(self.model, **solver_kwargs)
        elif isinstance(self.engine_config, MuJoCoEngineConfig):
            self.solver = SolverMuJoCo(self.model, **solver_kwargs)
        elif isinstance(self.engine_config, XPBDEngineConfig):
            self.solver = SolverXPBD(self.model, **solver_kwargs)
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

        # CUDA Graph Storage (Double Buffered for Odd steps)
        # Graph A -> B
        self.cuda_graph_primary: Optional[wp.Graph] = None
        # Graph B -> A (Only used if steps_per_segment is ODD)
        self.cuda_graph_secondary: Optional[wp.Graph] = None

    def run(self):
        """Main entry point to start the simulation."""
        pbar = tqdm(
            total=self.num_segments,
            desc="Simulating",
        )

        # Set initial paused state if requested (only for GL viewer)
        if self.rendering_config.start_paused and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True

        try:
            segment_num = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self._run_simulation_segment(segment_num)
                    segment_num += 1
                    pbar.update(1)
                self._render(segment_num)

                if self.rendering_config.vis_type == "gl":
                    wp.synchronize()
        finally:
            pbar.close()

            if isinstance(self.solver, AxionEngine):
                self.solver.events.print_timings()

            if self.rendering_config.vis_type == "usd":
                self.viewer.close()
                print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")

    def run_visualization(self):
        """Runs the visualization loop without advancing the physics simulation."""
        if not isinstance(self.viewer, newton.viewer.ViewerGL):
            print(
                "Error: run_visualization() only supports ViewerGL. Please set rendering.vis_type='gl'."
            )
            return

        print("Starting visualization mode (Physics paused)...")

        # Ensure we have initial contacts for visualization
        self.contacts = self.model.collide(self.current_state)

        while self.viewer.is_running():
            self.viewer.begin_frame(0.0)
            self.viewer.log_state(self.current_state)
            self.viewer.log_contacts(self.contacts, self.current_state)
            self.viewer.end_frame()

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

        if isinstance(self.solver, AxionEngine):
            self.solver.events.record_timings()

    def _run_segment_with_graph(self, segment_num: int):
        """Runs a segment by launching a pre-captured CUDA graph."""

        # If we haven't captured graphs yet, do so now.
        if self.cuda_graph_primary is None:
            self._capture_cuda_graphs()

        # If steps are even: Pointers always start at A and end at A.
        # We always use the primary graph.
        if self.steps_per_segment % 2 == 0:
            wp.capture_launch(self.cuda_graph_primary)

        # If steps are odd: Pointers toggle A->B, then B->A.
        # Segment 0, 2, 4... (Even Segments): Start at A, use Primary (A->B).
        # Segment 1, 3, 5... (Odd Segments): Start at B, use Secondary (B->A).
        else:
            if segment_num % 2 == 0:
                wp.capture_launch(self.cuda_graph_primary)
            else:
                wp.capture_launch(self.cuda_graph_secondary)

        if isinstance(self.solver, AxionEngine):
            self.solver.events.record_timings()

    def _capture_cuda_graphs(self):
        """
        Records the sequence of operations into CUDA graphs.
        If steps_per_segment is ODD, we record two graphs (ping-pong) to handle pointer swapping.
        """
        n_steps = self.steps_per_segment

        # Capture Graph 1 (Primary): Starts with current state pointers (A -> ...)
        print(f"INFO: Capturing CUDA Graph (Primary, steps={n_steps})...")
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph_primary = capture.graph

        # If steps are even, we are done. The python pointer swap in _single_physics_step
        # happened an even number of times, so self.current_state is back to "A".

        # If steps are odd, self.current_state is now "B". We must capture the return trip.
        if n_steps % 2 != 0:
            print(f"INFO: Capturing CUDA Graph (Secondary/Return, steps={n_steps})...")
            with wp.ScopedCapture() as capture:
                for i in range(n_steps):
                    self._single_physics_step(i)
            self.cuda_graph_secondary = capture.graph

            # Now self.current_state is back to "A".

    def _single_physics_step(self, step_num: int):
        """Performs one fundamental integration step of the simulation."""
        self.current_state.clear_forces()

        # Detect collisions
        self.contacts = self.model.collide(self.current_state)

        self.control_policy(self.current_state)
        self.viewer.apply_forces(self.current_state)

        # Compute simulation step
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.effective_timestep,
        )

        self.current_state, self.next_state = self.next_state, self.current_state

    def _resolve_timing_parameters(self):
        """
        Calculates timing parameters based on configuration and SyncMode.
        """
        mode = self.simulation_config.sync_mode
        target_dt = self.simulation_config.target_timestep_seconds

        if self.rendering_config.vis_type in ["usd", "gl"]:
            target_fps = self.rendering_config.target_fps

            # --- STRATEGY 0: MODIFY DT (Fit Physics into Frame) ---
            if mode == SyncMode.ALIGN_DT_TO_FPS:
                # We adjust the timestep so it fits perfectly into the target FPS.
                # We allow odd steps because our CUDA graph logic now handles it.
                self.effective_timestep, self.steps_per_segment = calculate_render_aligned_timestep(
                    target_dt, target_fps, force_even=False
                )

            # --- STRATEGY 1: MODIFY FPS (Fit Frame around Physics) ---
            elif mode == SyncMode.ALIGN_FPS_TO_DT:
                # We keep the requested timestep EXACT.
                self.effective_timestep = target_dt

                # Calculate how many of these exact steps fit into the requested frame duration
                target_frame_duration = 1.0 / target_fps
                ideal_steps = round(target_frame_duration / target_dt)

                # Ensure at least 1 step (if physics step > frame duration)
                self.steps_per_segment = max(1, ideal_steps)

                # The resulting frame duration is purely a multiple of the physics step.
                # This might result in an FPS that is slightly different from target_fps,
                # but it guarantees the physics dt is preserved.
                actual_frame_duration = self.steps_per_segment * self.effective_timestep
                new_fps = 1.0 / actual_frame_duration

                # Only log if there is a significant change in FPS
                if abs(new_fps - target_fps) > 0.1:
                    print(
                        f"\nINFO: Rendering FPS adjusted from {target_fps} to {new_fps:.2f} "
                        f"to maintain fixed timestep {target_dt*1000:.1f}ms "
                        f"({self.steps_per_segment} steps/frame)."
                    )
                self.rendering_config.target_fps = new_fps

        else:
            # Headless / Null mode
            self.effective_timestep = target_dt
            self.steps_per_segment = self.execution_config.headless_steps_per_segment

        self.effective_duration, self.num_segments = align_duration_to_segment(
            self.simulation_config.duration_seconds, self.effective_timestep, self.steps_per_segment
        )
        if self.rendering_config.vis_type == "gl":
            self.num_segments = None

    @property
    def use_cuda_graph(self) -> bool:
        """Determines if conditions are met to use CUDA graph optimization."""
        # Disable graph if HDF5 logging is enabled (requires CPU execution)
        if (
            isinstance(self.engine_config, AxionEngineConfig)
            and self.engine_config.enable_hdf5_logging
        ):
            return False

        return self.execution_config.use_cuda_graph and wp.get_device().is_cuda

    def _get_newton_iters(self) -> int:
        """Get the number of Newton iterations, or 0 if not using AxionEngine."""
        return self.engine_config.max_newton_iters if isinstance(self.solver, AxionEngine) else 0

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
