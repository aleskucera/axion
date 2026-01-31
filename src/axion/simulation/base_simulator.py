from __future__ import annotations

import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from typing import Literal

import newton
import warp as wp
from axion import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from axion.core.engine_config import FeatherstoneEngineConfig
from axion.core.engine_config import MuJoCoEngineConfig
from axion.core.engine_config import SemiImplicitEngineConfig
from axion.core.engine_config import XPBDEngineConfig
from axion.core.joint_types import JointMode
from axion.core.model_builder import AxionModelBuilder
from newton import Model
from newton.solvers import SolverFeatherstone
from newton.solvers import SolverMuJoCo
from newton.solvers import SolverSemiImplicit
from newton.solvers import SolverXPBD


class SyncMode(Enum):
    """Defines how the simulator synchronizes physics time with render time."""

    ALIGN_DT_TO_FPS = 0
    ALIGN_FPS_TO_DT = 1


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


class BaseSimulator(ABC):
    """
    The foundational class for Axion simulations.
    Handles configuration, model building, solver initialization, and timing resolution.
    """

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
        self.total_sim_steps: int = 0
        self.steps_per_segment: int = 0
        self.num_segments: int | None = 0
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
            "differentiable_simulation",  # Not supported by 3rd party
            "max_trajectory_steps",  # Not supported by 3rd party
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
        elif isinstance(self.engine_config, SemiImplicitEngineConfig):
            self.solver = SolverSemiImplicit(self.model, **solver_kwargs)
        else:
            raise ValueError(f"Unsupported engine configuration type: {type(self.engine_config)}")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.current_state)

        # Viewer initialization is deferred to subclasses (Interactive vs Headless/Diff)
        self.viewer = None

    @abstractmethod
    def build_model(self) -> Model:
        """
        Builds the physics model for the simulation.
        This method MUST be implemented by any subclass.
        """
        pass

    def control_policy(self, current_state: newton.State):
        """
        Implements the control policy for the simulation.
        This method may be optionally overridden by any subclass.
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
        Initialization hook for Axion engine.
        """
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    def _copy_state(self, dest: newton.State, src: newton.State):
        """Copies the physics state data from src to dest."""
        wp.copy(dest.body_q, src.body_q)
        wp.copy(dest.body_qd, src.body_qd)
        wp.copy(dest.joint_q, src.joint_q)
        wp.copy(dest.joint_qd, src.joint_qd)

    def _single_physics_step(self, step_num: int):
        """Performs one fundamental integration step of the simulation."""
        self.current_state.clear_forces()

        # Detect collisions
        self.contacts = self.model.collide(self.current_state)

        self.control_policy(self.current_state)

        if self.viewer:
            self.viewer.apply_forces(self.current_state)

        # Compute simulation step
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.effective_timestep,
        )

        # Explicitly copy next_state back to current_state
        # This ensures the data flow is recorded in Tape and CUDA Graphs
        self._copy_state(self.current_state, self.next_state)

    def _resolve_timing_parameters(self):
        """
        Calculates timing parameters based on configuration and SyncMode.
        """
        mode = self.simulation_config.sync_mode
        target_dt = self.simulation_config.target_timestep_seconds

        if self.rendering_config.vis_type in ["usd", "gl"]:
            target_fps = self.rendering_config.target_fps

            if mode == SyncMode.ALIGN_DT_TO_FPS:
                self.effective_timestep, self.steps_per_segment = calculate_render_aligned_timestep(
                    target_dt, target_fps, force_even=False
                )

            elif mode == SyncMode.ALIGN_FPS_TO_DT:
                self.effective_timestep = target_dt
                target_frame_duration = 1.0 / target_fps
                ideal_steps = round(target_frame_duration / target_dt)
                self.steps_per_segment = max(1, ideal_steps)
                actual_frame_duration = self.steps_per_segment * self.effective_timestep
                new_fps = 1.0 / actual_frame_duration

                if abs(new_fps - target_fps) > 0.1:
                    print(
                        f"\nINFO: Rendering FPS adjusted from {target_fps} to {new_fps:.2f} "
                        f"to maintain fixed timestep {target_dt*1000:.1f}ms "
                        f"({self.steps_per_segment} steps/frame)."
                    )
                self.rendering_config.target_fps = new_fps

        else:
            self.effective_timestep = target_dt
            self.steps_per_segment = self.execution_config.headless_steps_per_segment

        self.effective_duration, self.num_segments = align_duration_to_segment(
            self.simulation_config.duration_seconds, self.effective_timestep, self.steps_per_segment
        )

        self.total_sim_steps = self.steps_per_segment * self.num_segments
        if self.rendering_config.vis_type == "gl":
            self.num_segments = None
