from abc import ABC
from enum import auto
from enum import Enum
from typing import Optional

import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import EngineConfig
from axion.core.logging_config import LoggingConfig
from tqdm import tqdm

from .base_simulator import BaseSimulator
from .base_simulator import ExecutionConfig
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@wp.kernel
def randomize_velocities_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),
    lin_bound_min: float,
    lin_bound_max: float,
    ang_bound_min: float,
    ang_bound_max: float,
    seed: int,
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)

    # Generate random angular velocity (wx, wy, wz)
    wx = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min
    wy = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min
    wz = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min

    # Generate random linear velocity (vx, vy, vz)
    vx = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min
    vy = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min
    vz = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min

    rand_vel = wp.spatial_vector(wx, wy, wz, vx, vy, vz)

    # Forcefully overwrite both state arrays
    body_qd[tid] = rand_vel


@wp.kernel
def sync_prev_poses_kernel(
    pose: wp.array(dtype=wp.transform), pose_prev: wp.array(dtype=wp.transform)
):
    tid = wp.tid()
    pose_prev[tid] = pose[tid]


class StartupState(Enum):
    PRE_RESOLVE = auto()
    POST_RESOLVE = auto()
    RUNNING = auto()


class DatasetSimulator(BaseSimulator, ABC):
    """
    Simulator designed for real-time visualization and interactive sessions.
    Supports GL/USD rendering, FPS synchronization, and CUDA graphs.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            execution_config,
            engine_config,
            logging_config,
        )

        self.viewer = self.rendering_config.create_viewer(
            model=self.model,
            num_segments=self.num_segments,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((20.0, 20.0, 0.0))

        # CUDA Graph Storage
        self.cuda_graph: Optional[wp.Graph] = None

        self._startup_state = StartupState.PRE_RESOLVE
        if self.rendering_config.start_paused and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        else:
            self._startup_state = StartupState.RUNNING

    def _resolve_constraints(self):
        print("Resolving constraints...")

        for i in range(10):
            self.contacts = self.model.collide(self.current_state)
            self.solver.step(
                state_in=self.current_state,
                state_out=self.next_state,
                control=self.control,
                contacts=self.contacts,
                dt=self.clock.dt,
            )

            # Copy only positions
            wp.copy(self.current_state.body_q, self.next_state.body_q)
            self.current_state.body_qd.zero_()
            # self.next_state.body_qd.zero_()
            # wp.copy(self.next_state.body_qd, self.current_state.body_qd)

        wp.launch(
            kernel=randomize_velocities_kernel,
            dim=self.current_state.body_qd.shape[0],
            inputs=[
                self.current_state.body_qd,
                self.lin_min,
                self.lin_max,
                self.ang_min,
                self.ang_max,
                self.seed,
            ],
            device=self.model.device,
        )

    def run(self):
        """Main entry point to start the simulation."""

        pbar = tqdm(
            total=self.num_segments,
            desc="Simulating",
        )

        try:
            segment_num = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    if self._startup_state == StartupState.PRE_RESOLVE:
                        self._resolve_constraints()
                        self._startup_state = StartupState.POST_RESOLVE
                        self.viewer._paused = True  # Pause again to show post-resolve state.
                    elif self._startup_state == StartupState.POST_RESOLVE:
                        self._startup_state = StartupState.RUNNING
                        # Fall through to the simulation run.
                        self._run_simulation_segment(segment_num)
                        segment_num += 1
                        pbar.update(1)
                    elif self._startup_state == StartupState.RUNNING:
                        self._run_simulation_segment(segment_num)
                        segment_num += 1
                        pbar.update(1)

                self._render(segment_num)

                if self.rendering_config.vis_type == "gl":
                    wp.synchronize()
        finally:
            pbar.close()

            if isinstance(self.solver, AxionEngine):
                # self.solver.events.print_timings()
                self.solver.save_logs()

            if self.rendering_config.vis_type == "usd":
                self.viewer.close()
                print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")

    def _render(self, segment_num: int):
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        self.viewer.end_frame()

    def _run_simulation_segment(self, segment_num: int):
        if self.use_cuda_graph:
            self._run_segment_with_graph(segment_num)
        else:
            self._run_segment_without_graph(segment_num)

    def _run_segment_without_graph(self, segment_num: int):
        n_steps = self.steps_per_segment
        for step in range(n_steps):
            self._single_physics_step(step)

    def _run_segment_with_graph(self, segment_num: int):
        if self.cuda_graph is None:
            self._capture_cuda_graphs()

        wp.capture_launch(self.cuda_graph)

    def _capture_cuda_graphs(self):
        n_steps = self.steps_per_segment
        print(f"INFO: Capturing CUDA Graph (steps={n_steps})...")
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph = capture.graph
