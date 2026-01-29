from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional

import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from axion.core.engine_config import SemiImplicitEngineConfig
from newton import State

from .base_simulator import BaseSimulator
from .base_simulator import ExecutionConfig
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@dataclass
class TrajectoryData:
    body_q: wp.array
    body_u: wp.array
    step_count: wp.array


@wp.kernel
def save_to_trajectory_kernel_1d(
    body_q: wp.array(dtype=wp.transform, ndim=1),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=1),
    num_worlds: int,
    num_bodies: int,
    step_idx: int,
    body_q_trajectory: wp.array(dtype=wp.transform, ndim=3),
    body_u_trajectory: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    world_idx, body_idx = wp.tid()
    if world_idx >= num_worlds or body_idx >= num_bodies:
        return
    flat_idx = world_idx * num_bodies + body_idx
    body_q_trajectory[step_idx, world_idx, body_idx] = body_q[flat_idx]
    body_u_trajectory[step_idx, world_idx, body_idx] = body_u[flat_idx]


@wp.kernel
def save_to_trajectory_kernel_2d(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    num_worlds: int,
    num_bodies: int,
    step_idx: int,
    body_q_trajectory: wp.array(dtype=wp.transform, ndim=3),
    body_u_trajectory: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    world_idx, body_idx = wp.tid()
    if world_idx >= num_worlds or body_idx >= num_bodies:
        return
    body_q_trajectory[step_idx, world_idx, body_idx] = body_q[world_idx, body_idx]
    body_u_trajectory[step_idx, world_idx, body_idx] = body_u[world_idx, body_idx]


@wp.kernel
def flatten_trajectory_slice_kernel(
    traj_q: wp.array(dtype=wp.transform, ndim=3),
    step_idx: int,
    flat_q: wp.array(dtype=wp.transform, ndim=1),
    num_bodies: int,
    num_worlds: int,
):
    world_idx, body_idx = wp.tid()
    if world_idx >= num_worlds or body_idx >= num_bodies:
        return
    flat_idx = world_idx * num_bodies + body_idx
    flat_q[flat_idx] = traj_q[step_idx, world_idx, body_idx]


class DifferentiableSimulator(BaseSimulator, ABC):
    """
    A specialized simulator for running differentiable physics episodes.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: EngineConfig,
    ):
        # 1. Enforce Differentiable Configuration
        if isinstance(engine_config, AxionEngineConfig):
            updates = {}
            if not engine_config.differentiable_simulation:
                print("INFO: DifferentiableSimulator forcing differentiable_simulation=True")
                updates["differentiable_simulation"] = True
            if engine_config.max_trajectory_steps <= 0:
                updates["max_trajectory_steps"] = 1024
            if updates:
                engine_config = replace(engine_config, **updates)

        # 2. Enforce Null Viewer and No Graph
        if rendering_config.vis_type != "null":
            rendering_config.vis_type = "null"
        if execution_config.use_cuda_graph:
            execution_config.use_cuda_graph = False

        super().__init__(simulation_config, rendering_config, execution_config, engine_config)

        self.state_chain = []
        self.trajectory: Optional[TrajectoryData] = None

    def save_usd(self, output_path: str, fps: int = 60):
        """
        Exports the recorded trajectory to a USD file for visualization.
        """
        try:
            from newton.viewer import ViewerUSD
        except ImportError:
            print("Warning: ViewerUSD not available. Install usd-core to export USD.")
            return

        if self.trajectory is None or self.trajectory.step_count.numpy()[0] == 0:
            print("Warning: No trajectory data to save.")
            return
            
        print(f"Exporting trajectory to {output_path}...")
        
        # Use a temporary state for rendering
        render_state = self.model.state()
        num_steps = self.trajectory.step_count.numpy()[0]
        
        viewer = ViewerUSD(output_path, fps=fps, num_frames=num_steps)
        viewer.set_model(self.model)
        
        for i in range(num_steps):
            time = i * self.effective_timestep
            viewer.begin_frame(time)
            
            wp.launch(
                kernel=flatten_trajectory_slice_kernel,
                dim=(self.model.num_worlds, self.model.body_count),
                inputs=[
                    self.trajectory.body_q,
                    i,
                    render_state.body_q,
                    self.model.body_count,
                    self.model.num_worlds,
                ],
                device=self.model.device
            )
            
            viewer.log_state(render_state)
            viewer.end_frame()
            
        viewer.close()

    def reset_state(self, q: Optional[wp.array] = None, qd: Optional[wp.array] = None):
        if q is not None:
            wp.copy(self.current_state.body_q, q)
        if qd is not None:
            wp.copy(self.current_state.body_qd, qd)

        self.current_state.clear_forces()
        if self.trajectory:
            self.trajectory.step_count.zero_()

    def forward(
        self,
        steps: int = None,
        tape: Optional[wp.Tape] = None,
        q: Optional[wp.array] = None,
        qd: Optional[wp.array] = None,
    ):
        """
        Runs the simulation.
        CRITICAL: This method MUST NOT be called inside an active `with tape:` block.
        Pass the tape object explicitly instead.
        """
        if wp.context.runtime.tape is not None:
            raise RuntimeError(
                "DifferentiableSimulator.forward() was called inside an active Tape. "
                "You must close/exit the tape before calling forward, and pass the tape "
                "as an argument to this function instead. This is required to handle "
                "non-differentiable operations (like collision) correctly."
            )

        if steps is None:
            if isinstance(self.engine_config, AxionEngineConfig):
                steps = self.engine_config.max_trajectory_steps
            else:
                steps = 128

        # Reset to initial conditions
        if tape:
            with tape:
                self.reset_state(q=q, qd=qd)
        else:
            self.reset_state(q=q, qd=qd)

        # Dispatch engine
        if isinstance(self.engine_config, AxionEngineConfig):
            self._run_axion_simulation(steps, tape)
        elif isinstance(self.engine_config, SemiImplicitEngineConfig):
            self._run_newton_simulation(steps, tape)
        else:
            raise NotImplementedError(
                "Differentiation only supported for Axion and SemiImplicit engines."
            )

    def _allocate_trajectory(self, steps: int):
        if self.trajectory is None or self.trajectory.body_q.shape[0] < steps:
            trajectory_shape = (steps, self.model.num_worlds, self.model.body_count)
            self.trajectory = TrajectoryData(
                body_q=wp.zeros(trajectory_shape, dtype=wp.transform, requires_grad=True),
                body_u=wp.zeros(trajectory_shape, dtype=wp.spatial_vector, requires_grad=True),
                step_count=wp.zeros(1, dtype=wp.int32),
            )
        else:
            self.trajectory.step_count.zero_()

    def _run_axion_simulation(self, steps: int, tape: Optional[wp.Tape]):
        self._allocate_trajectory(steps)
        for i in range(steps):
            # Axion engine handles collision internally, but if we need to support
            # differentiation, we might need to expose the phases.
            # For now, assuming AxionEngine.step handles tape safety or is fully differentiable.
            # (If AxionEngine also runs non-diff collision, it needs similar splitting).

            if tape:
                with tape:
                    self._single_physics_step(i)
                    self._save_trajectory_axion(i)
            else:
                self._single_physics_step(i)
                self._save_trajectory_axion(i)
        
        self.trajectory.step_count.fill_(steps)

    def _save_trajectory_axion(self, step_idx):
        wp.launch(
            kernel=save_to_trajectory_kernel_2d,
            dim=(self.model.num_worlds, self.model.body_count),
            inputs=[
                self.solver.data.body_q,
                self.solver.data.body_u,
                self.model.num_worlds,
                self.model.body_count,
                step_idx,
                self.trajectory.body_q,
                self.trajectory.body_u,
            ],
            device=self.solver.device,
        )

    def _run_newton_simulation(self, steps: int, tape: Optional[wp.Tape]):
        # 1. Allocate Chain
        required_len = steps + 1
        current_len = len(self.state_chain)
        if current_len < required_len:
            for _ in range(required_len - current_len):
                self.state_chain.append(self.model.state())

        if tape:
            with tape:
                self._copy_state(self.state_chain[0], self.current_state)
        else:
            self._copy_state(self.state_chain[0], self.current_state)
        
        self._allocate_trajectory(steps)

        # 2. Loop
        for i in range(steps):
            state_curr = self.state_chain[i]
            state_next = self.state_chain[i + 1]

            # --- A. Non-Differentiable Phase (Collision) ---
            # Run strictly OUTSIDE any tape
            state_curr.clear_forces()
            contacts = self.model.collide(state_curr)

            # --- B. Differentiable Phase (Dynamics) ---
            if tape:
                with tape:
                    self._step_dynamics(i, state_curr, state_next, contacts)
            else:
                self._step_dynamics(i, state_curr, state_next, contacts)

        # Finalize
        self._copy_state(self.current_state, self.state_chain[steps])
        self.trajectory.step_count.fill_(steps)

    def _step_dynamics(self, step_idx, state_curr, state_next, contacts):
        self.control_policy(state_curr)
        self.solver.step(
            state_in=state_curr,
            state_out=state_next,
            control=self.control,
            contacts=contacts,
            dt=self.effective_timestep,
        )
        wp.launch(
            kernel=save_to_trajectory_kernel_1d,
            dim=(self.model.num_worlds, self.model.body_count),
            inputs=[
                state_next.body_q,
                state_next.body_qd,
                self.model.num_worlds,
                self.model.body_count,
                step_idx,
                self.trajectory.body_q,
                self.trajectory.body_u,
            ],
            device=wp.get_device(),
        )

    @abstractmethod
    def compute_loss(self) -> wp.array:
        pass
