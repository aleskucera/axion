from abc import ABC
from dataclasses import dataclass
from typing import Optional

import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
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
def save_to_trajectory_kernel(
    # --- Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Parameters ---
    num_worlds: int,
    num_bodies: int,
    # --- Outputs ---
    step_count: wp.array(dtype=wp.int32),
    body_q_trajectory: wp.array(dtype=wp.transform, ndim=3),
    body_u_trajectory: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    world_idx, body_idx = wp.tid()

    if world_idx >= num_worlds or body_idx >= num_bodies:
        return

    step_idx = step_count[0] * num_worlds * num_bodies
    body_q_trajectory[step_idx, world_idx, body_idx] = body_q[world_idx, body_idx]
    body_u_trajectory[step_idx, world_idx, body_idx] = body_u[world_idx, body_idx]

    if world_idx == 0 and body_idx == 0:
        step_count[0] = step_count[0] + 1


class DifferentiableSimulator(BaseSimulator, ABC):
    """
    A specialized simulator for running differentiable physics episodes.
    Optimized for Tape recording, fixed-horizon execution, and backpropagation.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: EngineConfig,
    ):
        # 2. Enforce Null Viewer
        if rendering_config.vis_type != "null":
            print("INFO: DifferentiableSimulator forcing vis_type='null'")
            rendering_config.vis_type = "null"

        # 3. Disable CUDA Graphs
        if execution_config.use_cuda_graph:
            print("INFO: DifferentiableSimulator forcing use_cuda_graph=False")
            execution_config.use_cuda_graph = False

        super().__init__(simulation_config, rendering_config, execution_config, engine_config)

    def reset_state(
        self,
        q: Optional[wp.array] = None,
        qd: Optional[wp.array] = None,
    ):
        """
        Resets the current simulation state and trajectory counter.
        """
        if q is not None:
            wp.copy(self.current_state.body_q, q)

        if qd is not None:
            wp.copy(self.current_state.body_qd, qd)

    def compute_gradient(self, steps: int = None):
        """
        Runs the simulation for a specified number of steps, optionally recording gradients.
        """
        if isinstance(self.engine_config, AxionEngineConfig):
            self._axion_solver_forward(steps)
        else:
            self._axion_newton_forward(steps)

    def _axion_solver_simulation(self, steps: int):
        trajectory_shape = (steps, self.model.num_worlds, self.model.body_count)
        self.trajectory = TrajectoryData(
            body_q=wp.zeros(trajectory_shape, dtype=wp.transform),
            body_u=wp.zeros(trajectory_shape, dtype=wp.spatial_vector),
            current_step=wp.zeros(1, dtype=wp.int32),
        )

        for i in range(steps):
            self._single_physics_step(i)
            wp.launch(
                kernel=save_to_trajectory_kernel,
                dim=(self.model.num_worlds, self.model.body_count),
                inputs=[
                    self.solver.data.body_q,
                    self.solver.data.body_u,
                    self.model.num_worlds,
                    self.model.body_count,
                ],
                outputs=[
                    self.trajectory.step_count,
                    self.trajectory.body_q,
                    self.trajectory.body_u,
                ],
                device=self.device,
            )

        # TODO: Implement the backwarp pass later

    def _newton_solver_simulation(self, steps: int):
        loss = wp.zeros(1, dtype=wp.float32)
        tape = wp.Tape()
        with tape:
            for i in range(steps):
                self._single_physics_step(i)

            self.loss_fn(self.current_state, loss)

        tape.backward(loss)

    def loss_fn(self, state: State, target_state: State, loss: wp.array):
        pass
