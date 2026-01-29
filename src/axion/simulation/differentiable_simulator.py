from abc import ABC
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional

import warp as wp
from newton import State

from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from .base_simulator import BaseSimulator
from .base_simulator import ExecutionConfig
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@dataclass
class TrajectoryData:
    step_count: int

    body_q_traj: wp.array
    body_u_traj: wp.array

    current_step: int = 0


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
        # 1. Enforce Differentiable Configuration
        if isinstance(engine_config, AxionEngineConfig):
            updates = {}
            if not engine_config.differentiable_simulation:
                print("INFO: DifferentiableSimulator forcing differentiable_simulation=True")
                updates["differentiable_simulation"] = True

            if engine_config.max_trajectory_steps <= 0:
                default_steps = 1024
                print(
                    f"INFO: DifferentiableSimulator setting max_trajectory_steps={default_steps}"
                )
                updates["max_trajectory_steps"] = default_steps

            if updates:
                engine_config = replace(engine_config, **updates)

        # 2. Enforce Null Viewer
        if rendering_config.vis_type != "null":
            print("INFO: DifferentiableSimulator forcing vis_type='null'")
            rendering_config.vis_type = "null"

        # 3. Disable CUDA Graphs
        if execution_config.use_cuda_graph:
            print("INFO: DifferentiableSimulator forcing use_cuda_graph=False")
            execution_config.use_cuda_graph = False

        super().__init__(
            simulation_config, rendering_config, execution_config, engine_config
        )

        # 4. Allocate Trajectory Storage
        # Note: We allocate this here in the simulator, NOT in the engine.
        trajectory_steps = self.engine_config.max_trajectory_steps
        self.trajectory = TrajectoryData(
            step_count=trajectory_steps,
            body_q_traj=wp.zeros(
                (trajectory_steps, self.model.num_worlds, self.model.body_count),
                dtype=wp.transform,
            ),
            body_u_traj=wp.zeros(
                (trajectory_steps, self.model.num_worlds, self.model.body_count),
                dtype=wp.spatial_vector,
            ),
        )

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

        self.current_state.clear_forces()
        self.trajectory.current_step = 0

    def forward(self, steps: int = None, record_tape: bool = True) -> Optional[wp.Tape]:
        """
        Runs the simulation for a specified number of steps, optionally recording gradients.
        """
        if steps is None:
            if isinstance(self.engine_config, AxionEngineConfig):
                steps = self.engine_config.max_trajectory_steps
            else:
                raise ValueError("Must specify 'steps' for non-Axion engine configs.")

        if steps > self.trajectory.step_count:
            raise ValueError(
                f"Requested {steps} steps but trajectory capacity is {self.trajectory.step_count}."
            )

        self.trajectory.current_step = 0

        tape = wp.Tape() if record_tape else None
        if tape:
            tape.__enter__()

        try:
            for i in range(steps):
                self._single_physics_step(i)
                self._save_to_trajectory(i)
                self.trajectory.current_step += 1
                
        finally:
            if tape:
                tape.__exit__(None, None, None)

        return tape

    def _save_to_trajectory(self, step_idx: int):
        """Helper to copy current engine state into the trajectory buffer."""
        # Using the engine's internal data directly for speed/access
        data = self.solver.data
        
        wp.copy(self.trajectory.body_q_traj[step_idx], data.body_q)
        wp.copy(self.trajectory.body_u_traj[step_idx], data.body_u)
