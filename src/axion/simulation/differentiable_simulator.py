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
    # --- Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=1),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=1),
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

    step_idx = step_count[0]
    flat_idx = world_idx * num_bodies + body_idx

    body_q_trajectory[step_idx, world_idx, body_idx] = body_q[flat_idx]
    body_u_trajectory[step_idx, world_idx, body_idx] = body_u[flat_idx]


@wp.kernel
def save_to_trajectory_kernel_2d(
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

    step_idx = step_count[0]

    body_q_trajectory[step_idx, world_idx, body_idx] = body_q[world_idx, body_idx]
    body_u_trajectory[step_idx, world_idx, body_idx] = body_u[world_idx, body_idx]


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
                print(f"INFO: DifferentiableSimulator setting max_trajectory_steps={default_steps}")
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

        super().__init__(simulation_config, rendering_config, execution_config, engine_config)

        # Storage for explicit solver state chain (required for Tape differentiation of explicit steps)
        self.state_chain = []
        self.trajectory: Optional[TrajectoryData] = None

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

        # Reset trajectory counter
        if self.trajectory:
            self.trajectory.step_count.zero_()

        def forward(
            self,
            steps: int = None,
            record_tape: bool = True,
            q: Optional[wp.array] = None,
            qd: Optional[wp.array] = None,
        ) -> Optional[wp.Tape]:
            """

            Runs the simulation for a specified number of steps, optionally recording gradients.

            Automatically dispatches to the correct differentiation strategy based on the engine.



            Args:

                steps: Number of steps to simulate.

                record_tape: Whether to record operations.

                q: Optional initial positions to reset to.

                qd: Optional initial velocities to reset to.

            """

            if steps is None:

                if isinstance(self.engine_config, AxionEngineConfig):

                    steps = self.engine_config.max_trajectory_steps

                else:

                    # Default for other solvers if not specified

                    steps = 128

            # Reset to initial conditions (optionally overridden)

            self.reset_state(q=q, qd=qd)

            tape = wp.Tape() if record_tape else None

            try:

                if isinstance(self.engine_config, AxionEngineConfig):

                    self._run_axion_simulation(steps, tape)

                elif isinstance(self.engine_config, SemiImplicitEngineConfig):

                    self._run_newton_simulation(steps, tape)

                else:

                    raise NotImplementedError(
                        "Differentiation only supported for Axion and SemiImplicit engines."
                    )

            finally:

                pass

            return tape

        def _allocate_trajectory(self, steps: int):
            """Allocates the SoA trajectory buffer if needed or if size changed."""

            if self.trajectory is None or self.trajectory.body_q.shape[0] < steps:

                trajectory_shape = (steps, self.model.num_worlds, self.model.body_count)

                self.trajectory = TrajectoryData(
                    body_q=wp.zeros(trajectory_shape, dtype=wp.transform),
                    body_u=wp.zeros(trajectory_shape, dtype=wp.spatial_vector),
                    step_count=wp.zeros(1, dtype=wp.int32),
                )

            else:

                self.trajectory.step_count.zero_()

        def _run_axion_simulation(self, steps: int, tape: Optional[wp.Tape]):
            """

            Runs AxionEngine.

            Note: Axion handles its own internal state, so we just snapshot to trajectory.

            """

            self._allocate_trajectory(steps)

            for i in range(steps):

                if tape:

                    tape.__enter__()

                try:

                    # This uses the BaseSimulator logic

                    self._single_physics_step(i)

                    # Save to trajectory buffer

                    wp.launch(
                        kernel=save_to_trajectory_kernel_2d,
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
                        device=self.solver.device,
                    )

                finally:

                    if tape:

                        tape.__exit__(None, None, None)

        def _run_newton_simulation(self, steps: int, tape: Optional[wp.Tape]):
            """

            Runs Newton Semi-Implicit Solver.

            CRITICAL: Allocates a chain of State objects to preserve history for wp.Tape.

            """

            # 1. Allocate Chain: [S_0, S_1, ..., S_steps]

            required_len = steps + 1

            current_len = len(self.state_chain)

            if current_len < required_len:

                for _ in range(required_len - current_len):

                    self.state_chain.append(self.model.state())

            # 2. Initialize S_0 from current_state

            self._copy_state(self.state_chain[0], self.current_state)

            # 3. Allocate Trajectory

            self._allocate_trajectory(steps)

            # 4. Simulation Loop

            for i in range(steps):

                state_curr = self.state_chain[i]

                state_next = self.state_chain[i + 1]

                # --- Non-recorded operations (Collision etc) ---

                state_curr.clear_forces()

                contacts = self.model.collide(state_curr)

                # --- Recorded operations ---

                if tape:

                    tape.__enter__()

                try:

                    # Control

                    self.control_policy(state_curr)

                    # Step

                    self.solver.step(
                        state_in=state_curr,
                        state_out=state_next,
                        control=self.control,
                        contacts=contacts,
                        dt=self.effective_timestep,
                    )

                    # Save to trajectory

                    wp.launch(
                        kernel=save_to_trajectory_kernel_1d,
                        dim=(self.model.num_worlds, self.model.body_count),
                        inputs=[
                            state_next.body_q,
                            state_next.body_qd,
                            self.model.num_worlds,
                            self.model.body_count,
                        ],
                        outputs=[
                            self.trajectory.step_count,
                            self.trajectory.body_q,
                            self.trajectory.body_u,
                        ],
                        device=wp.get_device(),
                    )

                finally:

                    if tape:

                        tape.__exit__(None, None, None)

            # Update the simulator's main current_state to the final state

            self._copy_state(self.current_state, self.state_chain[steps])

    @abstractmethod
    def compute_loss(self) -> wp.array:
        """
        User must implement this to define the loss function based on self.trajectory.
        """
        pass

