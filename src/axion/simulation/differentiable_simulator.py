from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import newton
import numpy as np
import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from axion.core.engine_config import SemiImplicitEngineConfig
from axion.core.logging_config import LoggingConfig

from .base_simulator import BaseSimulator
from .sim_config import ExecutionConfig
from .sim_config import RenderingConfig
from .sim_config import SimulationConfig


class DifferentiableSimulator(BaseSimulator, ABC):
    """
    A specialized simulator for running differentiable physics episodes.

    This simulator manages a full trajectory of states to support backpropagation
    through time (BPTT) or implicit differentiation schemes.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: Optional[LoggingConfig] = None,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            execution_config,
            engine_config,
            logging_config,
        )

        # 1. Validation for Differentiability
        if not self.model.body_mass.requires_grad:
            raise RuntimeError(
                "DifferentiableSimulator requires a differentiable model.\n"
                "Error: The model provided by `build_model()` was not finalized with gradients enabled.\n"
                "Fix: Ensure your build_model() returns `builder.finalize(requires_grad=True)`."
            )

        # 2. State Management
        # Unlike interactive sim (which keeps 1 state), we need the full history for gradients.
        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.clock.total_sim_steps + 1)
        ]
        self.control = self.model.control()

        # 3. Collision Setup
        self.collision_pipeline = newton.CollisionPipeline.from_model(self.model)
        # Initialize contact data
        self.contacts = self.model.collide(self.states[0], self.collision_pipeline)

        # 4. Viewer Initialization (Using Factory)
        self.viewer = self.rendering_config.create_viewer(
            self.model, num_segments=self.clock.total_sim_steps
        )
        if self.viewer:
            self.viewer.set_model(self.model)

        # 5. Gradient Tape & Graphing
        self.cuda_graph: Optional[wp.Graph] = None
        self.tape = wp.Tape()
        self.loss = wp.zeros(1, dtype=wp.float32)

        self._tracked_bodies = {}

    # ============== ABSTRACT METHODS ==============
    # These methods together with the world model define the experiment
    @abstractmethod
    def update(self):
        """Should modify self.states[0] before the run."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Should take self.states[-1] (or full trajectory) and modify self.loss."""
        pass

    # ============== DIFFERENTIABLE SIMULATION METHODS ==============
    def diff_step(self):
        if self.use_cuda_graph and self.cuda_graph is None:
            with wp.ScopedCapture() as capture:
                self._forward_backward()
            self.cuda_graph = capture.graph

        if self.use_cuda_graph and self.cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self._forward_backward()

    def _forward_backward(self):
        if isinstance(self.engine_config, AxionEngineConfig):
            self._axion_forward_backward_explicit()
        elif isinstance(self.engine_config, SemiImplicitEngineConfig):
            self._newton_forward_backward()
        else:
            raise NotImplementedError(
                "Differentiation only supported for Axion and SemiImplicit engines."
            )

    def _axion_forward_backward_explicit(self):
        # --- FORWARD PASS ---
        for i in range(self.clock.total_sim_steps):
            # Clear forces on the tape
            with self.tape:
                self.states[i].clear_forces()

            # Collision is usually non-differentiable or handled separately
            self.contacts = self.model.collide(self.states[i], self.collision_pipeline)

            # Integrate step on the tape
            with self.tape:
                self.control_policy(self.states[i])
                self.solver.step(
                    state_in=self.states[i],
                    state_out=self.states[i + 1],
                    control=self.control,
                    contacts=self.contacts,
                    dt=self.clock.dt,
                )

        with self.tape:
            self.compute_loss()

        # --- BACKWARD PASS ---
        self.tape.backward(self.loss)

    def _axion_forward_backward_implicit(self):
        pass

    def _newton_forward_backward(self):
        """
        Standard explicit differentiation (unrolling loop on tape).
        Used for Newton solvers (SemiImplicit, etc).
        """

        # Zero-out the gradients
        self.tape.zero()

        # --- FORWARD PASS ---
        for i in range(self.clock.total_sim_steps):
            # Clear forces on the tape
            with self.tape:
                self.states[i].clear_forces()

            # Collision is usually non-differentiable or handled separately
            self.contacts = self.model.collide(self.states[i], self.collision_pipeline)

            # Optional: Apply control policy if defined
            # self.control_policy(self.states[i])

            # Integrate step on the tape
            with self.tape:
                self.solver.step(
                    state_in=self.states[i],
                    state_out=self.states[i + 1],
                    control=self.control,
                    contacts=self.contacts,
                    dt=self.clock.dt,
                )

        # Compute loss on the final state (or trajectory)
        with self.tape:
            self.compute_loss()

        # --- BACKWARD PASS ---
        self.tape.backward(self.loss)

    # ============== VISUALIZATION METHODS ==============
    def track_body(self, body_idx: int, name: str = None, color: tuple = (1.0, 0.0, 0.0)):
        """
        Register a body to have its trajectory visualized automatically.
        """
        if name is None:
            name = f"body_{body_idx}"
        self._tracked_bodies[body_idx] = {"name": name, "color": color}

    def render_episode(
        self,
        iteration: int = 0,
        callback: Optional[Callable[[Any, int, newton.State], None]] = None,
        loop: bool = False,
        loops_count: int = 1,
        playback_speed: float = 1.0,
    ):
        """
        Replays the simulation episode in the viewer, rendering tracked trajectories.

        Args:
            iteration: Current training iteration (used for unique names).
            callback: Optional function(viewer, step_idx, state) for extra rendering.
            loop: Whether to replay the episode multiple times.
            loops_count: Number of times to replay if loop is True.
            playback_speed: Speed multiplier (1.0 = real time, 0.5 = slow motion).
        """
        if not self.viewer:
            return

        import time

        # 1. Pre-calculate trajectories for all tracked bodies
        # This prevents re-extracting numpy arrays every single frame
        trajectories = {}
        for body_idx, metadata in self._tracked_bodies.items():
            path = []
            for state in self.states:
                # Extract position: (num_worlds, num_bodies, 7) -> (x, y, z)
                # We use [0] assuming single world for differentiable sim
                q = state.body_q.numpy()[body_idx]
                path.append(q[:3])
            trajectories[body_idx] = np.array(path, dtype=np.float32)

        # 2. Setup Timing
        dt = self.clock.dt
        total_sim_time = len(self.states) * dt
        plays = loops_count if loop else 1

        # 3. Replay Loop
        for play_idx in range(plays):
            # Sync start time for this loop iteration
            start_wall_time = time.time()

            while True:
                # Calculate how much simulation time has passed based on wall clock
                current_wall_time = time.time()
                elapsed_sim_time = (current_wall_time - start_wall_time) * playback_speed

                # Check if we reached the end of the episode
                if elapsed_sim_time > total_sim_time:
                    break

                # Map time to the closest state index
                step_idx = int(elapsed_sim_time / dt)
                step_idx = min(step_idx, len(self.states) - 1)
                state = self.states[step_idx]

                self.viewer.begin_frame(elapsed_sim_time)

                # A. Log Physics State
                self.viewer.log_state(state)

                # B. Log Trajectories (Draw full path)
                for body_idx, metadata in self._tracked_bodies.items():
                    pts = trajectories[body_idx]
                    if len(pts) > 1:
                        line_name = f"/traj_{iteration}/{metadata['name']}"
                        self.viewer.log_lines(
                            line_name,
                            wp.array(pts[:-1], dtype=wp.vec3),
                            wp.array(pts[1:], dtype=wp.vec3),
                            metadata["color"],
                        )

                # C. Custom Callback (Target, Loss, etc.)
                if callback:
                    callback(self.viewer, step_idx, state)

                self.viewer.end_frame()
