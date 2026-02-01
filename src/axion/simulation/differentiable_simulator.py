from abc import ABC
from abc import abstractmethod
from typing import Optional

import newton
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
            self.model.state(requires_grad=True) for _ in range(self.total_sim_steps + 1)
        ]
        self.control = self.model.control()

        # 3. Collision Setup
        self.collision_pipeline = newton.CollisionPipeline.from_model(self.model)
        # Initialize contact data
        self.contacts = self.model.collide(self.states[0], self.collision_pipeline)

        # 4. Viewer Initialization (Using Factory)
        self.viewer = self.rendering_config.create_viewer(
            self.model, num_segments=self.total_sim_steps
        )
        if self.viewer:
            self.viewer.set_model(self.model)

        # 5. Gradient Tape & Graphing
        self.cuda_graph: Optional[wp.Graph] = None
        self.tape = wp.Tape()
        self.loss = wp.zeros(1, dtype=wp.float32)

    def forward_backward(self):
        """
        Runs the simulation forward and computes gradients backward.
        """
        self.tape = wp.Tape()

        # Dispatch engine execution strategy
        if isinstance(self.engine_config, AxionEngineConfig):
            self._run_axion_forward()
        elif isinstance(self.engine_config, SemiImplicitEngineConfig):
            self._run_newton_forward()
        else:
            raise NotImplementedError(
                "Differentiation only supported for Axion and SemiImplicit engines."
            )

        self.tape.backward(self.loss)

    def capture(self):
        """Captures the simulation step into a CUDA graph if enabled."""
        if (
            self.execution_config.use_cuda_graph
            and wp.get_device().is_cuda
            and self.cuda_graph is None
        ):
            # We must run one pass to capture the graph
            self.tape = wp.Tape()
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.cuda_graph = capture.graph

    def diff_step(self):
        """Executes one differentiable step (either graph-launched or eager)."""
        if self.cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self.forward_backward()

    def perform_step(self):
        """
        Runs the simulation step (differentiable).
        API Compatibility method.
        """
        self.diff_step()

    def _run_axion_forward(self):
        # TODO: Implement implicit differentiation trajectory for Axion
        pass

    def _run_newton_forward(self):
        """
        Standard explicit differentiation (unrolling loop on tape).
        Used for Newton solvers (SemiImplicit, etc).
        """
        dt = self.effective_timestep

        for i in range(self.total_sim_steps):
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
                    dt=dt,
                )

        # Compute loss on the final state (or trajectory)
        with self.tape:
            self.compute_loss()

    @abstractmethod
    def update(self):
        """Should modify self.states[0] before the run."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Should take self.states[-1] (or full trajectory) and modify self.loss."""
        pass
