from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import newton
import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from axion.core.engine_config import SemiImplicitEngineConfig

from .base_simulator import BaseSimulator
from .base_simulator import ExecutionConfig
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@dataclass
class TrajectoryData:
    body_q: wp.array
    body_u: wp.array
    step_count: wp.array


# @wp.kernel
# def save_to_trajectory_kernel_1d(
#     body_q: wp.array(dtype=wp.transform, ndim=1),
#     body_u: wp.array(dtype=wp.spatial_vector, ndim=1),
#     num_worlds: int,
#     num_bodies: int,
#     step_idx: int,
#     body_q_trajectory: wp.array(dtype=wp.transform, ndim=3),
#     body_u_trajectory: wp.array(dtype=wp.spatial_vector, ndim=3),
# ):
#     world_idx, body_idx = wp.tid()
#     if world_idx >= num_worlds or body_idx >= num_bodies:
#         return
#     flat_idx = world_idx * num_bodies + body_idx
#     body_q_trajectory[step_idx, world_idx, body_idx] = body_q[flat_idx]
#     body_u_trajectory[step_idx, world_idx, body_idx] = body_u[flat_idx]
#
#
# @wp.kernel
# def save_to_trajectory_kernel_2d(
#     body_q: wp.array(dtype=wp.transform, ndim=2),
#     body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
#     num_worlds: int,
#     num_bodies: int,
#     step_idx: int,
#     body_q_trajectory: wp.array(dtype=wp.transform, ndim=3),
#     body_u_trajectory: wp.array(dtype=wp.spatial_vector, ndim=3),
# ):
#     world_idx, body_idx = wp.tid()
#     if world_idx >= num_worlds or body_idx >= num_bodies:
#         return
#     body_q_trajectory[step_idx, world_idx, body_idx] = body_q[world_idx, body_idx]
#     body_u_trajectory[step_idx, world_idx, body_idx] = body_u[world_idx, body_idx]
#
#
# @wp.kernel
# def flatten_trajectory_slice_kernel(
#     traj_q: wp.array(dtype=wp.transform, ndim=3),
#     step_idx: int,
#     flat_q: wp.array(dtype=wp.transform, ndim=1),
#     num_bodies: int,
#     num_worlds: int,
# ):
#     world_idx, body_idx = wp.tid()
#     if world_idx >= num_worlds or body_idx >= num_bodies:
#         return
#     flat_idx = world_idx * num_bodies + body_idx
#     flat_q[flat_idx] = traj_q[step_idx, world_idx, body_idx]


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
        super().__init__(
            simulation_config,
            rendering_config,
            execution_config,
            engine_config,
        )

        if not self.model.body_mass.requires_grad:
            raise RuntimeError(
                "DifferentiableSimulator requires a differentiable model.\n"
                "Error: The model provided by `build_model()` was not finalized with gradients enabled.\n"
                "Fix: Ensure your build_model() returns `builder.finalize(requires_grad=True)`."
            )

        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.total_sim_steps + 1)
        ]
        # self.trajectory: Optional[TrajectoryData] = None
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline.from_model(self.model)
        self.contacts = self.model.collide(self.states[0], self.collision_pipeline)

        self.viewer = newton.viewer.ViewerGL()
        self.viewer.set_model(self.model)
        self.cuda_graph: Optional[wp.Graph] = None
        self.tape = wp.Tape()
        self.loss = wp.zeros(1, dtype=wp.float32)

    def forward_backward(self):
        self.tape = wp.Tape()
        
        # Dispatch engine
        # Note: We do NOT wrap this in `with self.tape:` because the engine runner
        # handles granular tape scoping (e.g. excluding collision).
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
        if self.cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self.forward_backward()

    def perform_step(self):
        """
        Runs the simulation step (differentiable).
        """
        self.diff_step()

    # def _allocate_trajectory(self, steps: int):
    #     if self.trajectory is None or self.trajectory.body_q.shape[0] < steps:
    #         trajectory_shape = (steps, self.model.num_worlds, self.model.body_count)
    #         self.trajectory = TrajectoryData(
    #             body_q=wp.zeros(trajectory_shape, dtype=wp.transform, requires_grad=True),
    #             body_u=wp.zeros(trajectory_shape, dtype=wp.spatial_vector, requires_grad=True),
    #             step_count=wp.zeros(1, dtype=wp.int32),
    #         )
    #     else:
    #         self.trajectory.step_count.zero_()

    # def _save_trajectory_axion(self, step_idx):
    #     wp.launch(
    #         kernel=save_to_trajectory_kernel_2d,
    #         dim=(self.model.num_worlds, self.model.body_count),
    #         inputs=[
    #             self.solver.data.body_q,
    #             self.solver.data.body_u,
    #             self.model.num_worlds,
    #             self.model.body_count,
    #             step_idx,
    #             self.trajectory.body_q,
    #             self.trajectory.body_u,
    #         ],
    #         device=self.solver.device,
    #     )

    def _run_axion_forward(self):
        return
        # self._allocate_trajectory(steps)
        # for i in range(steps):
        #     # Axion engine handles collision internally, but if we need to support
        #     # differentiation, we might need to expose the phases.
        #     # For now, assuming AxionEngine.step handles tape safety or is fully differentiable.
        #     # (If AxionEngine also runs non-diff collision, it needs similar splitting).
        #
        #     if tape:
        #         with tape:
        #             self._single_physics_step(i)
        #             self._save_trajectory_axion(i)
        #     else:
        #         self._single_physics_step(i)
        #         self._save_trajectory_axion(i)
        #
        # self.trajectory.step_count.fill_(steps)

    def _run_newton_forward(self):
        for i in range(self.total_sim_steps):
            with self.tape:
                self.states[i].clear_forces()

            self.contacts = self.model.collide(self.states[i], self.collision_pipeline)
            # self.control_policy(self.states[i])

            with self.tape:
                self.solver.step(
                    state_in=self.states[i],
                    state_out=self.states[i + 1],
                    control=self.control,
                    contacts=self.contacts,
                    dt=self.effective_timestep,
                )

        with self.tape:
            self.compute_loss()

    @abstractmethod
    def update(self):
        # Should modify the self.states[0]
        pass

    @abstractmethod
    def compute_loss(self):
        # This should take the self.states[-1] and modify the self.loss
        pass
