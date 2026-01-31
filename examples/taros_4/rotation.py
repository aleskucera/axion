import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.taros_4.common import create_taros4_model
except ImportError:
    from common import create_taros4_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class TarosRotationSimulator(InteractiveSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config):
        self.left_indices_cpu = []
        self.right_indices_cpu = []
        super().__init__(sim_config, render_config, exec_config, engine_config)

        # Taros-4 DOFs: 6 (Base) + 4 (Wheels) = 10
        # Rotation: Left wheels -12.0, Right wheels 12.0
        # Indices: 6:FL, 7:FR, 8:RL, 9:RR
        robot_joint_target = np.array([0.0] * 6 + [-12.0, 12.0, -12.0, 12.0], dtype=np.float32)

        joint_target = np.tile(robot_joint_target, self.simulation_config.num_worlds)
        self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        wp.copy(self.control.joint_target_vel, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Taros-4 model for the rotation example.
        """

        # Robot position
        robot_x = -1.0
        robot_y = 0.0
        robot_z = 1.0

        create_taros4_model(
            self.builder, xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity())
        )

        # Environment parameters from original rotation.py
        FRICTION = 0.8
        RESTITUTION = 0.0
        KE = 60000.0
        KD = 30000.0
        KF = 500.0

        # Ground plane
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=KE, kd=KD, kf=KF, mu=FRICTION, restitution=RESTITUTION
            )
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def taros4_rotation_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = TarosRotationSimulator(sim_config, render_config, exec_config, engine_config)
    simulator.run()


if __name__ == "__main__":
    taros4_rotation_example()
