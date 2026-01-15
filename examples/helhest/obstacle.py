import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class HelhestObstacleSimulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
        )

        # Helhest DOFs: 6 (Base) + 1 (Left) + 1 (Right) + 1 (Rear) = 9
        # Drive forward: Left and Right wheels
        robot_joint_target = np.array([0.0] * 6 + [6.0, 6.0, 6.0], dtype=np.float32)

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
        wp.copy(self.control.joint_target, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model with an obstacle.
        """

        # Robot position
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 0.6

        create_helhest_model(
            self.builder, xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity())
        )

        # Ground Plane parameters
        FRICTION = 1.0
        RESTITUTION = 0.0
        KE = 1.0e4
        KD = 1.0e3
        KF = 1.0e3

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        # Reduced size: hx=0.5 (1m length), hy=1.0 (2m width), hz=0.1 (0.2m height)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.16,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((8.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.22,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # Ground plane
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_obstacle_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = HelhestObstacleSimulator(sim_config, render_config, exec_config, engine_config)
    simulator.run()


if __name__ == "__main__":
    helhest_obstacle_example()
