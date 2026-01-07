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
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.taros_4.common import create_taros4_model
except ImportError:
    from common import create_taros4_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        # Taros-4 DOFs: 6 (Base) + 4 (Wheels) = 10
        # High speed for flipping: 30.0 rad/s
        robot_joint_target = np.array([0.0] * 6 + [-30.0, -30.0, -30.0, -30.0], dtype=np.float32)

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
        Builds the unified Taros-4 model for the flip scenario.
        """

        # Robot position
        robot_x = -3.0
        robot_y = 0.0
        robot_z = 1.0

        create_taros4_model(
            self.builder,
            xform=wp.transform(
                (robot_x, robot_y, robot_z),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            ),
        )

        # Environment parameters from original flip.py
        FRICTION = 0.8
        RESTITUTION = 0.0
        KE = 90000.0
        KD = 50000.0
        KF = 2000.0

        # --- Add Static Obstacles and Ground ---

        # Obstacle 1
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=1.75,
            hy=1.5,
            hz=0.15,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
                ke=KE,
                kd=KD,
                kf=KF,
            ),
        )

        # Obstacle 2
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=0.75,
            hy=1.75,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
                ke=KE,
                kd=KD,
                kf=KF,
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


@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def taros4_flip_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    taros4_flip_example()
