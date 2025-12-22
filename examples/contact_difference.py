import os
import pathlib
from typing import override

import hydra
import newton
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.joinpath("assets")


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

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    def build_model(self) -> newton.Model:
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 0.0
        RESTITUTION = 0.0
        WHEEL_DENSITY = 300

        ball_x = 0.5
        ball_y = 4.0
        ball_z = 2.0

        # Left Wheel
        ball1 = self.builder.add_body(
            xform=wp.transform((ball_x, ball_y, ball_z), wp.quat_identity()),
            key="left_wheel",
        )
        self.builder.add_shape_sphere(
            body=ball1,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                contact_margin=0.1,
                is_visible=True,
            ),
        )

        # Right Wheel
        ball2 = self.builder.add_body(
            xform=wp.transform((ball_x + 4.0, ball_y, ball_z), wp.quat_identity()),
            key="right_wheel",
        )
        self.builder.add_shape_sphere(
            body=ball2,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                contact_margin=0.1,
                is_visible=True,
            ),
        )

        # --- Add Static Obstacles and Ground ---
        box = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            key="box",
            mass=0.0,
        )

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        self.builder.add_shape_box(
            body=box,
            xform=wp.transform((ball_x + 2.0, ball_y, ball_z - 2.0), wp.quat_identity()),
            hx=1.75,
            hy=1.5,
            hz=0.15,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                contact_margin=0.5,
                density=0.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            gravity=-9.81,
            rigid_contact_margin=0.0,
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_simple_example(cfg: DictConfig):
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
    helhest_simple_example()
