import math
import os
import pathlib

import hydra
import newton
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class Simulator(InteractiveSimulator):
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

    def build_model(self) -> newton.Model:
        FRICTION = 0.0
        RESTITUTION = 0.0

        ball1 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 3.0), wp.quat_identity()), key="ball"
        )

        self.builder.add_shape_sphere(
            body=ball1,
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10.0,
                ke=6000.0,
                kd=1000.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=6000.0,
                kd=1000.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )

        # 1. Define a rotation (45 degrees around the X-axis)
        # We use a normalized vector (1,0,0) for the axis and radians for the angle.
        rot_angle = math.radians(45.0)
        rotation_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), rot_angle)

        # # 2. Define initial velocity (Throwing it along Y axis and slightly Up on Z)
        # initial_velocity = wp.spatial_vector(0.0, 1.0, 3.0, 3.0, 0.0, 0.0)

        # 3. Create the body with the new name, rotation, and velocity
        box_body = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 1.0), rotation_quat), key="box"
        )

        self.builder.add_shape_box(
            body=box_body,
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10.0,
                ke=6000.0,
                kd=1000.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        # self.builder.body_qd[0] = initial_velocity
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
    )

    simulator.run()


if __name__ == "__main__":
    ball_bounce_example()
