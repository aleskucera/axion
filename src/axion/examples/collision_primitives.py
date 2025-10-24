from importlib.resources import files

import hydra
import newton
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> newton.Model:
        FRICTION = 0.7
        RESTITUTION = 0.5
        DENSITY = 10.0
        KE = 200.0
        KD = 50.0
        KF = 200.0

        builder = newton.ModelBuilder()

        ball1 = builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), key="ball1"
        )
        builder.add_shape_sphere(
            body=ball1,
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        ball2 = builder.add_body(
            xform=wp.transform((0.0, 0.3, 4.5), wp.quat_identity()), key="ball2"
        )

        builder.add_shape_sphere(
            body=ball2,
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        ball3 = builder.add_body(
            xform=wp.transform((0.0, -0.6, 6.5), wp.quat_identity()), key="ball3"
        )

        builder.add_shape_sphere(
            body=ball3,
            radius=0.4,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        ball4 = builder.add_body(
            xform=wp.transform((0.0, -0.6, 10.5), wp.quat_identity()), key="ball4"
        )

        builder.add_shape_sphere(
            body=ball4,
            radius=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        box1 = builder.add_body(xform=wp.transform((0.0, 0.0, 9.0), wp.quat_identity()), key="box1")

        builder.add_shape_box(
            body=box1,
            hx=0.4,
            hy=0.4,
            hz=0.4,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        builder.add_joint_free(parent=-1, child=ball1)
        builder.add_joint_free(parent=-1, child=ball2)
        builder.add_joint_free(parent=-1, child=ball3)
        builder.add_joint_free(parent=-1, child=ball4)
        builder.add_joint_free(parent=-1, child=box1)

        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )
        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def collision_primitives_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    profile_config: ProfilingConfig = hydra.utils.instantiate(cfg.profiling)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    simulator.run()


if __name__ == "__main__":
    collision_primitives_example()
