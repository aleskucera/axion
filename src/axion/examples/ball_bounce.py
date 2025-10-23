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
        FRICTION = 0.8
        RESTITUTION = 0.9

        builder = newton.ModelBuilder()

        ball1 = builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), key="ball1"
        )

        builder.add_shape_sphere(
            body=ball1,
            radius=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10.0,
                ke=2000.0,
                kd=10.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
            ),
        )

        builder.add_joint_free(parent=-1, child=ball1)

        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0, kd=10.0, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )

        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig):
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
    ball_bounce_example()
