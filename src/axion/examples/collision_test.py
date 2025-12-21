import os
from importlib.resources import files

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

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")


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

    def build_model(self) -> newton.Model:
        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -0.9, drop_z)
        body_sphere = self.builder.add_body(
            xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), key="sphere"
        )
        self.builder.add_shape_sphere(
            body_sphere,
            radius=0.5,
            cfg=self.builder.ShapeConfig(
                thickness=0.0,
                contact_margin=0.7,
                mu=0.0,
            ),
        )

        # BOX
        self.box_pos = wp.vec3(0.0, 0.0, drop_z)
        body_box = self.builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), key="box"
        )
        self.builder.add_shape_box(
            body_box,
            hx=0.5,
            hy=0.35,
            hz=0.25,
            cfg=self.builder.ShapeConfig(
                thickness=0.0,
                contact_margin=0.0,
                mu=0.0,
                density=1e6,
            ),
        )

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 0.7, drop_z)
        body_capsule = self.builder.add_body(
            xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), key="capsule"
        )
        self.builder.add_shape_capsule(
            body_capsule,
            radius=0.3,
            half_height=0.1,
            cfg=self.builder.ShapeConfig(
                thickness=0.0,
                contact_margin=0.1,
                mu=0.0,
            ),
        )

        self.builder.rigid_contact_margin = 0.00
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            gravity=0.0,
            rigid_contact_margin=0.00,
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def basic_shape_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    basic_shape_example()
