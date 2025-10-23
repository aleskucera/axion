from importlib.resources import files
from typing import override

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
        builder = newton.ModelBuilder()

        rigid_cfg = builder.ShapeConfig()
        rigid_cfg.restitution = 0.2
        rigid_cfg.has_shape_collision = True
        rigid_cfg.mu = 1.0

        # add ground plane
        builder.add_ground_plane(cfg=rigid_cfg)

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = builder.add_body(
            xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), key="sphere"
        )
        builder.add_shape_sphere(body_sphere, radius=0.5, cfg=rigid_cfg)

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 0.0, drop_z)
        body_capsule = builder.add_body(
            xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), key="capsule"
        )
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7, cfg=rigid_cfg)

        # BOX
        self.box_pos = wp.vec3(0.0, 2.0, drop_z)
        body_box = builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), key="box"
        )
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25, cfg=rigid_cfg)

        builder.add_joint_free(parent=-1, child=body_sphere)
        builder.add_joint_free(parent=-1, child=body_capsule)
        builder.add_joint_free(parent=-1, child=body_box)

        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_example(cfg: DictConfig):
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
    helhest_example()
