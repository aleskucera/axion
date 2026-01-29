import os
import pathlib
from typing import override

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
        hx = 0.5
        hy = 0.5
        hz = 0.5

        # Create link
        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        # Tilted rail configuration
        # Rotate -30 degrees around Y axis so the local X axis points slightly down
        tilt_angle = -wp.pi / 6.0
        q_tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt_angle)
        
        # Add Prismatic Joint
        # Slide along local X axis
        j0 = self.builder.add_joint_prismatic(
            parent=-1,
            child=link_0,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=q_tilt),
            child_xform=wp.transform_identity(),
            # Optional: limits (currently not enforced by positional constraint kernel directly, 
            # but handled by joint limit solver if active. Visuals might show it sliding through limits 
            # if limits aren't fully hooked up in this specific constraint path yet, 
            # but let's define them anyway)
            limit_lower=-5.0,
            limit_upper=5.0,
        )

        # Create articulation
        self.builder.add_articulation([j0], key="slider")

        self.builder.add_ground_plane()

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def basic_prismatic_example(cfg: DictConfig):
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
    basic_prismatic_example()
