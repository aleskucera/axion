import os
import pathlib
from typing import override

import hydra
import newton
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class Simulator(AbstractSimulator):
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

    @override
    def control_policy(self, state: newton.State):
        wp.copy(self.control.joint_f, wp.array([0.0, 800.0], dtype=wp.float32))

    def build_model(self) -> newton.Model:

        hx = 1.0
        hy = 0.1
        hz = 0.1

        # create first link
        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        link_1 = self.builder.add_link()
        self.builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        # add joints
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            # rotate pendulum around the z-axis to appear sideways to the viewer
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        j1 = self.builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # Create articulation from joints
        self.builder.add_articulation([j0, j1], key="pendulum")

        self.builder.add_ground_plane()

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
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
    basic_pendulum_example()
