import os
from importlib.resources import files

import hydra
import newton
import numpy as np
import openmesh
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
ASSETS_DIR = files("axion").joinpath("examples").joinpath("assets")


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
        FRICTION = 0.8
        RESTITUTION = 0.2

        surface_m = openmesh.read_trimesh(f"{ASSETS_DIR}/surface.obj")
        # mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
        # surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        scale = np.array([3.0, 3.0, 4.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.01])

        surface_mesh = newton.Mesh(mesh_points, mesh_indices)
        self.builder.add_articulation(key="surface")

        ball1 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), key="ball1"
        )

        self.builder.add_shape_sphere(
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

        self.builder.add_joint_free(parent=-1, child=ball1)

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0,
                kd=10.0,
                kf=0.0,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )
        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
            ),
        )

        final_builder = newton.ModelBuilder()
        final_builder.replicate(
            self.builder,
            num_worlds=self.simulation_config.num_worlds,
        )

        model = final_builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig):
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
    ball_bounce_example()
