import os
import pathlib

import hydra
import newton
import warp as wp
from axion import EngineConfig
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Path to the config directory relative to this file
CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class CurlingSimulator(InteractiveSimulator):
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

        # Target for visualization (matching curling_cylinder2.py)
        self.target_pos = wp.vec3(0.0, 3.0, 0.1)

    def build_model(self) -> newton.Model:
        # Use friction that allows sliding (e.g., 0.05 for ice-like behavior)
        shape_config = newton.ModelBuilder.ShapeConfig(
            ke=1e5, kd=1e2, kf=1e3, mu=0.05, contact_margin=0.3, density=10.0
        )

        # 1. The Stone (Cylinder)
        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_cylinder(body=0, radius=0.3, half_height=0.1, cfg=shape_config)

        # 2. The Ice/Floor
        self.builder.add_ground_plane(cfg=shape_config)

        # Initial velocity (linear_y = 2.0)
        # Spatial vector is (linear, angular) for Newton: (vx, vy, vz, wx, wy, wz)
        self.builder.body_qd[0] = wp.spatial_vector(0.0, 1.733, 0.0, 0.0, 0.0, 0.0)
        # self.builder.body_qd[0] = wp.spatial_vector(0.0, 1.92, 0.0, 0.0, 0.0, 0.0)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
        )

    def _render(self, segment_num: int):
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)

        # Draw Target Marker (Red thin Cylinder)
        self.viewer.log_shapes(
            "/target",
            newton.GeoType.CYLINDER,
            (0.3, 0.01),  # radius, half_height
            wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
            wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),  # Red
        )

        self.viewer.end_frame()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = CurlingSimulator(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
    )
    sim.run()


if __name__ == "__main__":
    main()
