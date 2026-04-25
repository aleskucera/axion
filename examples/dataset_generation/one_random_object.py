import os
import pathlib

import hydra
import newton
import warp as wp
from axion import DatasetSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.generation.scene_generator_new import SceneGenerator
from axion.simulation.dataset_simulator import random_velocities_kernel
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class OneObjectSimulator(DatasetSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        self.pos_min = wp.vec3(-1.0, -1.0, 0.5)
        self.pos_max = wp.vec3(1.0, 1.0, 3.0)
        self.lin_vel_min, self.lin_vel_max = -3.0, 3.0
        self.ang_vel_min, self.ang_vel_max = -3.14, 3.14
        self.joint_target_lower_bound = 0.0
        self.joint_target_upper_bound = 0.0
        self.seed = int(logging_config.dataset_log_file.split("/")[-1].split("_")[1].split(".")[0])

        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2
        self.builder.add_ground_plane()

        gen = SceneGenerator(self.builder, seed=self.seed)
        gen.generate_random_object(
            pos_bounds=((-1, -1, 0.5), (1, 1, 3)),
            density_bounds=(10.0, 100.0),
            size_bounds=(0.1, 0.3),
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    def _resolve_constraints(self):
        super()._resolve_constraints()
        # Apply random initial velocities after constraint resolution zeroes them out
        wp.launch(
            kernel=random_velocities_kernel,
            dim=self.model.body_count,
            inputs=[
                self.current_state.body_qd,
                self.lin_vel_min,
                self.lin_vel_max,
                self.ang_vel_min,
                self.ang_vel_max,
                self.seed + 1,
            ],
            device=self.model.device,
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def one_random_object(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = OneObjectSimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    one_random_object()
