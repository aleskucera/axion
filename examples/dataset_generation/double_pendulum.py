import math
import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
from axion import DatasetSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.dataset_simulator import random_velocities_kernel
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

# Fixed pendulum geometry and physical parameters
ANCHOR_POS = wp.vec3(0.0, 0.0, 5.0)
LINK_RADIUS = 0.03
LINK_HALF_HEIGHT = 0.4
LINK_DENSITY = 500.0  # kg/m³


@wp.kernel
def random_joint_velocities_kernel(
    joint_qd: wp.array(dtype=float),
    ang_vel_min: float,
    ang_vel_max: float,
    seed: int,
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    joint_qd[tid] = wp.randf(state) * (ang_vel_max - ang_vel_min) + ang_vel_min


class PendulumSimulator(DatasetSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        # pos_min/pos_max only affect free-joint randomization — unused here
        self.pos_min = wp.vec3(-1.0, -1.0, 0.0)
        self.pos_max = wp.vec3(1.0, 1.0, 10.0)
        # No random linear velocity: pendulum pivot is fixed
        self.lin_vel_min, self.lin_vel_max = 0.0, 0.0
        self.ang_vel_min, self.ang_vel_max = -5.0, 5.0
        self.joint_target_lower_bound = 0.0
        self.joint_target_upper_bound = 0.0
        self.seed = 70

        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:
        cfg = self.builder.ShapeConfig(density=LINK_DENSITY)

        # Random horizontal axis — determines the plane in which the pendulum swings
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        axis = wp.vec3(float(np.cos(theta)), float(np.sin(theta)), 0.0)

        # --- Link 1 ---
        link1_pos = ANCHOR_POS - wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT)
        link1 = self.builder.add_link(xform=wp.transform(link1_pos, wp.quat_identity()))
        self.builder.add_shape_capsule(
            link1, radius=LINK_RADIUS, half_height=LINK_HALF_HEIGHT, cfg=cfg
        )

        # Revolute joint: world → top of link1
        j1 = self.builder.add_joint_revolute(
            parent=-1,
            child=link1,
            parent_xform=wp.transform(ANCHOR_POS, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT), wp.quat_identity()),
            axis=axis,
            limit_lower=-math.pi,
            limit_upper=math.pi,
        )

        # --- Link 2 ---
        link2_pos = link1_pos - wp.vec3(0.0, 0.0, 2.0 * LINK_HALF_HEIGHT)
        link2 = self.builder.add_link(xform=wp.transform(link2_pos, wp.quat_identity()))
        self.builder.add_shape_capsule(
            link2, radius=LINK_RADIUS, half_height=LINK_HALF_HEIGHT, cfg=cfg
        )

        # Revolute joint: bottom of link1 → top of link2
        j2 = self.builder.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF_HEIGHT), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT), wp.quat_identity()),
            axis=axis,
            limit_lower=-math.pi,
            limit_upper=math.pi,
        )

        self.builder.add_articulation([j1, j2])

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    def _resolve_constraints(self):
        super()._resolve_constraints()
        # Set random joint velocities and update cartesian state via FK
        wp.launch(
            kernel=random_joint_velocities_kernel,
            dim=self.model.joint_dof_count,
            inputs=[
                self.current_state.joint_qd,
                self.ang_vel_min,
                self.ang_vel_max,
                self.seed + 1,
            ],
            device=self.model.device,
        )
        newton.eval_fk(
            self.model, self.current_state.joint_q, self.current_state.joint_qd, self.current_state
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="gnn", version_base=None)
def pendulum_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = PendulumSimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    pendulum_example()
