import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


@wp.kernel
def apply_force_ramp(
    dt: wp.float32,
    time_seconds: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
):
    wp.atomic_add(time_seconds, 0, dt)
    t = time_seconds[0]
    if t < 0.1:
        return

    joint_f[6] = (-1.0) * (t) * 10.0
    joint_f[7] = (t) * 10.0

    wp.printf("Joint f: %f, %f\n", joint_f[6], joint_f[7])


class HelhestRotationSimulator(InteractiveSimulator):
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

        # Helhest DOFs: 6 (Base) + 1 (Left) + 1 (Right) + 1 (Rear) = 9
        # Rotation: Left -4.0, Right 4.0, Rear 0.0
        robot_joint_target = np.array([0.0] * 6 + [-200.0, 200.0, 0.0], dtype=np.float32)

        joint_target = np.tile(robot_joint_target, self.simulation_config.num_worlds)
        self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)
        self.time = wp.zeros(1, dtype=wp.float32)

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        # wp.copy(self.control.joint_f, self.joint_target)
        wp.launch(
            apply_force_ramp,
            dim=1,
            inputs=[self.effective_timestep, self.time],
            outputs=[self.control.joint_f],
            # device=self.device,
        )

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model for the rotation example.
        """

        # Robot position
        robot_x = -1.0
        robot_y = 0.0
        robot_z = 0.4

        create_helhest_model(
            self.builder, xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity())
        )

        # Environment parameters from original rotation.py
        FRICTION = 0.8
        RESTITUTION = 0.0
        KE = 60000.0
        KD = 30000.0
        KF = 500.0

        # Ground plane
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=KE, kd=KD, kf=KF, mu=FRICTION, restitution=RESTITUTION
            )
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_rotation_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = HelhestRotationSimulator(sim_config, render_config, exec_config, engine_config)
    simulator.run()


if __name__ == "__main__":
    helhest_rotation_example()
