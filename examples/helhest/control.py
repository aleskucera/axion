import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import AbstractSimulator
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
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class HelhestControlSimulator(AbstractSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config):
        self.left_indices_cpu = []
        self.right_indices_cpu = []
        super().__init__(sim_config, render_config, exec_config, engine_config)

        # 6 Base DOFs + 3 Wheel DOFs (Left, Right, Rear)
        # We only control the last 3.
        self.target_velocities = wp.zeros(3, dtype=wp.float32, device=self.model.device)

        # Initialize full target array
        # First 6 are for the free joint (base), ignored by control usually but good to keep 0
        self.joint_target = wp.zeros(9, dtype=wp.float32, device=self.model.device)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Check keyboard input and update wheel velocities."""
        base_speed = 5.0
        turn_speed = 2.5

        left_v = 0.0
        right_v = 0.0

        # Simple WASD/Arrow style logic
        # Forward/Backward
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):  # Forward
                left_v += base_speed
                right_v += base_speed
            if self.viewer.is_key_down("k"):  # Backward
                left_v -= base_speed
                right_v -= base_speed

            # Turn Left/Right
            if self.viewer.is_key_down("j"):  # Left
                left_v -= turn_speed
                right_v += turn_speed
            if self.viewer.is_key_down("l"):  # Right
                left_v += turn_speed
                right_v -= turn_speed

        # Rear wheel logic: Average of left and right?
        # Or maybe just let it spin freely if we could (but we are in velocity mode).
        # Let's try driving it with the average linear speed.
        rear_v = (left_v + right_v) / 2.0

        # Update targets
        # Indices in Helhest model (from common.py):
        # 0-5: Base (Free)
        # 6: Left Wheel
        # 7: Right Wheel
        # 8: Rear Wheel

        # We need to update the last 3 elements of self.joint_target
        # We can't slice assign easily in Warp from python like numpy,
        # so we'll build a small numpy array and copy.

        # Current logic: The control kernel reads the whole array.
        # We can construct the array on CPU then copy.

        targets_cpu = np.zeros(9, dtype=np.float32)
        # Leave 0-5 as 0.0
        targets_cpu[6] = left_v
        targets_cpu[7] = right_v
        targets_cpu[8] = rear_v

        wp.copy(
            self.joint_target, wp.array(targets_cpu, dtype=wp.float32, device=self.model.device)
        )

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
        wp.copy(self.control.joint_target, self.joint_target)

    def build_model(self) -> newton.Model:
        # --- 1. Ground ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Obstacle 1: Stairs (Stepped boxes)
        num_steps = 6
        step_depth = 0.6
        step_height = 0.08  # Smaller steps for Helhest
        step_width = 4.0
        start_x = 5.0
        start_y = -4.0

        for i in range(num_steps):
            h_curr = (i + 1) * step_height
            z_curr = h_curr / 2.0
            x_curr = start_x + i * step_depth

            self.builder.add_shape_box(
                -1,
                wp.transform(wp.vec3(x_curr, start_y, z_curr), wp.quat_identity()),
                hx=step_depth / 2.0,
                hy=step_width / 2.0,
                hz=h_curr / 2.0,
                cfg=ground_cfg,
            )

        # Obstacle 2: Ramp
        ramp_length = 5.0
        ramp_width = 4.0
        ramp_height = 1.0
        ramp_angle = float(np.arctan2(ramp_height, ramp_length))

        ramp_x = 7.0
        ramp_y = 4.0
        ramp_z = ramp_height / 2.0 - 0.05

        q_ramp = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -ramp_angle)

        self.builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(ramp_x, ramp_y, ramp_z), q_ramp),
            hx=ramp_length / 2.0,
            hy=ramp_width / 2.0,
            hz=0.05,
            cfg=ground_cfg,
        )

        # Obstacle 3: Uneven terrain (Small boulders)
        import random

        random.seed(42)
        for _ in range(30):
            rx = random.uniform(3.0, 12.0)
            ry = random.uniform(-2.0, 2.0)
            rz = 0.05
            self.builder.add_shape_box(
                -1,
                wp.transform(
                    wp.vec3(rx, ry, rz),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random.uniform(0, 3.14)),
                ),
                hx=random.uniform(0.15, 0.5),
                hy=random.uniform(0.15, 0.5),
                hz=random.uniform(0.05, 0.15),
                cfg=ground_cfg,
            )

        # --- 2. Robot ---
        robot_x = 0.0
        robot_y = 0.0
        robot_z = 0.5  # Slightly above ground

        create_helhest_model(
            self.builder, xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity())
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_control_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = HelhestControlSimulator(sim_config, render_config, exec_config, engine_config)
    simulator.run()


if __name__ == "__main__":
    helhest_control_example()
