import math
import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import EngineConfig
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import JointMode
from axion import LoggingConfig
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
def compute_wheel_angles_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_target_pos: wp.array(dtype=wp.float32),
    current_wheel_angles: wp.array(dtype=wp.float32),
    target_pos: wp.vec3,
    kp_linear: float,
    kp_angular: float,
    max_linear_speed: float,
    max_angular_speed: float,
    dt: float,
    chassis_idx: int,
    left_wheel_idx: int,
    right_wheel_idx: int,
    rear_wheel_idx: int,
):
    # Get chassis transform
    xform = body_q[chassis_idx]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    # Vector to target
    diff = target_pos - pos
    diff_2d = wp.vec3(diff[0], diff[1], 0.0)
    dist = wp.length(diff_2d)

    # Robot forward vector
    forward = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
    forward_2d = wp.normalize(wp.vec3(forward[0], forward[1], 0.0))

    # Calculate heading error
    cross_z = forward_2d[0] * diff_2d[1] - forward_2d[1] * diff_2d[0]
    dot = wp.dot(forward_2d, wp.normalize(diff_2d))

    # Control logic (Velocity Output)
    v_linear = 0.0
    v_angular = 0.0

    if dist > 0.1:
        v_linear = dist * kp_linear
        v_angular = cross_z * kp_angular

        # Cap speeds
        if v_linear > max_linear_speed:
            v_linear = max_linear_speed
        if v_angular > max_angular_speed:
            v_angular = max_angular_speed
        if v_angular < -max_angular_speed:
            v_angular = -max_angular_speed

        if dot < 0.0:
            v_linear *= 0.1
            v_angular = (
                wp.sign(cross_z) * max_angular_speed if cross_z != 0.0 else max_angular_speed
            )

    # Differential drive kinematics
    track_width = 0.72
    v_l = v_linear - v_angular * (track_width / 2.0)
    v_r = v_linear + v_angular * (track_width / 2.0)
    v_rear = (v_l + v_r) / 2.0

    # Integrate to get position
    # We update the 'current_wheel_angles' state
    # Indices in the angle state array: 0=Left, 1=Right, 2=Rear

    new_angle_l = current_wheel_angles[0] + v_l * dt
    new_angle_r = current_wheel_angles[1] + v_r * dt
    new_angle_rear = current_wheel_angles[2] + v_rear * dt

    current_wheel_angles[0] = new_angle_l
    current_wheel_angles[1] = new_angle_r
    current_wheel_angles[2] = new_angle_rear

    # Set targets in global joint array
    joint_target_pos[left_wheel_idx] = new_angle_l
    joint_target_pos[right_wheel_idx] = new_angle_r
    joint_target_pos[rear_wheel_idx] = new_angle_rear


class HelhestPositionControlSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "position",
        k_p: float = 50.0,
        k_d: float = 0.1,
        friction: float = 0.7,
    ):
        # 6 Base DOFs + 3 Wheel DOFs
        # Note: Model is built during super().__init__, so we need some vars ready

        # Target position in world space
        self.target_pos = wp.vec3(5.0, 5.0, 0.0)

        self.kp_linear = 2.0
        self.kp_angular = 5.0
        self.max_linear = 8.0
        self.max_angular = 5.0

        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction

        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        self.joint_target = wp.zeros(9, dtype=wp.float32, device=self.model.device)

        # Internal state for wheel angles (since we need to integrate them)
        # 0: Left, 1: Right, 2: Rear
        self.wheel_angles = wp.zeros(3, dtype=wp.float32, device=self.model.device)

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
        # Update targets
        wp.launch(
            kernel=compute_wheel_angles_kernel,
            dim=1,
            inputs=[
                current_state.body_q,
                self.joint_target,
                self.wheel_angles,
                self.target_pos,
                self.kp_linear,
                self.kp_angular,
                self.max_linear,
                self.max_angular,
                self.clock.dt,
                0,  # chassis
                6,  # left
                7,  # right
                8,  # rear
            ],
            device=self.model.device,
        )

        # Apply to control
        if self.control_mode == "velocity":
            # For velocity mode, the navigation kernel would need to output velocities
            # But here it outputs integrated angles. To support velocity mode properly
            # we would need to modify the kernel.
            # For now, let's just copy to joint_target_pos as it was intended for position control.
            wp.copy(self.control.joint_target_pos, self.joint_target)
        else:
            wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        # Ground
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Marker
        self.builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(self.target_pos, wp.quat_identity()),
            radius=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=True, collision_group=-1),
        )

        # Robot
        create_helhest_model(
            self.builder,
            xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_pos_control_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestPositionControlSimulator(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_pos_control_example()
