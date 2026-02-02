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
from axion import RenderingConfig
from axion import SimulationConfig
from axion import LoggingConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import (
        create_helhest_model,
        HelhestConfig,
        _load_wheel_mesh,
        _add_chassis,
        _add_wheel,
        _add_fixed_component,
    )
except ImportError:
    from common import (
        create_helhest_model,
        HelhestConfig,
        _load_wheel_mesh,
        _add_chassis,
        _add_wheel,
        _add_fixed_component,
    )

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


def create_helhest_model_position(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
):
    """
    Creates a Helhest robot model configured for POSITION CONTROL.
    """

    wheel_mesh_render = _load_wheel_mesh()

    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, key="base_joint")

    # 2. Wheels
    left_wheel = _add_wheel(
        builder,
        xform,
        "left_wheel",
        HelhestConfig.LEFT_WHEEL_POS,
        0.7,
        wheel_mesh_render,
        is_visible,
    )
    right_wheel = _add_wheel(
        builder,
        xform,
        "right_wheel",
        HelhestConfig.RIGHT_WHEEL_POS,
        0.7,
        wheel_mesh_render,
        is_visible,
    )
    rear_wheel = _add_wheel(
        builder,
        xform,
        "rear_wheel",
        HelhestConfig.REAR_WHEEL_POS,
        0.4,
        wheel_mesh_render,
        is_visible,
    )

    # 3. Wheel Joints
    Y_AXIS = (0.0, 1.0, 0.0)

    # Note: We use TARGET_POSITION mode here!
    # We might need higher gains for position tracking to be stiff enough
    ke_pos = 1000.0
    kd_pos = 10.0

    j_left = builder.add_joint_revolute(
        parent=chassis,
        child=left_wheel,
        parent_xform=wp.transform(HelhestConfig.LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=ke_pos,
        target_kd=kd_pos,
        key="left_wheel_j",
        custom_attributes={
            "joint_dof_mode": [JointMode.TARGET_POSITION],
        },
    )

    j_right = builder.add_joint_revolute(
        parent=chassis,
        child=right_wheel,
        parent_xform=wp.transform(HelhestConfig.RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=ke_pos,
        target_kd=kd_pos,
        key="right_wheel_j",
        custom_attributes={
            "joint_dof_mode": [JointMode.TARGET_POSITION],
        },
    )

    j_rear = builder.add_joint_revolute(
        parent=chassis,
        child=rear_wheel,
        parent_xform=wp.transform(HelhestConfig.REAR_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=ke_pos,
        target_kd=kd_pos,
        key="rear_wheel_j",
        custom_attributes={
            "joint_dof_mode": [JointMode.TARGET_POSITION],
        },
    )

    # 4. Fixed Components
    fixed_joints = []
    for name, params in HelhestConfig.FIXED_COMPONENTS.items():
        j_fixed = _add_fixed_component(builder, chassis, xform, name, params, is_visible)
        fixed_joints.append(j_fixed)

    # 5. Articulation
    builder.add_articulation([j_base, j_left, j_right, j_rear] + fixed_joints, key="helhest")

    return chassis, [left_wheel]


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
    ):
        # 6 Base DOFs + 3 Wheel DOFs
        # Note: Model is built during super().__init__, so we need some vars ready

        # Target position in world space
        self.target_pos = wp.vec3(5.0, 5.0, 0.0)

        self.kp_linear = 2.0
        self.kp_angular = 5.0
        self.max_linear = 8.0
        self.max_angular = 5.0

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
                self.effective_timestep,
                0,  # chassis
                6,  # left
                7,  # right
                8,  # rear
            ],
            device=self.model.device,
        )

        # Apply to control
        # Note: We are writing to joint_target (which is position/angle target)
        # Not joint_target_vel
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
        create_helhest_model_position(
            self.builder, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())
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
    )
    simulator.run()


if __name__ == "__main__":
    helhest_pos_control_example()
