"""
Test: Wheeled robot driving uphill on a heightfield terrain.

Creates a simple ramp heightfield and places the Taros-4 robot at the bottom.
The robot drives forward up the slope.
"""
import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
from axion import EngineConfig
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

from axion.core.types import JointMode

# 4-wheel robot config
CHASSIS_HX = 0.6
CHASSIS_HY = 0.4
CHASSIS_HZ = 0.08
CHASSIS_MASS = 100.0
WHEEL_RADIUS = 0.22
WHEEL_MASS = 10.0
NUM_WHEELS = 4


def create_4wheel_robot(
    builder,
    xform=wp.transform_identity(),
    is_visible=True,
    k_p=200.0,
    k_d=0.0,
    wheel_mu=0.3,
):
    """Simple symmetric 4-wheel robot with sphere wheels."""
    # Chassis
    chassis = builder.add_link(xform=xform, label="chassis")
    chassis_vol = CHASSIS_HX * CHASSIS_HY * CHASSIS_HZ * 8.0
    builder.add_shape_box(
        body=chassis,
        xform=wp.transform_identity(),
        hx=CHASSIS_HX,
        hy=CHASSIS_HY,
        hz=CHASSIS_HZ,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=CHASSIS_MASS / chassis_vol,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    # 4 wheels at corners — dropped below chassis
    wheel_z = -CHASSIS_HZ - WHEEL_RADIUS * 0.3  # wheels hang below chassis
    wheel_positions = {
        "front_left": wp.vec3(CHASSIS_HX, CHASSIS_HY, wheel_z),
        "front_right": wp.vec3(CHASSIS_HX, -CHASSIS_HY, wheel_z),
        "rear_left": wp.vec3(-CHASSIS_HX, CHASSIS_HY, wheel_z),
        "rear_right": wp.vec3(-CHASSIS_HX, -CHASSIS_HY, wheel_z),
    }

    joints = [j_base]
    for name, pos_local in wheel_positions.items():
        pos_world = wp.transform_point(xform, pos_local)
        I_spin = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2
        I_other = (1.0 / 12.0) * WHEEL_MASS * (3.0 * WHEEL_RADIUS**2 + 0.1**2)
        wheel_I = wp.mat33(I_other, 0.0, 0.0, 0.0, I_spin, 0.0, 0.0, 0.0, I_other)

        is_rear = "rear" in name
        wheel_link = builder.add_link(
            xform=wp.transform(pos_world, xform.q),
            label=name,
            mass=WHEEL_MASS,
            inertia=wheel_I,
        )
        # Capsule collision, rotated so the axis aligns with the wheel spin axis (y)
        wheel_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        builder.add_shape_capsule(
            body=wheel_link,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wheel_rot),
            radius=WHEEL_RADIUS,
            half_height=0.06,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                is_visible=is_visible,
                collision_group=-1,
                mu=wheel_mu,
            ),
        )
        # Rear wheels: motorized. Front wheels: free-rolling (ke=0)
        j = builder.add_joint_revolute(
            parent=chassis,
            child=wheel_link,
            parent_xform=wp.transform(pos_local, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=(0.0, 1.0, 0.0),
            target_ke=k_p if is_rear else 0.0,
            target_kd=k_d,
            label=f"{name}_j",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
        )
        joints.append(j)

    builder.add_articulation(joints, label="robot")
    return chassis


# Terrain parameters
SLOPE_ANGLE = 15.0  # degrees
SLIP_WIDTH = 1.5  # slippery strip width (meters)


def create_flat_heightfield(hx, hy, z_height, resolution=10):
    """Create a flat heightfield at a given z height."""
    nrow, ncol = resolution, resolution
    data = np.ones((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(
        data=data,
        nrow=nrow,
        ncol=ncol,
        hx=hx,
        hy=hy,
        min_z=z_height,
        max_z=z_height,
    )
    return hfield


def create_ramp_heightfield(hx, hy, z_start, z_end, resolution=20):
    """Create a heightfield that ramps from z_start (left) to z_end (right)."""
    nrow, ncol = resolution, resolution
    x_norm = np.linspace(0.0, 1.0, ncol, dtype=np.float32)
    data = np.tile(x_norm, (nrow, 1))
    hfield = newton.Heightfield(
        data=data,
        nrow=nrow,
        ncol=ncol,
        hx=hx,
        hy=hy,
        min_z=z_start,
        max_z=z_end,
    )
    return hfield


@wp.kernel
def update_wheel_friction_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_material_mu: wp.array(dtype=wp.float32),
    wheel_shape_indices: wp.array(dtype=wp.int32),
    slip_y_start: float,
    slip_y_end: float,
    mu_normal: float,
    mu_slippery: float,
):
    """Set wheel friction based on chassis y-position."""
    chassis_y = wp.transform_get_translation(body_q[0])[1]
    mu = mu_normal
    if chassis_y > slip_y_start and chassis_y < slip_y_end:
        mu = mu_slippery
    for i in range(wheel_shape_indices.shape[0]):
        shape_material_mu[wheel_shape_indices[i]] = mu


@wp.kernel
def drive_kernel(
    joint_target_vel: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    step_counter: wp.array(dtype=wp.int32),
    settle_steps: int,
    wheel_speed: float,
    dof_offset: int,
    steering_gain: float,
):
    """Drive with heading correction. Steers left/right wheels to stay on x=0."""
    current_step = step_counter[0]
    if current_step >= settle_steps:
        # Chassis is body 0 — read its x position for lateral correction
        chassis_pos = wp.transform_get_translation(body_q[0])
        x_error = chassis_pos[0]  # want x=0

        # Also correct heading: read chassis orientation
        chassis_rot = wp.transform_get_rotation(body_q[0])
        # Forward direction in world frame (chassis local +x rotated)
        fwd = wp.quat_rotate(chassis_rot, wp.vec3(1.0, 0.0, 0.0))
        # Heading error: forward should point along +y, so fwd.x should be 0
        heading_error = fwd[0]

        correction = steering_gain * x_error + steering_gain * 2.0 * heading_error

        # Front wheels (DOF 0, 1): free-rolling (no motor)
        # Rear wheels (DOF 2, 3): driven with differential steering
        joint_target_vel[dof_offset + 0] = 0.0
        joint_target_vel[dof_offset + 1] = 0.0
        joint_target_vel[dof_offset + 2] = wheel_speed + correction
        joint_target_vel[dof_offset + 3] = wheel_speed - correction

    wp.atomic_add(step_counter, 0, 1)


class UphillDrive(InteractiveSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self._step_counter = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._settle_steps = int(1.0 / self.clock.dt)

        # Find wheel shape indices (spheres attached to wheel bodies)
        # Bodies: 0=chassis, 1-4=wheels
        wheel_shape_ids = []
        shape_body = self.model.shape_body.numpy()
        for si in range(self.model.shape_count):
            if shape_body[si] in [1, 2, 3, 4]:  # wheel bodies
                wheel_shape_ids.append(si)
        self._wheel_shape_indices = wp.array(
            wheel_shape_ids, dtype=wp.int32, device=self.model.device
        )

        # Position camera to see the hill from the side
        if self.viewer:
            self.viewer.set_camera(
                pos=wp.vec3(-22.0, -14.0, 10.0),
                pitch=-20.0,
                yaw=30.0,
            )

    def build_model(self):
        self.builder.rigid_gap = 0.2

        # Layout along x-axis:
        #   [-20, -10]: flat start (z=0)
        #   [-10,   0]: uphill ramp (z=0 -> z_top)
        #   [  0,  10]: slippery middle (ramp continues, low friction)
        #   [ 10,  20]: uphill ramp top (z continues -> z_peak)
        #   [ 20,  30]: flat top (z=z_peak)

        slope_rad = np.radians(SLOPE_ANGLE)
        hy = 3.0
        z_per_m = np.tan(slope_rad)
        ncol, nrow = 30, 12

        # Part 1: flat(3m) → ramp(5m)  total=8m
        # Part 2: slippery ramp        total=SLIP_WIDTH
        # Part 3: ramp(5m) → flat(10m) total=15m
        p1_len, p2_len, p3_len = 8.0, SLIP_WIDTH, 15.0
        flat_len, ramp1_len, ramp2_len = 3.0, 5.0, 5.0

        z_ramp1_start = 0.0
        z_ramp1_end = z_ramp1_start + z_per_m * ramp1_len
        z_slip_end = z_ramp1_end + z_per_m * p2_len
        z_ramp2_end = z_slip_end + z_per_m * ramp2_len

        normal_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=1.0)
        slippery_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=0.01)
        rot_z90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)

        def make_heightfield(length, z_profile_fn):
            x = np.linspace(0.0, length, ncol)
            z = np.array([z_profile_fn(xi) for xi in x], dtype=np.float32)
            data = np.tile(z, (nrow, 1))
            return newton.Heightfield(
                data=data,
                nrow=nrow,
                ncol=ncol,
                hx=length / 2.0,
                hy=hy,
                min_z=float(z.min()),
                max_z=max(float(z.max()), float(z.min()) + 0.001),
            )

        # Part 1: flat then ramp up (corner is inside this heightfield)
        def p1_profile(x):
            if x < flat_len:
                return z_ramp1_start
            return z_ramp1_start + z_per_m * (x - flat_len)

        hf1 = make_heightfield(p1_len, p1_profile)
        y1 = p1_len / 2.0
        self.builder.add_shape_heightfield(
            heightfield=hf1,
            xform=wp.transform(wp.vec3(0.0, y1, 0.0), rot_z90),
            cfg=normal_cfg,
            label="terrain_bottom",
        )

        # Part 2: slippery ramp (just a straight ramp, seams are smooth)
        hf2 = make_heightfield(p2_len, lambda x: z_ramp1_end + z_per_m * x)
        y2 = p1_len + p2_len / 2.0
        self.builder.add_shape_heightfield(
            heightfield=hf2,
            xform=wp.transform(wp.vec3(0.0, y2, 0.0), rot_z90),
            cfg=slippery_cfg,
            label="terrain_slippery",
        )

        # Part 3: ramp up then flat (corner is inside this heightfield)
        def p3_profile(x):
            if x < ramp2_len:
                return z_slip_end + z_per_m * x
            return z_ramp2_end

        hf3 = make_heightfield(p3_len, p3_profile)
        y3 = p1_len + p2_len + p3_len / 2.0
        self.builder.add_shape_heightfield(
            heightfield=hf3,
            xform=wp.transform(wp.vec3(0.0, y3, 0.0), rot_z90),
            cfg=normal_cfg,
            label="terrain_top",
        )

        self._slip_y_start = p1_len
        self._slip_y_end = p1_len + p2_len
        self._z_flat_start = 0.0

        # Robot on the flat start section, facing +y (rotated 90 deg)
        robot_xform = wp.transform(
            wp.vec3(0.0, 1.5, WHEEL_RADIUS + 0.2),
            rot_z90,
        )

        create_4wheel_robot(
            self.builder,
            xform=robot_xform,
            is_visible=True,
            k_p=500.0,
            k_d=0.0,
            wheel_mu=1.0,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            gravity=-9.81,
        )

    def control_policy(self, current_state):
        # Update wheel friction based on position
        wp.launch(
            kernel=update_wheel_friction_kernel,
            dim=1,
            inputs=[
                current_state.body_q,
                self.model.shape_material_mu,
                self._wheel_shape_indices,
                self._slip_y_start,
                self._slip_y_end,
                1.0,  # mu_normal
                0.01,  # mu_slippery
            ],
            device=self.model.device,
        )

        # Drive with heading correction
        wp.launch(
            kernel=drive_kernel,
            dim=1,
            inputs=[
                self.control.joint_target_vel,
                current_state.body_q,
                self._step_counter,
                self._settle_steps,
                20.0,  # wheel_speed
                6,  # dof_offset (skip free joint 6 DOFs)
                0.0,  # steering_gain
            ],
            device=self.model.device,
        )


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config_uphill")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = UphillDrive(sim_config, render_config, exec_config, engine_config, logging_config)
    sim.run()


if __name__ == "__main__":
    main()
