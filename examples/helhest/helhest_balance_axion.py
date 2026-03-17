import math
import os
import pathlib
import time

import hydra
import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from newton import Model
from omegaconf import DictConfig

try:
    from examples.helhest.common import HelhestConfig
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import HelhestConfig
    from common import create_helhest_model

from axion.optim.trajectory_adam import TrajectoryAdam

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# Tuning parameters for balancing
BALANCE_PITCH = math.pi / 2.0  # ~45 degrees back. Tune this so COM is over the wheels!


def bourke_color_map(v_min, v_max, v):
    c = wp.vec3(1.0, 1.0, 1.0)
    v = np.clip(v, v_min, v_max)
    dv = v_max - v_min

    if v < (v_min + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - v_min) / dv
    elif v < (v_min + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (v_min + 0.25 * dv - v) / dv
    elif v < (v_min + 0.75 * dv):
        c[0] = 4.0 * (v - v_min - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (v_min + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c


@wp.kernel
def balance_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    target_pos: wp.vec3,
    weight_rot: float,
    weight_pos: float,
    loss: wp.array(dtype=wp.float32),
):
    """Dense loss over all timesteps to keep the robot upright and in place."""
    sim_step = wp.tid()

    # Body 0 is the chassis
    xform = body_pose[sim_step, 0, 0]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    # 1. Positional Drift Loss (Only care about X and Y, ignore Z bounces)
    pos_err = wp.vec3(pos[0] - target_pos[0], pos[1] - target_pos[1], 0.0)
    p_loss = wp.dot(pos_err, pos_err)

    # 2. Orientation Loss (Compare local 'UP' vectors)
    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)

    up_err = current_up - target_up
    r_loss = wp.dot(up_err, up_err)

    # Accumulate loss for this specific timestep
    step_loss = (weight_pos * p_loss) + (weight_rot * r_loss)
    wp.atomic_add(loss, 0, step_loss)


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, 0, dof_idx] - target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


class HelhestEndpointOptimizer(AxionDifferentiableSimulator):
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

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Loss weights
        self.weight_rot = 200.0  # High penalty for falling over
        self.weight_pos = 10.0  # Moderate penalty for drifting away
        self.smoothness_weight = 1e-1
        self.regularization_weight = 1e-4  # Penalize excessive wheel speeds

        self.frame = 0

        # Start with completely stationary wheels so it falls over on Iter 0
        self.init_wheel_vel = (0.0, 0.0, 0.0)

        # Track chassis (body 0)
        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.7,
            ke=50.0,
            kd=50.0,
            kf=50.0,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Spawn tilted back on two wheels!
        initial_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), initial_rot),
            control_mode="velocity",
            k_p=HelhestConfig.TARGET_KE,
            k_d=HelhestConfig.TARGET_KD,
            friction_left_right=0.7,
            friction_rear=0.35,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        # We want it to stay right where it started, at the initial pitch angle
        target_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        target_pos = wp.vec3(0.0, 0.0, 0.0)

        wp.launch(
            kernel=balance_loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[
                self.trajectory.body_pose,
                target_rot,
                target_pos,
                self.weight_rot,
                self.weight_pos,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=smoothness_kernel,
            dim=(self.clock.total_sim_steps - 1, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.smoothness_weight,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # Apply Adam step directly
        self.optimizer.step(self.trajectory.joint_target_vel.grad)
        self.trajectory.joint_target_vel.grad.zero_()

        # Sync updated trajectory values back into per-step controls
        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 50.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=2.0,
        )

        self.frame += 1

    def _make_wheel_vel_array(self, vels: tuple) -> np.ndarray:
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        arr = np.zeros((self.clock.total_sim_steps, 1, num_dofs), dtype=np.float32)
        arr[:, :, WHEEL_DOF_OFFSET + 0] = vels[0]  # left
        arr[:, :, WHEEL_DOF_OFFSET + 1] = vels[1]  # right
        arr[:, :, WHEEL_DOF_OFFSET + 2] = vels[2]  # rear
        return arr

    def train(self, iterations=60):
        # Evaluate forward kinematics to place bodies at their rest pose
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        # Initialize trajectory buffer and per-step controls with zeroes
        init_arr = self._make_wheel_vel_array(self.init_wheel_vel)
        wp.copy(self.trajectory.joint_target_vel, wp.array(init_arr, dtype=wp.float32))

        init_ctrl = np.zeros(num_dofs, dtype=np.float32)
        init_ctrl_wp = wp.array(init_ctrl, dtype=wp.float32, device=self.model.device)

        self.optimizer = TrajectoryAdam(
            params=self.trajectory.joint_target_vel,
            lr=1.0,
            betas=(0.9, 0.999),
            dof_offset=WHEEL_DOF_OFFSET,
            num_dofs=NUM_WHEEL_DOFS,
        )

        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, init_ctrl_wp)

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            current_vel_l = self.trajectory.joint_target_vel.numpy()[0, 0, WHEEL_DOF_OFFSET]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | left_vel[0]={current_vel_l:.3f} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="helhest_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestEndpointOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=100)


if __name__ == "__main__":
    main()
