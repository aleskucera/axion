"""
Uphill drive optimization via differentiable simulation.

Optimizes a single wheel speed spline so the 4-wheel robot drives up
a hill with a slippery middle section.

Loss: -sum_t chassis_z(t) (maximize total height over trajectory)
"""
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.engine_config import AxionEngineConfig
from axion.core.types import JointMode
from newton import Model

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Import robot config and terrain builder from uphill_drive
from examples.differentiable.uphill_drive import (
    build_mesh_terrain,
    create_4wheel_robot,
    CHASSIS_HX,
    CHASSIS_HY,
    CHASSIS_HZ,
    CHASSIS_MASS,
    WHEEL_MASS,
    WHEEL_RADIUS,
    NUM_WHEELS,
    SLIP_WIDTH,
)

SLOPE_ANGLE = 20.0
WHEEL_DOF_OFFSET = 6  # free joint takes DOFs 0-5
# Only rear wheels are driven (DOFs 8, 9), front are free (DOFs 6, 7)
REAR_DOF_OFFSET = 8
NUM_DRIVEN = 2


# ─── Spline utilities ──────────────────────────────────────────────────────


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    col_sums = W.sum(axis=0)
    return W, col_sums


class SplineAdam:
    def __init__(self, K, lr, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros(K, dtype=np.float64)
        self.v = np.zeros(K, dtype=np.float64)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─── Kernels ───────────────────────────────────────────────────────────────


@wp.kernel
def target_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance from target at every timestep."""
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def smoothness_loss_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    dof_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize changes in speed between consecutive timesteps."""
    t = wp.tid()
    v_curr = target_vel[t, 0, dof_idx]
    v_next = target_vel[t + 1, 0, dof_idx]
    diff = v_next - v_curr
    wp.atomic_add(loss, 0, weight * diff * diff)


@wp.kernel
def energy_loss_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    dof_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize total wheel speed (energy usage)."""
    t = wp.tid()
    v = target_vel[t, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


# ─── Simulator ─────────────────────────────────────────────────────────────


class UphillOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=10,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        self.K = num_control_points
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.frame = 0

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        # Target position: on the flat top, a bit after the ramp ends
        # Terrain layout: flat(3m) + ramp1(5m) + slip(SLIP_WIDTH) + ramp2(5m) + flat_top
        z_per_m = np.tan(np.radians(SLOPE_ANGLE))
        total_ramp = 5.0 + SLIP_WIDTH + 5.0
        target_y = 3.0 + total_ramp + 2.0  # 2m into flat top
        target_z = z_per_m * total_ramp + WHEEL_RADIUS
        self._target_pos = wp.vec3(0.0, target_y, target_z)

        if self.viewer:
            self.viewer.set_camera(
                pos=wp.vec3(-18.0, -14.0, 15.0),
                pitch=-25.0,
                yaw=55.0,
            )
            self.viewer.show_ui = False

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.3

        build_mesh_terrain(self.builder, SLOPE_ANGLE, SLIP_WIDTH)

        rot_z90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        robot_xform = wp.transform(wp.vec3(0.0, 1.5, WHEEL_RADIUS + 0.2), rot_z90)
        create_4wheel_robot(
            self.builder,
            xform=robot_xform,
            is_visible=True,
            k_p=500.0,
            k_d=0.0,
            wheel_mu=0.1,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        T = self.clock.total_sim_steps
        num_steps = self.trajectory.body_pose.shape[0]  # T + 1

        # 1. Target tracking: L2 from target at every timestep
        wp.launch(
            kernel=target_loss_kernel,
            dim=num_steps,
            inputs=[self.trajectory.body_pose, self._target_pos, 1.0 / num_steps],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

        # 2. Smoothness: penalize speed changes (for both rear wheels)
        for dof in [REAR_DOF_OFFSET, REAR_DOF_OFFSET + 1]:
            wp.launch(
                kernel=smoothness_loss_kernel,
                dim=T - 1,
                inputs=[self.trajectory.joint_target_vel, dof, 1e-3 / T],
                outputs=[self.loss],
                device=self.solver.model.device,
            )

        # 3. Energy: penalize total wheel speed
        for dof in [REAR_DOF_OFFSET, REAR_DOF_OFFSET + 1]:
            wp.launch(
                kernel=energy_loss_kernel,
                dim=T,
                inputs=[self.trajectory.joint_target_vel, dof, 1e-4 / T],
                outputs=[self.loss],
                device=self.solver.model.device,
            )

    # ─── Single-speed spline ───────────────────────────────────────────

    def _expand(self, params):
        """K control points → T timestep speeds."""
        return self.W @ params  # (T,)

    def _contract(self, grad_v):
        """T timestep gradients → K control point gradients."""
        return (self.W.T @ grad_v) / self.W_col_sums  # (K,)

    def _apply_params(self, params):
        """Write speed spline to rear wheel DOFs (8, 9). Front wheels get 0."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)  # (T,)

        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        # Rear left (DOF 8) and rear right (DOF 9) get the same speed
        vel_np[:, 0, REAR_DOF_OFFSET] = expanded
        vel_np[:, 0, REAR_DOF_OFFSET + 1] = expanded

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def update(self):
        # Sum gradients from both rear wheel DOFs
        grad_all = self.trajectory.joint_target_vel.grad.numpy()  # (T, 1, dof_count)
        grad_speed = grad_all[:, 0, REAR_DOF_OFFSET] + grad_all[:, 0, REAR_DOF_OFFSET + 1]  # (T,)

        grad_params = self._contract(grad_speed)  # (K,)
        self.trajectory.joint_target_vel.grad.zero_()

        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self.spline_params = np.clip(self.spline_params, 0.0, 30.0)
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if not self.viewer:
            return
        if self.frame > 0 and train_iter % 10 != 0:
            return

        loss_val = self.loss.numpy()[0]

        target_xform = wp.array(
            [wp.transform(self._target_pos, wp.quat_identity())], dtype=wp.transform
        )
        target_color = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.SPHERE,
                (0.15, 0.15, 0.15),
                target_xform,
                target_color,
            )

        print(f"  Rendering iter {train_iter} (Loss: {loss_val:.2f})")
        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,
        )
        self.frame += 1

    def train(self, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        # Initial guess: 10 rad/s
        self.spline_params = np.ones(self.K, dtype=np.float64) * 5.0
        self.spline_adam = SplineAdam(K=self.K, lr=0.2)
        self._apply_params(self.spline_params)

        print(f"Uphill optimization: {T} steps, {self.K} control points, rear-wheel drive")
        print(
            f"{'Iter':>5} {'Loss':>10} {'AvgZ':>8} {'Speed[0]':>10} {'Speed[mid]':>10} {'Speed[-1]':>10} {'time':>7}"
        )
        print("-" * 65)

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            elapsed = time.perf_counter() - t0

            loss_val = self.loss.numpy()[0]
            avg_z = -loss_val / (T + 1)

            if i % 10 == 0 or i == iterations - 1:
                p = self.spline_params
                print(
                    f"{i:5d} {loss_val:10.2f} {avg_z:8.3f}"
                    f" {p[0]:10.2f} {p[self.K//2]:10.2f} {p[-1]:10.2f}"
                    f" {elapsed:6.2f}s"
                )

            self.render(i)
            self.update()
            self.tape.zero()
            self.loss.zero_()


def main():
    sim_config = SimulationConfig(
        duration_seconds=6.0,
        target_timestep_seconds=5e-2,
        num_worlds=1,
    )
    render_config = RenderingConfig(vis_type="gl")
    exec_config = ExecutionConfig(use_cuda_graph=True)
    engine_config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
        backtrack_min_iter=12,
        newton_mode="convergence",
        linear_mode="convergence",
        newton_atol=1e-3,
        linear_tol=1e-5,
        linear_atol=1e-5,
        joint_compliance=5e-8,
        contact_compliance=1e-6,
        friction_compliance=1e-6,
        regularization=1e-6,
        contact_fb_alpha=1.0,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        enable_linesearch=False,
        max_contacts_per_world=512,
        joint_constraint_level="pos",
        contact_constraint_level="pos",
    )
    logging_config = LoggingConfig()

    sim = UphillOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=10,
    )
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
