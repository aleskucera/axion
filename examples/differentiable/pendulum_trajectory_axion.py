"""
Pendulum initial angular velocity optimization using implicit differentiation.

Goal: find the initial angular velocity of a single pendulum that reproduces
a target trajectory. The optimizer starts from a different angular velocity
and uses the Axion implicit gradient to recover the target.

Scene: single pendulum, pivot fixed at (0, 0, 5), link swings in the XZ plane
       (revolute joint around world Y axis).
Optimization parameter: initial angular velocity around Y (scalar).
Loss: L2 trajectory matching across all timesteps.
"""
import os
import pathlib

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

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


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
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    loss: wp.array(dtype=wp.float32),
):
    """L2 trajectory loss: sum of squared position errors across all timesteps."""
    tid = wp.tid()

    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])

    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


@wp.kernel
def update_kernel(
    body_vel_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    body_qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    """
    Update only the Y angular velocity component.
    spatial_vector layout: [linear (top), angular (bottom)]
    For our Y-axis revolute joint: angular gradient is in spatial_bottom(...)[1].
    """
    tid = wp.tid()
    if tid > 0:
        return

    angular_grad = wp.spatial_bottom(body_vel_grad[0, 0])
    grad_y = angular_grad[1]

    max_grad = 100.0
    grad_y = wp.clamp(grad_y, -max_grad, max_grad)

    wp.printf("Angular gradient (Y): %f\n", grad_y)

    current = body_qd[0]
    omega = wp.spatial_bottom(current)
    omega_new = wp.vec3(omega[0], omega[1] - alpha * grad_y, omega[2])
    body_qd[0] = wp.spatial_vector(wp.spatial_top(current), omega_new)


class PendulumVelocityOptimizer(AxionDifferentiableSimulator):
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
        self.learning_rate = 5e-3

        self.frame = 0

        # Angular velocity around Y axis (rad/s).
        # spatial_vector layout: [linear, angular] → angular Y is component [4].
        self.init_omega = 1.0
        self.target_omega = 0.3

        self.track_body(body_idx=0, name="link", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        hx = 1.0
        hy = 0.1
        hz = 0.1

        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        # Pivot at (0, 0, 5). The -pi/2 rotation around Z makes the link
        # start horizontal, giving it room to swing downward under gravity.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        self.builder.add_articulation([j0], label="pendulum")

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_vel.grad[0],
                self.learning_rate,
            ],
            outputs=[self.states[0].body_qd],
        )

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 3 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 20.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,
        )

        self.frame += 1

    def _make_body_qd(self, omega_y: float) -> wp.array:
        """Build a body_qd array with the given angular velocity around Y."""
        return wp.array(
            [wp.spatial_vector(0.0, 0.0, 0.0, 0.0, omega_y, 0.0)],
            dtype=wp.spatial_vector,
        )

    def train(self, iterations=60):
        # Run target episode to populate target trajectory
        wp.copy(self.target_states[0].body_qd, self._make_body_qd(self.target_omega))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])
        self.run_target_episode()

        # Set initial guess
        wp.copy(self.states[0].body_qd, self._make_body_qd(self.init_omega))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            omega_y = self.states[0].body_qd.numpy()[0][4]  # spatial_bottom Y
            print(
                f"Iter {i}: Loss={curr_loss:.6f} | omega_y={omega_y:.4f} (target={self.target_omega})"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = PendulumVelocityOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=60)


if __name__ == "__main__":
    main()
