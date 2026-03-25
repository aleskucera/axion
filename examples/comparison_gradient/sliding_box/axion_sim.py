"""Curling box trajectory optimization using Axion (Newton/Warp, implicit differentiation).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial Y-velocity of a box sliding on a frictional ground plane
(mu=0.15). The box is in sustained ground contact throughout, making this a
contact-rich benchmark distinct from ball throw (single bounce).
Uses Adam optimizer (lr=0.3).
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.sim_config import SyncMode
from newton import Model

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 6e-2
DURATION = 2.0

INIT_VEL_Y = 1.0  # initial guess for Y-velocity
TARGET_VEL_Y = 2.5  # target Y-velocity to recover


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


@wp.kernel
def set_vy_kernel(
    vy: float,
    qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    qd[0] = wp.spatial_vector(0.0, vy, 0.0, 0.0, 0.0, 0.0)


class ScalarAdam:
    """Adam optimizer for a single scalar parameter."""

    def __init__(self, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def step(self, param: float, grad: float) -> float:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CurlingBoxOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        save_path: str = None,
    ):
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.save_path = save_path
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.adam = ScalarAdam(lr=0.1, betas=(0.3, 0.999))
        self.frame = 0

        self.init_vel = wp.spatial_vector(0.0, INIT_VEL_Y, 0.0, 0.0, 0.0, 0.0)
        self.target_init_vel = wp.spatial_vector(0.0, TARGET_VEL_Y, 0.0, 0.0, 0.0, 0.0)

        self.track_body(body_idx=0, name="box", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.5
        shape_config = newton.ModelBuilder.ShapeConfig(
            ke=1e5, kd=1e2, kf=1e3, mu=0.15, density=100.0
        )

        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.21), wp.quat_identity()),
        )
        self.builder.add_shape_box(body=0, hx=0.2, hy=0.2, hz=0.2, cfg=shape_config)
        self.builder.add_ground_plane(cfg=shape_config)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        grad = float(self.trajectory.body_vel.grad[0].numpy()[0, 0][1])
        vy = float(self.states[0].body_qd.numpy()[0][1])
        print(f"  grad_vy={grad:.4f}")
        new_vy = self.adam.step(vy, grad)
        wp.launch(
            kernel=set_vy_kernel,
            dim=1,
            inputs=[new_vy],
            outputs=[self.states[0].body_qd],
        )

    def render(self, train_iter):
        if self.save_path:
            return
        if self.frame > 0 and train_iter % 3 != 0:
            return
        loss_val = self.loss.numpy()[0]

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.18, 0.18, 0.18),
                wp.array(
                    [self.trajectory.target_body_pose.numpy()[-1, 0, 0]],
                    dtype=wp.transform,
                ),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")
        self.render_episode(
            iteration=train_iter, callback=draw_extras, loop=True, loops_count=1, playback_speed=1.0
        )
        self.frame += 1

    def train(self, iterations=30):
        wp.copy(
            self.target_states[0].body_qd,
            wp.array([self.target_init_vel], dtype=wp.spatial_vector),
        )
        self.run_target_episode()

        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        print(
            f"\nOptimizing: T={self.clock.total_sim_steps}, dt={self.clock.dt:.4f}, lr={self.adam.lr} (Adam)"
        )
        results = {
            "simulator": "Axion",
            "problem": "curling_box",
            "dt": self.clock.dt,
            "T": self.clock.total_sim_steps,
            "iterations": [],
            "loss": [],
            "time_ms": [],
        }
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000

            curr_loss = self.loss.numpy()[0]
            vy = self.states[0].body_qd.numpy()[0][1]
            print(f"Iter {i:3d}: loss={curr_loss:.4f} | vy={vy:.4f} | t={t_ms:.0f}ms")
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_ms)

            self.render(i)
            self.update()
            self.tape.zero()
            self.loss.zero_()

        if self.save_path:
            pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path).write_text(json.dumps(results, indent=2))
            print(f"Saved to {self.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and run headless")
    args = parser.parse_args()

    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=DT,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null" if args.save else "gl",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=10,
    )
    engine_config = AxionEngineConfig(
        max_newton_iters=12,
        max_linear_iters=12,
        backtrack_min_iter=8,
        newton_atol=1e-3,
        linear_atol=1e-3,
        linear_tol=1e-3,
        enable_linesearch=False,
        linesearch_conservative_step_count=16,
        linesearch_conservative_upper_bound=5e-2,
        linesearch_min_step=1e-6,
        linesearch_optimistic_step_count=48,
        linesearch_optimistic_window=0.4,
        joint_compliance=6e-8,
        contact_compliance=1e-6,
        friction_compliance=1e-6,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=256,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = CurlingBoxOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        save_path=args.save,
    )
    sim.train(iterations=30)


if __name__ == "__main__":
    main()
