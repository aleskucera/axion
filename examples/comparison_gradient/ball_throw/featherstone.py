"""Ball throw optimization using Newton Featherstone solver (BPTT).

Comparable to ball_throw_axion.py. Optimizes initial ball velocity via
backpropagation through time on Warp's tape.

Uses DT=5e-3 (T=300) — smaller than Axion's DT=3e-2 for stability with
penalty-contact dynamics, but short enough that BPTT gradients are finite.
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import ExecutionConfig
from axion import FeatherstoneEngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.differentiable_simulator import NewtonDifferentiableSimulator
from axion.simulation.sim_config import SyncMode
from newton import Model

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 5e-3
DURATION = 1.5

TARGET_VEL = wp.spatial_vector(0.0, 4.0, 7.0, 0.0, 0.0, 0.0)
INIT_VEL   = wp.spatial_vector(0.0, 2.0, 1.0, 0.0, 0.0, 0.0)
LR = 3.5e-1
MAX_GRAD = 20.0


@wp.kernel
def loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_body_q: wp.array(dtype=wp.transform),
    loss: wp.array(dtype=wp.float32),
):
    """Accumulate L2 position error for the ball (body 0)."""
    pos = wp.transform_get_translation(body_q[0])
    target_pos = wp.transform_get_translation(target_body_q[0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


class BallThrowFeatherstoneOptimizer(NewtonDifferentiableSimulator):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: FeatherstoneEngineConfig,
        logging_config: LoggingConfig,
        save_path: str = None,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            execution_config,
            engine_config,
            logging_config,
        )
        self.save_path = save_path
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

    def build_model(self) -> Model:
        self.builder.rigid_gap = 1.0
        ball = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_sphere(body=ball, radius=0.2)
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=1e4, kd=1e2, kf=1e3)
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def run_target_episode(self):
        self.episode_trajectory.save_state(self.target_states[0], 0)
        for i in range(self.clock.total_sim_steps):
            self.collision_pipeline.collide(self.target_states[i], self.contacts)
            self.solver.step(
                state_in=self.target_states[i],
                state_out=self.target_states[i + 1],
                control=self.target_controls[i],
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            self.episode_trajectory.save_state(self.target_states[i + 1], i + 1)

    def compute_loss(self):
        T = self.clock.total_sim_steps
        for t in range(T + 1):
            wp.launch(
                kernel=loss_kernel,
                dim=1,
                inputs=[self.states[t].body_q, self.episode_trajectory.body_q[t]],
                outputs=[self.loss],
                device=self.model.device,
            )

    def update(self):
        grad = self.states[0].body_qd.grad.numpy()[0]  # (6,) spatial_vector
        grad = np.nan_to_num(grad, nan=0.0, posinf=MAX_GRAD, neginf=-MAX_GRAD)
        grad = np.clip(grad, -MAX_GRAD, MAX_GRAD)
        current = self.states[0].body_qd.numpy()[0]
        new_vel = current - LR * grad
        wp.copy(
            self.states[0].body_qd,
            wp.array([new_vel], dtype=wp.spatial_vector, device=self.model.device),
        )
        self.states[0].body_qd.grad.zero_()

    def train(self, iterations=15):
        # Target episode
        wp.copy(
            self.target_states[0].body_qd,
            wp.array([TARGET_VEL], dtype=wp.spatial_vector, device=self.model.device),
        )
        self.run_target_episode()

        # Initial guess
        wp.copy(
            self.states[0].body_qd,
            wp.array([INIT_VEL], dtype=wp.spatial_vector, device=self.model.device),
        )

        T = self.clock.total_sim_steps
        print(f"\nOptimizing: T={T}, dt={self.clock.dt:.4f}, lr={LR} (Featherstone, BPTT)")
        results = {
            "simulator": "Featherstone",
            "problem": "ball_throw",
            "dt": self.clock.dt,
            "T": T,
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
            vel = self.states[0].body_qd.numpy()[0][0:3]  # linear velocity
            print(
                f"Iter {i:3d}: loss={curr_loss:.4f} | "
                f"vel=({vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f}) | t={t_ms:.0f}ms"
            )
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_ms)

            self.update()
            self.tape.zero()
            self.loss.zero_()

        if self.save_path:
            pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path).write_text(json.dumps(results, indent=2))
            print(f"Saved to {self.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    sim = BallThrowFeatherstoneOptimizer(
        simulation_config=SimulationConfig(
            duration_seconds=DURATION,
            target_timestep_seconds=DT,
            num_worlds=1,
            sync_mode=SyncMode.ALIGN_FPS_TO_DT,
        ),
        rendering_config=RenderingConfig(
            vis_type="null" if args.save else "gl",
            target_fps=30,
            usd_file=None,
            world_offset_x=5.0,
            world_offset_y=5.0,
            start_paused=False,
        ),
        execution_config=ExecutionConfig(
            use_cuda_graph=False,
            headless_steps_per_segment=10,
        ),
        engine_config=FeatherstoneEngineConfig(
            angular_damping=0.05,
            update_mass_matrix_interval=1,
            friction_smoothing=1.0,
            use_tile_gemm=False,
            fuse_cholesky=True,
        ),
        logging_config=LoggingConfig(
            enable_timing=False,
            enable_hdf5_logging=False,
        ),
        save_path=args.save,
    )
    sim.train(iterations=15)


if __name__ == "__main__":
    main()
