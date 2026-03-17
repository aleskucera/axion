"""Helhest trajectory optimization using Newton Featherstone solver (BPTT).

Comparable to examples/comparison/helhest/helhest_axion.py.

Uses the same spline control parameterization (K control points) as the Axion
comparison. Gradients are computed via backpropagation through time (BPTT)
on Warp's tape, unrolling all T simulation steps.
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

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 1e-3
DURATION = 3.0
K = 30  # number of spline control points

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

TARGET_CTRL = (1.0, 6.0, 0.0)
INIT_CTRL = (2.0, 5.0, 0.0)


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (T, K) linear interpolation matrix and per-column normalization."""
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
    """Adam optimizer for a (K, num_dofs) numpy parameter array."""

    def __init__(
        self, K: int, num_dofs: int, lr: float, betas=(0.9, 0.999), eps=1e-8, clip_grad=100.0
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.clip_grad = clip_grad
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        # Sanitize before clipping: np.clip(nan) returns nan, not 0
        grad = np.nan_to_num(grad, nan=0.0, posinf=self.clip_grad, neginf=-self.clip_grad)
        grad = np.clip(grad, -self.clip_grad, self.clip_grad)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@wp.kernel
def loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_body_q: wp.array(dtype=wp.transform),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance between chassis position at one timestep, accumulated into loss."""
    pos = wp.transform_get_translation(body_q[0])
    target_pos = wp.transform_get_translation(target_body_q[0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


class HelhestFeatherstoneOptimizer(NewtonDifferentiableSimulator):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config: FeatherstoneEngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = K,
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
        self.K = num_control_points
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        # Sample loss every LOSS_STRIDE steps to reduce tape kernel launches.
        # With T=3003, every-step loss = 3003 launches; stride=10 = ~300 launches.
        self.loss_stride = 10
        self.frame = 0

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.0

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.7,
            ke=1e4,
            kd=1e2,
            kf=1e3,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
            control_mode="velocity",
            k_p=0.0,
            k_d=100.0,
            friction_left_right=0.7,
            friction_rear=0.35,
            ke=5e4,
            kd=1e3,
            kf=1e3,
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

    def _expand(self, params: np.ndarray) -> np.ndarray:
        return self.W @ params  # (T, K) @ (K, 3) = (T, 3)

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        return (self.W.T @ grad_v) / self.W_col_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        T = self.clock.total_sim_steps
        num_dofs = self.model.joint_dof_count
        expanded = self._expand(params)
        for i in range(T):
            vel_np = np.zeros(num_dofs, dtype=np.float32)
            vel_np[WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(
                self.controls[i].joint_target_vel,
                wp.array(vel_np, dtype=wp.float32, device=self.model.device),
            )

    def compute_loss(self):
        T = self.clock.total_sim_steps
        sampled = list(range(0, T + 1, self.loss_stride))
        weight = self.trajectory_weight / len(sampled)
        for t in sampled:
            wp.launch(
                kernel=loss_kernel,
                dim=1,
                inputs=[
                    self.states[t].body_q,
                    self.episode_trajectory.body_q[t],
                    weight,
                ],
                outputs=[self.loss],
                device=self.model.device,
            )

    def update(self):
        T = self.clock.total_sim_steps
        num_dofs = self.model.joint_dof_count

        grads = np.zeros((T, num_dofs), dtype=np.float64)
        for i in range(T):
            grads[i] = self.controls[i].joint_target_vel.grad.numpy()

        grad_wheels = grads[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        nan_frac = np.mean(~np.isfinite(grad_wheels))
        finite_max = (
            np.max(np.abs(grad_wheels[np.isfinite(grad_wheels)]))
            if nan_frac < 1.0
            else float("nan")
        )
        print(f"  grad: nan_frac={nan_frac:.2%}  max_abs_finite={finite_max:.3e}")
        grad_params = self._contract(grad_wheels)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)

        for i in range(T):
            self.controls[i].joint_target_vel.grad.zero_()

        self._apply_params(self.spline_params)

    def train(self, iterations=50):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps
        num_dofs = self.model.joint_dof_count

        # --- Target episode ---
        for i in range(T):
            ctrl = np.zeros(num_dofs, dtype=np.float32)
            ctrl[WHEEL_DOF_OFFSET + 0] = TARGET_CTRL[0]
            ctrl[WHEEL_DOF_OFFSET + 1] = TARGET_CTRL[1]
            ctrl[WHEEL_DOF_OFFSET + 2] = TARGET_CTRL[2]
            wp.copy(
                self.target_controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )

        self.run_target_episode()

        if not self.save_path:
            print("Rendering target episode...")
            self.states, self.target_states = self.target_states, self.states
            self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=1.0)
            self.states, self.target_states = self.target_states, self.states

        # --- Spline setup ---
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array(
            [[INIT_CTRL[0], INIT_CTRL[1], INIT_CTRL[2]]] * self.K, dtype=np.float64
        )
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.05, betas=(0.9, 0.999), clip_grad=1.0
        )
        self._apply_params(self.spline_params)

        # --- Optimization ---
        print(
            f"\nOptimizing: T={T}, dt={self.clock.dt:.4f}, K={self.K}, lr=0.05, clip=1.0, stride={self.loss_stride} (SplineAdam, BPTT)"
        )
        results = {
            "simulator": "Featherstone",
            "problem": "helhest",
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "iterations": [],
            "loss": [],
            "time_ms": [],
        }
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            p0, pm, pN = (
                self.spline_params[0],
                self.spline_params[self.K // 2],
                self.spline_params[-1],
            )
            print(
                f"Iter {i:3d}: loss={curr_loss:.4f} | "
                f"cp[0]=({p0[0]:.2f},{p0[1]:.2f}) "
                f"cp[{self.K//2}]=({pm[0]:.2f},{pm[1]:.2f}) "
                f"cp[-1]=({pN[0]:.2f},{pN[1]:.2f}) | "
                f"t={t_iter * 1000:.0f}ms"
            )
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_iter * 1000)

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
        use_cuda_graph=False,  # BPTT requires dynamic tape; CUDA graphs not supported
        headless_steps_per_segment=10,
    )
    engine_config = FeatherstoneEngineConfig(
        angular_damping=0.05,
        update_mass_matrix_interval=1,
        friction_smoothing=1.0,
        use_tile_gemm=False,
        fuse_cholesky=True,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = HelhestFeatherstoneOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=K,
        save_path=args.save,
    )
    sim.train(iterations=50)


if __name__ == "__main__":
    main()
