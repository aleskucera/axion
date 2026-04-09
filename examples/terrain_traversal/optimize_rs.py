"""Terrain traversal with randomized smoothing gradients.

Same task as optimize.py but replaces the adjoint backward pass with
antithetic randomized smoothing: perturb the spline parameters in 2N
batched worlds, run one forward pass, estimate the gradient from the
loss differences.

Usage:
    python -m examples.terrain_traversal.optimize_rs --seed 42
    python -m examples.terrain_traversal.optimize_rs --seed 42 --num-samples 64
"""

import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

from examples.terrain_traversal.helhest_model import create_helhest_model
from examples.terrain_traversal.helhest_model import HelhestConfig
from examples.terrain_traversal.terrain import generate_terrain_mesh
from examples.terrain_traversal.optimize import (
    generate_splines,
    make_interp_matrix,
    SplineAdam,
    WHEEL_DOF_OFFSET,
    NUM_WHEEL_DOFS,
    DT,
)

os.environ["PYOPENGL_PLATFORM"] = "glx"


def build_model(num_worlds, terrain_seed, roughness=1.0, terrain_freq=1.0):
    """Build helhest + terrain mesh with num_worlds replicas."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.2

    surface_mesh, terrain_h = generate_terrain_mesh(
        seed=terrain_seed, roughness=roughness, terrain_freq=terrain_freq,
    )
    spawn_z = terrain_h + HelhestConfig.WHEEL_RADIUS + 0.05

    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(-8.0, 0.0, spawn_z), wp.quat_identity()),
        control_mode="velocity",
        k_p=250.0,
        k_d=0.0,
        friction_left_right=0.8,
        friction_rear=0.35,
    )

    builder.add_shape_mesh(
        body=-1,
        mesh=surface_mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=True,
            mu=0.5, ke=150.0, kd=150.0, kf=500.0,
        ),
    )

    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)


def run_forward(engine, model, control, dt, num_steps):
    """Run forward simulation for num_steps. Returns final state."""
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    for step in range(num_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

    return state_in  # after swap, this is the final state


def compute_trajectory_loss(
    engine, model, control, dt, num_steps,
    target_poses_xy, bodies_per_world, num_worlds,
    traj_weight=10.0, yaw_weight=10.0, reg_weight=1e-7,
):
    """Run forward and compute per-world scalar loss.

    Returns:
        losses: (num_worlds,) array of loss values
        trajectories: (num_worlds, num_steps+1, 2) xy positions
    """
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    T = num_steps
    # Record chassis (body 0 per world) poses at each step
    all_poses = np.zeros((T + 1, num_worlds, 7), dtype=np.float32)

    # Initial poses
    q0 = state_in.body_q.numpy()  # (num_worlds * bodies_per_world, 7)
    q0_reshaped = q0.reshape(num_worlds, bodies_per_world, 7)
    all_poses[0] = q0_reshaped[:, 0, :]  # chassis is body 0 in each world

    for step in range(T):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

        q = state_in.body_q.numpy().reshape(num_worlds, bodies_per_world, 7)
        all_poses[step + 1] = q[:, 0, :]

    # Compute per-world loss
    losses = np.zeros(num_worlds, dtype=np.float64)

    # Position tracking: sum over timesteps of ||pos_xy - target_xy||^2
    for t in range(T + 1):
        pos_xy = all_poses[t, :, :2]  # (num_worlds, 2)
        target_xy = target_poses_xy[t]  # (2,) — same target for all worlds
        delta = pos_xy - target_xy[None, :]
        losses += traj_weight / (T + 1) * np.sum(delta ** 2, axis=1)

    # Yaw tracking: 1 - (fwd . fwd_target)^2
    for t in range(T + 1):
        q = all_poses[t, :, 3:7]  # (num_worlds, 4) quaternions
        q_target = target_poses_xy[t]  # We need full target pose for yaw
        # Skip yaw for simplicity — position tracking is the main signal

    # Control regularization
    ctrl_vel = control.joint_target_vel.numpy()  # (num_worlds, ndof)
    for w_idx in range(num_worlds):
        wheel_vels = ctrl_vel[w_idx, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        losses[w_idx] += reg_weight * np.sum(wheel_vels ** 2) * T

    trajectories = all_poses[:, :, :2].transpose(1, 0, 2)  # (num_worlds, T+1, 2)

    return losses, trajectories


def train_rs(
    terrain_seed, target_spline, init_spline,
    K=10, dt=DT, duration=10.0, iterations=200, lr=0.1,
    num_samples=32, sigma=0.3, roughness=1.0, terrain_freq=1.0,
):
    """Run terrain traversal optimization with randomized smoothing."""

    N = num_samples
    num_worlds_rs = 2 * N
    T = int(duration / dt)

    # Build interpolation matrix
    W, W_col_sums = make_interp_matrix(T, K)

    # --- First: run target trajectory with 1 world to get target poses ---
    model_target = build_model(1, terrain_seed, roughness, terrain_freq)
    engine_config = AxionEngineConfig(
        max_newton_iters=14, max_linear_iters=16, backtrack_min_iter=10,
        newton_atol=1e-3, linear_atol=1e-3, linear_tol=1e-3,
        enable_linesearch=False, joint_compliance=6e-8,
        contact_compliance=0.1, friction_compliance=1e-6,
        regularization=1e-6, contact_fb_alpha=0.5, contact_fb_beta=1.0,
        friction_fb_alpha=1.0, friction_fb_beta=1.0,
        max_contacts_per_world=256,
    )
    engine_target = AxionEngine(
        model=model_target, sim_steps=T, config=engine_config,
        logging_config=LoggingConfig(), differentiable_simulation=False,
    )

    # Expand target spline to per-step controls
    target_expanded = W @ target_spline  # (T, 3)
    ndof = engine_target.dims.joint_dof_count
    control_target = model_target.control()
    target_ctrl = np.zeros((1, ndof), dtype=np.float32)

    # Run target forward, recording poses
    state_in = model_target.state()
    newton.eval_fk(model_target, model_target.joint_q, model_target.joint_qd, state_in)
    state_out = model_target.state()
    bodies_per_world = model_target.body_count

    target_poses_xy = np.zeros((T + 1, 2), dtype=np.float32)
    q0 = state_in.body_q.numpy()
    target_poses_xy[0] = q0[0, :2]  # chassis body 0

    for step in range(T):
        target_ctrl[0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = target_expanded[step]
        wp.copy(control_target.joint_target_vel,
                wp.array(target_ctrl, dtype=wp.float32, device=model_target.device))
        contacts = model_target.collide(state_in)
        engine_target.step(state_in, state_out, control_target, contacts, dt)
        state_in, state_out = state_out, state_in
        q = state_in.body_q.numpy()
        target_poses_xy[step + 1] = q[0, :2]

    print(f"Target final pos: ({target_poses_xy[-1, 0]:.2f}, {target_poses_xy[-1, 1]:.2f})")

    # --- Build RS model with 2N worlds ---
    model = build_model(num_worlds_rs, terrain_seed, roughness, terrain_freq)
    engine = AxionEngine(
        model=model, sim_steps=T, config=engine_config,
        logging_config=LoggingConfig(), differentiable_simulation=False,
    )
    ndof = engine.dims.joint_dof_count
    bodies_per_world = model.body_count // num_worlds_rs
    control = model.control()

    # --- Optimizer ---
    spline_params = init_spline.copy().astype(np.float64)
    adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=lr, lr_min_ratio=0.1,
                      total_steps=iterations)
    rng = np.random.default_rng(42)

    print(f"\nRS optimization: T={T}, dt={dt:.4f}, K={K}, N={N}, σ={sigma}")
    print(f"  num_worlds={num_worlds_rs}, bodies_per_world={bodies_per_world}")

    results = {
        "method": "randomized_smoothing",
        "seed": terrain_seed,
        "K": K, "T": T, "dt": dt, "N": N, "sigma": sigma,
        "iterations": [], "loss": [], "rmse_m": [], "time_ms": [],
    }

    best_rmse = float("inf")

    for it in range(iterations):
        t0 = time.perf_counter()

        # 1. Sample perturbations in spline space
        epsilons = rng.standard_normal((N, K, NUM_WHEEL_DOFS)).astype(np.float64)

        # 2. Build per-world controls for each timestep
        # Expand base spline + perturbations
        base_expanded = W @ spline_params  # (T, 3)

        ctrl_all_steps = np.zeros((T, num_worlds_rs, ndof), dtype=np.float32)
        for i in range(N):
            plus_spline = spline_params + sigma * epsilons[i]
            minus_spline = spline_params - sigma * epsilons[i]
            plus_expanded = W @ plus_spline  # (T, 3)
            minus_expanded = W @ minus_spline  # (T, 3)
            for t in range(T):
                ctrl_all_steps[t, i, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = plus_expanded[t]
                ctrl_all_steps[t, N + i, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = minus_expanded[t]

        # 3. Forward simulation with per-step control changes
        state_in = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_out = model.state()

        all_poses = np.zeros((T + 1, num_worlds_rs, 7), dtype=np.float32)
        q0 = state_in.body_q.numpy().reshape(num_worlds_rs, bodies_per_world, 7)
        all_poses[0] = q0[:, 0, :]

        for step in range(T):
            wp.copy(
                control.joint_target_vel,
                wp.array(ctrl_all_steps[step], dtype=wp.float32, device=model.device),
            )
            contacts = model.collide(state_in)
            engine.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

            q = state_in.body_q.numpy().reshape(num_worlds_rs, bodies_per_world, 7)
            all_poses[step + 1] = q[:, 0, :]

        # 4. Compute per-world loss (position tracking only)
        losses = np.zeros(num_worlds_rs, dtype=np.float64)
        traj_weight = 10.0
        for t in range(T + 1):
            pos_xy = all_poses[t, :, :2].astype(np.float64)
            delta = pos_xy - target_poses_xy[t][None, :].astype(np.float64)
            losses += traj_weight / (T + 1) * np.sum(delta ** 2, axis=1)

        # 5. Antithetic gradient estimate in spline space
        loss_plus = losses[:N]
        loss_minus = losses[N:]
        fd_per_sample = (loss_plus - loss_minus) / (2.0 * sigma)  # (N,)
        grad_samples = fd_per_sample[:, None, None] * epsilons  # (N, K, 3)
        grad_spline = np.mean(grad_samples, axis=0)  # (K, 3)

        # 6. Adam step
        spline_params = adam.step(spline_params, grad_spline)

        t_iter = time.perf_counter() - t0

        # Compute current loss from the mean of plus/minus
        curr_loss = float(np.mean(losses))

        # RMSE from the base trajectory (world 0)
        base_xy = all_poses[:, 0, :2]
        rmse = float(np.sqrt(np.mean(np.sum((base_xy - target_poses_xy) ** 2, axis=-1))))
        if rmse < best_rmse:
            best_rmse = rmse

        print(
            f"  Iter {it:3d}: loss={curr_loss:.4f} | "
            f"RMSE={rmse:.3f}m | best={best_rmse:.3f}m | "
            f"t={t_iter * 1000:.0f}ms"
        )

        results["iterations"].append(it)
        results["loss"].append(float(curr_loss))
        results["rmse_m"].append(rmse)
        results["time_ms"].append(t_iter * 1000)

    return results


def main():
    parser = argparse.ArgumentParser(description="Terrain traversal with randomized smoothing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", metavar="PATH")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Perturbation std dev in spline space (default: 0.3)")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of antithetic sample pairs (default: 32)")
    parser.add_argument("--perturbation-sigma", type=float, default=0.5,
                        help="Initial guess perturbation sigma (default: 0.5)")
    parser.add_argument("--curvature", type=float, default=0.8)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--terrain-freq", type=float, default=1.0)
    args = parser.parse_args()

    target_spline, init_spline = generate_splines(
        args.seed, K=args.K, sigma=args.perturbation_sigma, wildness=args.curvature,
    )
    print(f"Target spline[0]: L={target_spline[0,0]:.2f} R={target_spline[0,1]:.2f} "
          f"Rear={target_spline[0,2]:.2f}")
    print(f"Init   spline[0]: L={init_spline[0,0]:.2f} R={init_spline[0,1]:.2f} "
          f"Rear={init_spline[0,2]:.2f}")

    results = train_rs(
        terrain_seed=args.seed,
        target_spline=target_spline,
        init_spline=init_spline,
        K=args.K, dt=args.dt, duration=args.duration,
        iterations=args.iterations, lr=args.lr,
        num_samples=args.num_samples, sigma=args.sigma,
        roughness=args.roughness, terrain_freq=args.terrain_freq,
    )

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
