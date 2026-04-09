"""Benchmark: gradient quality diagnostics across three scenes.

Measures two things for each scene at increasing horizon lengths:
  1. Gradient accuracy: adjoint vs centered finite differences (relative error)
  2. Gradient norm decay: ||grad|| at each backward step (vanishing detection)

Scenes:
  A. Box on ground (passive, no control) — contact-only gradient path
  B. Pendulum with velocity control (no ground contact) — joint-only gradient path
  C. Helhest robot with fixed wheel velocities — full contact+friction+control path

Usage:
    python -m tests.differentiable_simulator.benchmark_gradient_quality
    python -m tests.differentiable_simulator.benchmark_gradient_quality --scene helhest
    python -m tests.differentiable_simulator.benchmark_gradient_quality --horizons 1 5 10 20 40
    python -m tests.differentiable_simulator.benchmark_gradient_quality --save results/ablation.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

wp.init()

import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.core.types import JointMode
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from helhest.common import HelhestConfig, create_helhest_model


# =============================================================================
# Scene builders
# =============================================================================


def build_box_on_ground():
    """Box sitting on ground plane — passive body with friction contact."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body, hx=0.5, hy=0.5, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def build_pendulum():
    """Pendulum with velocity-controlled revolute joint, no ground contact."""
    builder = AxionModelBuilder()

    link = builder.add_link()
    builder.add_shape_box(
        link, hx=0.5, hy=0.1, hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=100.0, has_shape_collision=False,
        ),
    )

    j = builder.add_joint_revolute(
        parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        target_ke=500.0, target_kd=10.0, label="pendulum",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
    )
    builder.add_articulation([j], label="pendulum")
    return builder.finalize_replicated(num_worlds=1, requires_grad=True)


def build_helhest():
    """Helhest 3-wheeled robot on flat ground with velocity control."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane()
    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()),
        is_visible=False,
        control_mode="velocity",
        k_p=150.0,
        k_d=0.0,
        friction_left_right=0.7,
        friction_rear=0.35,
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


# =============================================================================
# Scene descriptors
# =============================================================================

SCENES = {
    "box": {
        "build": build_box_on_ground,
        "dt": 0.01,
        "label": "Box on ground (passive)",
        # Gradient seed: random direction over all body velocities
        "seed_type": "vel",
        # For FD: perturb initial velocity
        "fd_type": "vel",
        # No control DOFs to perturb
        "control_vel": None,
    },
    "pendulum": {
        "build": build_pendulum,
        "dt": 0.01,
        "label": "Pendulum (velocity control)",
        "seed_type": "vel",
        "fd_type": "ctrl",  # perturb the target velocity
        "control_vel": lambda ndof: np.array([3.0] + [0.0] * (ndof - 1), dtype=np.float32),
    },
    "helhest": {
        "build": build_helhest,
        "dt": 0.01,
        "label": "Helhest robot (3 wheels, friction)",
        "seed_type": "vel",
        "fd_type": "ctrl",  # perturb wheel target velocities
        "control_vel": lambda ndof: _helhest_ctrl(ndof),
    },
}


def _helhest_ctrl(ndof):
    """Fixed wheel velocities for helhest: all wheels at 3 rad/s."""
    ctrl = np.zeros(ndof, dtype=np.float32)
    ctrl[6] = 3.0  # left
    ctrl[7] = 3.0  # right
    ctrl[8] = 3.0  # rear
    return ctrl


def make_engine(model, num_steps, config_overrides=None):
    kwargs = dict(
        max_newton_iters=20,
        max_linear_iters=200,
        linear_tol=1e-8,
        linear_atol=1e-8,
    )
    if config_overrides:
        kwargs.update(config_overrides)
    config = AxionEngineConfig(**kwargs)
    return AxionEngine(
        model=model, sim_steps=num_steps, config=config,
        logging_config=LoggingConfig(), differentiable_simulation=True,
    )


# =============================================================================
# Core measurement routines
# =============================================================================


def _should_normalize(config_overrides):
    return config_overrides and config_overrides.get("adjoint_gradient_normalization", False)


def measure_gradient_accuracy(scene_cfg, num_steps, rng, config_overrides=None):
    """Run forward + backward for `num_steps`, compare adjoint grad vs FD.

    Returns dict with per-DOF analytical, FD, and relative error.
    """
    model = scene_cfg["build"]()
    engine = make_engine(model, num_steps, config_overrides)
    dims = engine.dims
    dt = scene_cfg["dt"]

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()

    # Set control
    if scene_cfg["control_vel"] is not None:
        ctrl_np = scene_cfg["control_vel"](dims.joint_dof_count)
        wp.copy(
            control.joint_target_vel,
            wp.array(ctrl_np.reshape(1, -1), dtype=wp.float32, device=model.device),
        )

    # Random loss seed
    qd_size = state_in.body_qd.numpy().flatten().shape[0]
    w = rng.standard_normal(qd_size).astype(np.float32)
    w /= np.linalg.norm(w)  # unit norm for comparability

    # --- Forward ---
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=num_steps, device=model.device,
    )
    states = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states[0].body_q, state_in.body_q)
    wp.copy(states[0].body_qd, state_in.body_qd)
    for i in range(num_steps):
        contacts = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, contacts, dt)
        buffer.save_step(i, engine.data, engine.axion_contacts)

    # --- Backward (adjoint) ---
    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[num_steps],
        wp.array(
            w.reshape(engine.data.body_vel_grad.numpy().shape),
            dtype=wp.spatial_vector, device=model.device,
        ),
    )

    normalize = _should_normalize(config_overrides)
    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)
        if normalize and i > 0:
            buffer.normalize_gradients(i)

    if scene_cfg["fd_type"] == "vel":
        # Gradient w.r.t. initial velocity
        grad_a = buffer.body_vel.grad[0].numpy().flatten().copy()
    else:
        # Gradient w.r.t. control (accumulated over all steps)
        grad_a = np.zeros(dims.joint_dof_count, dtype=np.float32)
        for i in range(num_steps):
            grad_a += buffer.joint_target_vel.grad[i].numpy().flatten()

    # --- Finite differences ---
    eps = 1e-4 if scene_cfg["fd_type"] == "vel" else 1e-3

    if scene_cfg["fd_type"] == "vel":
        base = state_in.body_qd.numpy().flatten().copy()
        grad_fd = np.zeros_like(base)
        for idx in range(len(base)):
            losses = []
            for sign in [+1, -1]:
                perturbed = base.copy()
                perturbed[idx] += sign * eps
                wp.copy(
                    state_in.body_qd,
                    wp.array(
                        perturbed.reshape(state_in.body_qd.numpy().shape),
                        dtype=wp.spatial_vector, device=model.device,
                    ),
                )
                s = model.state()
                wp.copy(s.body_q, states[0].body_q)
                wp.copy(s.body_qd, state_in.body_qd)
                s_out = model.state()
                for step in range(num_steps):
                    c = model.collide(s)
                    engine.step(s, s_out, control, c, dt)
                    s, s_out = s_out, s
                losses.append(np.dot(w, s.body_qd.numpy().flatten()))
            grad_fd[idx] = (losses[0] - losses[1]) / (2 * eps)
        # Restore
        wp.copy(
            state_in.body_qd,
            wp.array(
                base.reshape(state_in.body_qd.numpy().shape),
                dtype=wp.spatial_vector, device=model.device,
            ),
        )
    else:
        # FD over control DOFs (only non-zero ones for speed)
        ctrl_base = scene_cfg["control_vel"](dims.joint_dof_count)
        active_dofs = np.nonzero(ctrl_base)[0]
        if len(active_dofs) == 0:
            active_dofs = np.arange(min(dims.joint_dof_count, 3))
        grad_fd = np.zeros(dims.joint_dof_count, dtype=np.float32)
        for dof in active_dofs:
            losses = []
            for sign in [+1, -1]:
                tv = ctrl_base.copy()
                tv[dof] += sign * eps
                wp.copy(
                    control.joint_target_vel,
                    wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device),
                )
                s_in = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
                s_out = model.state()
                for step in range(num_steps):
                    c = model.collide(s_in)
                    engine.step(s_in, s_out, control, c, dt)
                    s_in, s_out = s_out, s_in
                losses.append(np.dot(w, s_in.body_qd.numpy().flatten()))
            grad_fd[dof] = (losses[0] - losses[1]) / (2 * eps)
        # Restrict to active DOFs for error computation
        grad_a = grad_a[active_dofs]
        grad_fd = grad_fd[active_dofs]

    return grad_a, grad_fd


def measure_gradient_norm_decay(scene_cfg, num_steps, rng, config_overrides=None):
    """Run forward, then backward step-by-step, recording ||grad|| at each step.

    Returns array of shape (num_steps+1,) with gradient norms.
    norms[num_steps] = seed norm (1.0), norms[0] = norm at initial state.
    """
    model = scene_cfg["build"]()
    engine = make_engine(model, num_steps, config_overrides)
    dims = engine.dims
    dt = scene_cfg["dt"]

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()

    if scene_cfg["control_vel"] is not None:
        ctrl_np = scene_cfg["control_vel"](dims.joint_dof_count)
        wp.copy(
            control.joint_target_vel,
            wp.array(ctrl_np.reshape(1, -1), dtype=wp.float32, device=model.device),
        )

    # Random unit seed
    qd_size = state_in.body_qd.numpy().flatten().shape[0]
    w = rng.standard_normal(qd_size).astype(np.float32)
    w /= np.linalg.norm(w)

    # --- Forward ---
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=num_steps, device=model.device,
    )
    states = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states[0].body_q, state_in.body_q)
    wp.copy(states[0].body_qd, state_in.body_qd)
    for i in range(num_steps):
        contacts = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, contacts, dt)
        buffer.save_step(i, engine.data, engine.axion_contacts)

    # --- Backward, recording norms at each step ---
    norms = np.zeros(num_steps + 1, dtype=np.float64)
    norms[num_steps] = 1.0  # seed norm

    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[num_steps],
        wp.array(
            w.reshape(engine.data.body_vel_grad.numpy().shape),
            dtype=wp.spatial_vector, device=model.device,
        ),
    )

    normalize = _should_normalize(config_overrides)
    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)

        # Measure the gradient norm propagated to step i (before normalization)
        vel_grad = buffer.body_vel.grad[i].numpy().flatten()
        pose_grad = buffer.body_pose.grad[i].numpy().flatten()
        norms[i] = np.sqrt(
            np.dot(vel_grad, vel_grad) + np.dot(pose_grad, pose_grad)
        )

        if normalize and i > 0:
            buffer.normalize_gradients(i)

    return norms


# =============================================================================
# Reporting
# =============================================================================


def relative_error(a, fd):
    """Element-wise relative error, handling near-zero."""
    denom = np.maximum(np.abs(a), np.abs(fd))
    denom = np.maximum(denom, 1e-8)
    return np.abs(a - fd) / denom


def print_accuracy_report(scene_name, label, horizons, results):
    """Print a table of gradient accuracy vs horizon."""
    print(f"\n{'='*70}")
    print(f"  GRADIENT ACCURACY: {label}")
    print(f"{'='*70}")
    print(f"  {'Steps':>5}  {'Max RelErr':>10}  {'Mean RelErr':>11}  "
          f"{'||Adjoint||':>11}  {'||FD||':>11}  {'Cosine':>7}")
    print(f"  {'-'*60}")

    for n, (ga, gf) in zip(horizons, results):
        re = relative_error(ga, gf)
        norm_a = np.linalg.norm(ga)
        norm_f = np.linalg.norm(gf)
        cos = np.dot(ga, gf) / max(norm_a * norm_f, 1e-15)
        print(f"  {n:5d}  {np.max(re):10.4f}  {np.mean(re):11.4f}  "
              f"{norm_a:11.4e}  {norm_f:11.4e}  {cos:7.4f}")


def print_decay_report(scene_name, label, num_steps, norms):
    """Print gradient norm at each backward step."""
    print(f"\n{'='*70}")
    print(f"  GRADIENT NORM DECAY: {label} ({num_steps} steps)")
    print(f"{'='*70}")

    # Print a subset: first, last, and evenly-spaced intermediate
    if num_steps <= 20:
        indices = list(range(num_steps + 1))
    else:
        indices = sorted(set(
            [num_steps] +
            list(range(num_steps, -1, -max(1, num_steps // 10))) +
            [0]
        ))

    print(f"  {'Step':>5}  {'||grad||':>12}  {'Relative':>10}  {'Bar'}")
    print(f"  {'-'*50}")

    max_norm = max(norms.max(), 1e-15)
    for idx in indices:
        rel = norms[idx] / max_norm
        bar_len = int(rel * 40)
        bar = '#' * bar_len
        print(f"  {idx:5d}  {norms[idx]:12.4e}  {rel:10.4f}  {bar}")

    # Compute effective per-step decay rate
    if norms[0] > 1e-15 and norms[num_steps] > 1e-15:
        decay_rate = (norms[0] / norms[num_steps]) ** (1.0 / num_steps)
        print(f"\n  Effective per-step decay factor: {decay_rate:.4f}")
        print(f"  (1.0 = no decay, <1.0 = vanishing, >1.0 = exploding)")
    else:
        print(f"\n  Gradient collapsed to zero — cannot compute decay rate")


# =============================================================================
# Main
# =============================================================================


def run_scene(scene_name, horizons, decay_steps, config_overrides=None):
    scene_cfg = SCENES[scene_name]
    label = scene_cfg["label"]
    if config_overrides:
        label += f" [{', '.join(f'{k}={v}' for k, v in config_overrides.items())}]"
    rng = np.random.default_rng(42)

    # --- Gradient accuracy at increasing horizons ---
    accuracy_results = []
    for n in horizons:
        ga, gf = measure_gradient_accuracy(scene_cfg, n, rng, config_overrides)
        accuracy_results.append((ga, gf))

    print_accuracy_report(scene_name, label, horizons, accuracy_results)

    # --- Gradient norm decay ---
    norms = measure_gradient_norm_decay(scene_cfg, decay_steps, rng, config_overrides)
    print_decay_report(scene_name, label, decay_steps, norms)

    # --- Collect structured results ---
    accuracy_records = []
    for n, (ga, gf) in zip(horizons, accuracy_results):
        re = relative_error(ga, gf)
        norm_a = float(np.linalg.norm(ga))
        norm_f = float(np.linalg.norm(gf))
        cos = float(np.dot(ga, gf) / max(norm_a * norm_f, 1e-15))
        accuracy_records.append({
            "steps": n,
            "max_rel_err": float(np.max(re)),
            "mean_rel_err": float(np.mean(re)),
            "norm_adjoint": norm_a,
            "norm_fd": norm_f,
            "cosine": cos,
        })

    decay_rate = None
    if norms[0] > 1e-15 and norms[decay_steps] > 1e-15:
        decay_rate = float((norms[0] / norms[decay_steps]) ** (1.0 / decay_steps))

    return {
        "scene": scene_name,
        "label": label,
        "config_overrides": config_overrides or {},
        "accuracy": accuracy_records,
        "decay": {
            "num_steps": decay_steps,
            "norms": norms.tolist(),
            "decay_rate": decay_rate,
            "norm_at_step_0": float(norms[0]),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gradient quality benchmark across scenes"
    )
    parser.add_argument(
        "--scene", nargs="*", default=None,
        choices=list(SCENES.keys()),
        help="Scene(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--horizons", nargs="*", type=int, default=[1, 5, 10, 20],
        help="Horizon lengths for accuracy test (default: 1 5 10 20)",
    )
    parser.add_argument(
        "--decay-steps", type=int, default=30,
        help="Number of steps for decay measurement (default: 30)",
    )
    parser.add_argument(
        "--soft-blending", action="store_true",
        help="Enable adjoint soft mode blending (sigmoid interpolation)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05,
        help="Soft blending temperature (default: 0.05)",
    )
    parser.add_argument(
        "--regularization", type=float, default=0.0,
        help="Adjoint regularization gamma (default: 0.0 = off)",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Enable per-step gradient normalization",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    scenes = args.scene if args.scene else list(SCENES.keys())

    config_overrides = {}
    if args.soft_blending:
        config_overrides["adjoint_soft_blending"] = True
        config_overrides["adjoint_soft_blending_temperature"] = args.temperature
    if args.regularization > 0.0:
        config_overrides["adjoint_regularization"] = args.regularization
    if args.normalize:
        config_overrides["adjoint_gradient_normalization"] = True

    all_results = []
    for scene_name in scenes:
        result = run_scene(scene_name, args.horizons, args.decay_steps,
                           config_overrides or None)
        all_results.append(result)

    print(f"\n{'='*70}")
    print(f"  DONE — {len(scenes)} scene(s) benchmarked")
    print(f"{'='*70}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(all_results, indent=2))
        print(f"  Saved to {args.save}")


if __name__ == "__main__":
    main()
