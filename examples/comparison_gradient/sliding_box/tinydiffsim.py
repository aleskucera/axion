"""Sliding-box trajectory optimization using tiny-differentiable-simulator (CppAD).

Optimizes the initial Y-velocity of a sliding box (scalar parameter) to match a
target trajectory. Gradients are computed via CppAD reverse-mode AD recorded
through the full forward simulation including frictional ground contact.

The box (0.4m side, 6.4 kg) slides on a frictional ground plane (mu=0.15).
Contact uses TinyDiffSim's impulse-based LCP/PGS solver via
TinyMultiBodyConstraintSolver. CppAD differentiates through the piecewise-linear
contact/friction impulses using sub-gradients.

Usage:
    python examples/comparison_gradient/sliding_box/tinydiffsim.py
    python examples/comparison_gradient/sliding_box/tinydiffsim.py --save results/tinydiffsim.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pytinydiffsim_ad as pd

from config import DT, DURATION, INIT_VEL_Y, TARGET_VEL_Y

T = int(DURATION / DT)

LEARNING_RATE = 1e-2
MAX_GRAD = 200.0


def _make_ground(world):
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "ground"
    ine = pd.TinyUrdfInertial()
    ine.mass = pd.ADDouble(0.0)
    ine.inertia_xxyyzz = pd.Vector3(0.0, 0.0, 0.0)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.PLANE_TYPE
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(False)
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    return mb


def _make_box(world, vy: pd.ADDouble):
    """Build a floating-base box with initial Y-velocity vy."""
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "box"
    ine = pd.TinyUrdfInertial()
    ine.mass = pd.ADDouble(6.4)
    I = 6.4 * (0.4 ** 2) / 6.0
    ine.inertia_xxyyzz = pd.Vector3(I, I, I)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.BOX_TYPE
    col.geometry.box.extents = pd.Vector3(0.4, 0.4, 0.4)  # full extents
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(True)  # floating base
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    # q = [qx, qy, qz, qw, x, y, z]
    mb.q[0] = pd.ADDouble(0.0)
    mb.q[1] = pd.ADDouble(0.0)
    mb.q[2] = pd.ADDouble(0.0)
    mb.q[3] = pd.ADDouble(1.0)
    mb.q[4] = pd.ADDouble(0.0)
    mb.q[5] = pd.ADDouble(0.0)
    mb.q[6] = pd.ADDouble(0.21)
    # qd = [wx, wy, wz, vx, vy, vz]
    mb.qd[4] = vy
    return mb


def rollout_ad(vy: pd.ADDouble):
    """Run T steps and return list of Y-positions (one per step)."""
    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)
    world.friction = pd.ADDouble(0.15)

    ground_mb = _make_ground(world)
    box_mb = _make_box(world, vy)
    mbs = [ground_mb, box_mb]

    dispatcher = world.get_collision_dispatcher()
    solver = pd.TinyMultiBodyConstraintSolver()
    dt = pd.ADDouble(DT)

    y_positions = []
    for _ in range(T):
        pd.forward_kinematics(box_mb, box_mb.q, box_mb.qd)
        pd.forward_dynamics(box_mb, world.gravity)
        contacts = world.compute_contacts_multi_body(mbs, dispatcher)
        flat_contacts = [c for sublist in contacts for c in sublist]
        solver.resolve_collision(flat_contacts, dt)
        pd.integrate_euler(box_mb, dt)
        y_positions.append(box_mb.q[5])  # Y position

    return y_positions


def rollout_float(vy_val: float):
    """Float rollout — returns target Y trajectory."""
    vy_ad = pd.ADDouble(vy_val)
    y_ad = rollout_ad(vy_ad)
    return np.array([y.value() for y in y_ad])


def grad_and_loss_ad(vy_val: float, target_y: np.ndarray):
    """Compute loss + gradient via CppAD.  Returns (loss, grad_scalar)."""
    v = [pd.ADDouble(vy_val)]
    v_ind = pd.independent(v)

    y_positions = rollout_ad(v_ind[0])

    # L2 loss over full trajectory
    loss_ad = pd.ADDouble(0.0)
    for i, y in enumerate(y_positions):
        dy = y - pd.ADDouble(float(target_y[i]))
        loss_ad = loss_ad + dy * dy

    fn = pd.ADFun(v_ind, [loss_ad])
    fn.optimize()

    jac = fn.Jacobian([vy_val])
    loss_val = loss_ad.value()
    return float(loss_val), float(jac[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    target_y = rollout_float(TARGET_VEL_Y)
    print(f"Target final y: {target_y[-1]:.3f}")

    vy = float(INIT_VEL_Y)
    print(f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (TinyDiffSim, CppAD reverse-mode)")

    results = {
        "simulator": "TinyDiffSim",
        "problem": "curling_box",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(30):
        t0 = time.perf_counter()
        loss, grad = grad_and_loss_ad(vy, target_y)
        t_iter = (time.perf_counter() - t0) * 1000

        grad_clamped = float(np.clip(grad, -MAX_GRAD, MAX_GRAD))
        vy = vy - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | vy={vy:.4f} | grad={grad:.4f} | t={t_iter:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter)

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
