"""Helhest chassis endpoint optimization using Genesis (Taichi-based differentiable physics).

Adapted from the official Genesis robot_grad.py template.
Uses the Helhest chassis mass/inertia loaded from an inline MJCF.

Problem: optimize a constant 3-D velocity [vx, vy, vz=0] so that after T steps
the chassis reaches a target XY position.

Gradient method: loss.backward() through PyTorch–Taichi autograd bridge.

Genesis 0.4.1 articulation limitations (GPU):
  - loss.backward() hangs for articulated models (revolute wheel joints), even
    with disable_constraint=True.  Only single free-floating bodies work.
  - scene._backward() also hangs for all rigid bodies with constraints/contact.
  - Comparison is therefore limited to the chassis body (mass 85 kg, Helhest
    inertia) — wheels are omitted because their revolute joints break backward.

DOF layout of the chassis-only MJCF:
  [0..5] = freejoint: linear (0,1,2) + angular (3,4,5) velocity

This is physically identical to ball_throw_genesis.py but:
  - Helhest chassis mass/inertia instead of a light sphere
  - Horizontal XY motion (no projectile)
  - No gravity (gravity=(0,0,0)) so the target stays reachable

Setup:
    pip install genesis-world torch
    python examples/comparison/genesis/robot_grad.py
    python examples/comparison/genesis/robot_grad.py --save examples/comparison/results/helhest_genesis.json
"""
import argparse
import json
import os
import pathlib
import tempfile
import time

import genesis as gs
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT = 2e-3
T  = 100   # 0.2 s

# Control: constant chassis velocity [vx, vy, vz]
TARGET_V0 = [0.5, 0.3, 0.0]
INIT_V0   = [0.1, 0.1, 0.0]

# Chassis-only MJCF (wheels omitted — their revolute backward hangs in Genesis 0.4.1)
HELHEST_CHASSIS_MJCF = """
<mujoco model="helhest_chassis">
  <worldbody>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

_MJCF_FILE = None


def _mjcf_file() -> str:
    global _MJCF_FILE
    if _MJCF_FILE is None:
        fd, path = tempfile.mkstemp(suffix=".xml", prefix="helhest_chassis_")
        with os.fdopen(fd, "w") as f:
            f.write(HELHEST_CHASSIS_MJCF)
        _MJCF_FILE = path
    return _MJCF_FILE


def build_scene(requires_grad: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),   # no gravity — pure horizontal motion
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=_mjcf_file()))
    scene.build()
    return scene, robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    # --- Compute target position ---
    print("Computing target position...")
    scene_fwd, robot_fwd = build_scene(requires_grad=False)
    print(f"  n_dofs={robot_fwd.n_dofs}, n_links={robot_fwd.n_links}")

    ctrl_target = gs.tensor(TARGET_V0)
    scene_fwd.reset()
    robot_fwd.set_dofs_velocity(ctrl_target, dofs_idx_local=[0, 1, 2])
    for _ in range(T):
        scene_fwd.step()

    target_state = robot_fwd.get_state()
    target_pos = torch.tensor(target_state.pos[0].tolist(), device=gs.device)
    print(f"  Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    scene_fwd.destroy()

    # --- Differentiable scene ---
    print("\nBuilding differentiable scene...")
    scene, robot = build_scene(requires_grad=True)

    v0 = gs.tensor(INIT_V0, requires_grad=True)
    optimizer = torch.optim.Adam([v0], lr=0.05)

    print(f"Optimizing: T={T}, dt={DT}, lr=0.05 (Adam, loss.backward())")
    print(f"  init_v0={INIT_V0}, target_v0={TARGET_V0}")

    results = {
        "simulator": "Genesis",
        "problem":   "helhest_chassis",
        "note":      "chassis-only (wheel revolute joints break Genesis backward)",
        "dt":        DT,
        "T":         T,
        "iterations": [],
        "loss":       [],
        "time_ms":    [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()

        scene.reset()
        robot.set_dofs_velocity(v0, dofs_idx_local=[0, 1, 2])
        for _ in range(T):
            scene.step()

        pos = robot.get_state().pos[0]   # gs.Tensor with grad_fn
        loss = ((pos[:2] - target_pos[:2]) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_ms = (time.perf_counter() - t0) * 1000
        loss_val = float(loss)
        v0_vals  = [round(float(x), 4) for x in v0.detach().tolist()]
        grad_vals = ([round(float(x), 4) for x in v0.grad.tolist()]
                     if v0.grad is not None else None)

        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"v0={v0_vals} | "
            f"grad={grad_vals} | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(loss_val)
        results["time_ms"].append(t_ms)

        if loss_val < 1e-6:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
