"""Rigid body gradient optimization using Genesis (Taichi-based differentiable physics).

Adapted from test_differentiable_rigid in the Genesis test suite:
  https://github.com/Genesis-Embodied-AI/Genesis

Optimizes the initial position and quaternion of a free-floating box so that
after `horizon` steps it reaches a target pose (goal_pos, goal_quat).
Gradients flow through `loss.backward()` via PyTorch autograd — Genesis
registers Taichi kernels as custom autograd functions.

IMPORTANT CONSTRAINTS for gradient to work in Genesis 0.4.1:
  - enable_collision=False     (contact backward is broken / hangs on GPU)
  - disable_constraint=True    (constraint solver backward hangs on GPU)
  No ground plane, no self-collision — pure free-body dynamics under gravity.

  Contrast with ball_throw_genesis.py which uses scene._backward() and hangs
  indefinitely even for a single free-floating sphere. This script works
  because it uses loss.backward() through the PyTorch–Taichi bridge and
  avoids the constraint/contact solver entirely.

Setup:
    pip install genesis-world torch
    python examples/comparison/genesis/rigid_grad_genesis.py
    python examples/comparison/genesis/rigid_grad_genesis.py --save examples/comparison/results/rigid_grad_genesis.json
"""
import argparse
import json
import pathlib
import time

import genesis as gs
import numpy as np
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT      = 1e-2
HORIZON = 100
LR      = 1e-2

GOAL_POS  = gs.tensor([0.7, 1.0, 0.05])
GOAL_QUAT = gs.tensor([0.3, 0.2, 0.1, 0.9])
GOAL_QUAT = GOAL_QUAT / torch.norm(GOAL_QUAT, dim=-1, keepdim=True)

INIT_POS  = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
INIT_QUAT = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)


def build_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            requires_grad=True,
            gravity=(0, 0, -1),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    box = scene.add_entity(
        gs.morphs.Box(pos=(0, 0, 0), size=(0.1, 0.1, 0.2)),
        surface=gs.surfaces.Default(color=(0.9, 0.0, 0.0, 1.0)),
    )
    scene.build()
    return scene, box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    scene, box = build_scene()

    init_pos  = gs.tensor(INIT_POS.tolist(),  requires_grad=True)
    init_quat = gs.tensor(INIT_QUAT.tolist(), requires_grad=True)
    optimizer = torch.optim.Adam([init_pos, init_quat], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=1e-3
    )

    print(f"Genesis rigid-body gradient optimization")
    print(f"  horizon={HORIZON}, dt={DT}, iters={args.iters}, lr={LR}")
    print(f"  goal_pos={GOAL_POS.tolist()}, goal_quat={np.round(GOAL_QUAT.cpu().numpy(), 3).tolist()}")
    print(f"  init_pos={INIT_POS.tolist()}, init_quat={INIT_QUAT.tolist()}")
    print()

    results = {
        "simulator": "Genesis",
        "problem": "rigid_grad",
        "dt": DT,
        "horizon": HORIZON,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()

        scene.reset()
        box.set_pos(init_pos)
        box.set_quat(init_quat)

        for _ in range(HORIZON):
            scene.step()

        box_state = box.get_state()
        loss = (torch.abs(box_state.pos  - GOAL_POS).sum()
              + torch.abs(box_state.quat - GOAL_QUAT).sum())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)

        t_ms = (time.perf_counter() - t0) * 1000
        loss_val = float(loss)

        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"pos=({float(init_pos[0]):.3f}, {float(init_pos[1]):.3f}, {float(init_pos[2]):.3f}) | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(loss_val)
        results["time_ms"].append(t_ms)

        if loss_val < 1e-2:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
