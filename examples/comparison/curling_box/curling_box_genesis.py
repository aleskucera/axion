"""Curling box trajectory optimization using Genesis (Taichi-based differentiable physics).

Comparable to examples/comparison/curling_box/curling_box_axion.py.

Optimizes the initial Y-velocity of a free box sliding on a frictional ground plane.

Gradient flow:
  - set_dofs_velocity(v0, dofs_idx_local=[0,1,2]) seeds the initial velocity.
  - scene.step() runs forward; Genesis registers each step as a custom PyTorch
    autograd Function via the PyTorch-Taichi bridge.
  - box.get_state().pos returns a tensor with a grad_fn connected to v0.
  - loss.backward() propagates through the full rollout back to v0.grad.

IMPORTANT: use box.get_state().pos, NOT box.get_links_pos()[0].
  get_links_pos() returns a plain detached tensor with no grad_fn.
  get_state().pos returns a gs.Tensor that is part of the autograd graph.

Setup:
    pip install genesis-world torch
    python examples/comparison/curling_box/curling_box_genesis.py
    python examples/comparison/curling_box/curling_box_genesis.py --save examples/comparison/results/curling_box_genesis.json
"""
import argparse
import json
import math
import pathlib
import time

import genesis as gs
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT = 3e-2
T = 66  # ~2.0 s of sliding

TARGET_VEL_Y = 2.5
INIT_VEL_Y = 1.0


def build_scene(requires_grad: bool) -> tuple:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.15),
    )
    # Box: full extents 0.4×0.4×0.4 m, density 100 kg/m³ → mass ≈ 6.4 kg
    box = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.21)),
        material=gs.materials.Rigid(friction=0.15),
    )
    scene.build()
    return scene, box


# --- Compute target position with a forward-only scene ---
scene_fwd, box_fwd = build_scene(requires_grad=False)
print(f"n_dofs={box_fwd.n_dofs}  (expected 6: freejoint linear+angular vel)")

# Set only Y-velocity for target
v0_target = gs.tensor([0.0, TARGET_VEL_Y, 0.0])
box_fwd.set_dofs_velocity(v0_target, dofs_idx_local=[0, 1, 2])
for _ in range(T):
    scene_fwd.step()
target_pos = torch.tensor(box_fwd.get_state().pos[0].tolist(), device="cuda")
print(f"Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
scene_fwd.destroy()

# --- Build differentiable scene ---
scene, box = build_scene(requires_grad=True)

LR = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    # Adam state — maintained manually so we can create a fresh v0 tensor each
    # iteration, which is required to prevent Genesis from accumulating gradients
    # across rollouts through its internal Taichi autograd buffers.
    vy = INIT_VEL_Y
    m, v_sq, t = 0.0, 0.0, 0
    beta1, beta2, eps = 0.3, 0.999, 1e-8

    print(f"\nOptimizing: T={T}, dt={DT}, lr={LR} (Adam, Genesis PyTorch autograd)")
    results = {
        "simulator": "Genesis",
        "problem": "curling_box",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()

        # Fresh tensor every iteration — breaks the cross-iteration gradient chain
        v0 = gs.tensor([0.0, vy, 0.0], requires_grad=True)

        scene.reset()
        box.set_dofs_velocity(v0, dofs_idx_local=[0, 1, 2])
        for _ in range(T):
            scene.step()

        pos = box.get_state().pos[0]  # gs.Tensor with grad_fn — do NOT use get_links_pos()
        loss = ((pos - target_pos) ** 2).sum()
        loss.backward()

        t_ms = (time.perf_counter() - t0) * 1000
        loss_val = float(loss)
        vy_grad = float(v0.grad[1]) if v0.grad is not None else 0.0

        # Adam update
        t += 1
        m = beta1 * m + (1.0 - beta1) * vy_grad
        v_sq = beta2 * v_sq + (1.0 - beta2) * vy_grad**2
        m_hat = m / (1.0 - beta1**t)
        v_hat = v_sq / (1.0 - beta2**t)
        vy = vy - LR * m_hat / (math.sqrt(v_hat) + eps)

        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"vy={round(vy, 4)} | "
            f"grad_vy={round(vy_grad, 4)} | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(loss_val)
        results["time_ms"].append(t_ms)

        if loss_val < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
