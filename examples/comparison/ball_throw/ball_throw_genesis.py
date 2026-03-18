"""Ball throw optimization using Genesis (Taichi-based differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial velocity of a free-floating ball to reach a target
position after T steps of ballistic flight.

Gradient flow:
  - set_dofs_velocity(v0, dofs_idx_local=[0,1,2]) seeds the initial velocity.
  - scene.step() runs forward; Genesis registers each step as a custom PyTorch
    autograd Function via the PyTorch-Taichi bridge.
  - ball.get_state().pos returns a tensor with a grad_fn connected to v0.
  - loss.backward() propagates through the full rollout back to v0.grad.

IMPORTANT: use ball.get_state().pos, NOT ball.get_links_pos()[0].
  get_links_pos() returns a plain detached tensor with no grad_fn.
  get_state().pos returns a gs.Tensor that is part of the autograd graph.

Previous approach (now fixed):
  The original version injected gradients manually via ball.set_pos_grad() and
  called scene._backward() — this hangs indefinitely on GPU (deadlock in
  kernel_step_2.grad / kernel_forward_dynamics_without_qacc.grad).
  Using loss.backward() directly through PyTorch autograd avoids this.

Setup:
    pip install genesis-world torch
    python examples/comparison/ball_throw/ball_throw_genesis.py
    python examples/comparison/ball_throw/ball_throw_genesis.py --save examples/comparison/results/ball_throw_genesis.json
"""
import argparse
import json
import pathlib
import time

import genesis as gs
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT = 3e-2
T = 50  # 1.5 s of ballistic flight

TARGET_V0 = [0.0, 4.0, 7.0]
INIT_V0 = [0.0, 2.0, 1.0]


def build_scene(requires_grad: bool) -> tuple:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        show_viewer=False,
    )
    ball = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.0, 0.0, 1.0)))
    scene.build()
    return scene, ball


# --- Compute target position with a forward-only scene ---
scene_fwd, ball_fwd = build_scene(requires_grad=False)
print(f"n_dofs={ball_fwd.n_dofs}  (expected 6: freejoint linear+angular vel)")

ball_fwd.set_dofs_velocity(gs.tensor(TARGET_V0), dofs_idx_local=[0, 1, 2])
for _ in range(T):
    scene_fwd.step()
# Convert to plain torch tensor before destroying the scene
target_pos = torch.tensor(ball_fwd.get_state().pos[0].tolist(), device="cuda")
print(f"Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
scene_fwd.destroy()

# --- Build differentiable scene ---
scene, ball = build_scene(requires_grad=True)

v0 = gs.tensor(INIT_V0, requires_grad=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    optimizer = torch.optim.Adam([v0], lr=0.15)

    print(f"\nOptimizing: T={T}, dt={DT}, lr=0.2 (Adam, Genesis PyTorch autograd)")
    results = {
        "simulator": "Genesis",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()

        scene.reset()
        ball.set_dofs_velocity(v0, dofs_idx_local=[0, 1, 2])
        for _ in range(T):
            scene.step()

        pos = ball.get_state().pos[0]  # gs.Tensor with grad_fn — do NOT use get_links_pos()
        loss = ((pos - target_pos) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_ms = (time.perf_counter() - t0) * 1000
        loss_val = float(loss)
        v0_vals = [round(float(x), 3) for x in v0.detach().tolist()]
        grad_vals = [round(float(x), 3) for x in v0.grad.tolist()]

        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"v0={v0_vals} | "
            f"grad={grad_vals} | "
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
