"""Ball throw optimization using Genesis (Taichi-based differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial velocity of a free-floating ball to reach a target
position after T steps of ballistic flight.

Gradient flow:
  - set_dofs_velocity(v0) at step 0 seeds the initial velocity.
  - scene._backward() runs Taichi reverse-mode AD through T steps.
  - process_input_grad() reads velocity gradient from Taichi, calls
    v0._backward_from_ti() to accumulate v0.grad via PyTorch autograd.
  - robot.set_pos_grad() injects dLoss/dPos at the final step.

KNOWN BUG (Genesis 0.4.1 + Taichi 1.7.3 + CUDA 12.8):
  scene._backward() hangs indefinitely for any rigid body with requires_grad=True.
  Root cause: Taichi backward kernels (kernel_step_2.grad,
  kernel_forward_dynamics_without_qacc.grad) deadlock on GPU.
  This affects all rigid body scenes regardless of joint type (freejoint, hinge, etc).
  Filed at: https://github.com/Genesis-Embodied-AI/Genesis/issues
"""

import genesis as gs
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT = 3e-2
T = 50  # 1.5 s of flight

TARGET_V0 = torch.tensor([0.0, 4.0, 7.0])  # target initial velocity
INIT_V0 = torch.tensor([0.0, 2.0, 1.0])    # initial guess


def build_scene(requires_grad: bool):
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


# --- Compute target final position with a forward-only scene ---
scene_fwd, ball_fwd = build_scene(requires_grad=False)
print(f"n_dofs={ball_fwd.n_dofs}  (expected 6: freejoint linear+angular vel)")

scene_fwd.reset()
ball_fwd.set_dofs_velocity(TARGET_V0, dofs_idx_local=[0, 1, 2])
for _ in range(T):
    scene_fwd.step()
target_pos = ball_fwd.get_links_pos()[0].clone()
print(f"Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
scene_fwd.destroy()

# --- Build differentiable scene for optimization ---
scene, ball = build_scene(requires_grad=True)

v0 = gs.tensor(INIT_V0.tolist(), requires_grad=True)
optimizer = torch.optim.Adam([v0], lr=0.5)

for i in range(100):
    scene.reset()
    ball.set_dofs_velocity(v0, dofs_idx_local=[0, 1, 2])
    for _ in range(T):
        scene.step()

    pos = ball.get_links_pos()[0]
    delta = pos - target_pos
    loss_val = (delta ** 2).sum().item()

    # Inject dLoss/dPos into Taichi gradient field
    pos_grad = torch.zeros(3, device=gs.device)
    pos_grad[:] = 2.0 * delta

    optimizer.zero_grad()
    ball.set_pos_grad(None, False, pos_grad)
    scene._backward()  # <-- HANGS HERE (see module docstring)

    print(
        f"Iter {i:3d}: loss={loss_val:.4f} | "
        f"v0=({v0[0]:.3f}, {v0[1]:.3f}, {v0[2]:.3f}) | "
        f"grad=({v0.grad[0]:.3f}, {v0.grad[1]:.3f}, {v0.grad[2]:.3f})"
    )
    optimizer.step()

    if loss_val < 1e-4:
        print("Converged!")
        break
