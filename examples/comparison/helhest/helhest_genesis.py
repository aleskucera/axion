"""Helhest endpoint optimization using Genesis (Taichi-based differentiable physics).

Follows the official Genesis differentiable simulation recipe:
  scene.reset() → set_dofs_velocity(ctrl) × T → scene._backward()

robot.get_links_pos() returns a plain torch.Tensor (no grad_fn), so the
gradient of the loss w.r.t. chassis position is injected manually via
robot.set_pos_grad(). Then scene._backward() runs Taichi reverse-mode AD
through all T steps, which accumulates ctrl.grad via PyTorch autograd.
"""

import os
import tempfile
import time

import genesis as gs
import torch

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)

DT = 2e-3
T = 100  # steps (0.2 s)

TARGET_CTRL = [1.0, 6.0, 0.0]
INIT_CTRL = [2.0, 5.0, 0.0]

WHEEL_DOFS = [6, 7, 8]  # freejoint=6 DOFs [0..5], wheels at [6,7,8]
CHASSIS_LINK_IDX = 1    # link 0 = world, link 1 = chassis

HELHEST_MJCF = """
<mujoco model="helhest">
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" rgba="0.3 0.3 0.8 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.35 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MJCF_FILE = None


def _mjcf_file() -> str:
    global _MJCF_FILE
    if _MJCF_FILE is None:
        fd, path = tempfile.mkstemp(suffix=".xml", prefix="helhest_genesis_")
        with os.fdopen(fd, "w") as f:
            f.write(HELHEST_MJCF)
        _MJCF_FILE = path
    return _MJCF_FILE


def build_scene(requires_grad: bool = False, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.4),
            camera_fov=40,
            run_in_thread=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=_mjcf_file()))
    scene.build()
    return scene, robot


def run_episode(scene, robot, ctrl_values, show_viewer: bool = False):
    """Forward-only episode. Returns final chassis xy."""
    ctrl = torch.tensor(ctrl_values, dtype=torch.float32, device=gs.device)
    scene.reset()
    for step in range(T):
        robot.set_dofs_velocity(ctrl, dofs_idx_local=WHEEL_DOFS)
        scene.step()
        if show_viewer and step % 20 == 0:
            pos = robot.get_links_pos()[CHASSIS_LINK_IDX]
            print(f"  step {step:3d}: xy=({pos[0]:.3f}, {pos[1]:.3f})")
    return robot.get_links_pos()[CHASSIS_LINK_IDX, :2].clone()


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    # --- Target episode ---
    print(f"T={T} steps, dt={DT}s ({T * DT:.2f}s total)")
    print("\nSimulating target episode (ctrl=[1.0, 6.0, 0.0])...")
    scene, robot = build_scene(requires_grad=False, show_viewer=True)
    target_xy = run_episode(scene, robot, TARGET_CTRL, show_viewer=True)
    print(f"Target final xy: ({target_xy[0]:.3f}, {target_xy[1]:.3f})")
    input("Press Enter to start optimization...")
    scene.destroy()

    # --- Build differentiable scene (single scene, no substeps_local) ---
    print("\nBuilding differentiable scene...")
    scene, robot = build_scene(requires_grad=True)
    print(f"  n_dofs={robot.n_dofs}  n_links={robot.n_links}")

    # ctrl on gs.device, requires_grad=True (plain torch.Tensor per docs recipe)
    ctrl = torch.tensor(INIT_CTRL, dtype=torch.float32,
                        requires_grad=True, device=gs.device)
    optimizer = torch.optim.Adam([ctrl], lr=0.5)

    print(f"\nOptimizing (Adam lr=0.5, T={T})...")
    for i in range(200):
        t0 = time.perf_counter()

        scene.reset()
        for _ in range(T):
            robot.set_dofs_velocity(ctrl, dofs_idx_local=WHEEL_DOFS)
            scene.step()

        final_xy = robot.get_links_pos()[CHASSIS_LINK_IDX, :2]
        delta = final_xy - target_xy
        loss_val = 10.0 * (delta[0] ** 2 + delta[1] ** 2).item()

        # get_links_pos() returns a plain torch.Tensor (no grad_fn).
        # Inject dLoss/dPos manually into Taichi's gradient field, then
        # run Taichi backward, which calls ctrl._backward_from_ti() to
        # accumulate ctrl.grad via PyTorch autograd.
        optimizer.zero_grad()
        pos_grad = torch.zeros(3, device=gs.device)
        pos_grad[0] = 20.0 * delta[0].item()
        pos_grad[1] = 20.0 * delta[1].item()
        robot.set_pos_grad(None, False, pos_grad)
        scene._backward()

        t_iter = time.perf_counter() - t0
        grad_str = (f"({ctrl.grad[0]:.3f}, {ctrl.grad[1]:.3f}, {ctrl.grad[2]:.3f})"
                    if ctrl.grad is not None else "None")
        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"ctrl=({ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f}) | "
            f"grad={grad_str} | t={t_iter * 1000:.0f}ms"
        )

        optimizer.step()

        if loss_val < 1e-4:
            print("Converged!")
            break

    final_ctrl = ctrl.detach().tolist()
    print(f"\nOptimized: [{final_ctrl[0]:.4f}, {final_ctrl[1]:.4f}, {final_ctrl[2]:.4f}]")
    print(f"Target:    [{TARGET_CTRL[0]:.4f}, {TARGET_CTRL[1]:.4f}, {TARGET_CTRL[2]:.4f}]")

    # --- Visualize result ---
    print("\nSimulating optimized episode...")
    scene.destroy()
    scene, robot = build_scene(requires_grad=False, show_viewer=True)
    final_xy = run_episode(scene, robot, final_ctrl, show_viewer=True)
    print(f"Optimized xy: ({final_xy[0]:.3f}, {final_xy[1]:.3f})")
    print(f"Target    xy: ({target_xy[0]:.3f}, {target_xy[1]:.3f})")
    input("Press Enter to exit...")
    scene.destroy()


if __name__ == "__main__":
    main()
