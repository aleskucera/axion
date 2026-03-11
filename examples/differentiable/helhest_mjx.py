"""Helhest endpoint optimization using MuJoCo MJX (JAX-based differentiable physics).

Comparable to examples/helhest/gradient/endpoint_featherstone.py.

Optimizes constant wheel velocities (left, right, rear) to drive the chassis
to a target endpoint position.

Contact differentiation: uses iterations=1 in the MJX solver so the constraint
loop is a fixed-size fori_loop compatible with JAX reverse-mode AD.
"""
import os
import time

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)  # force GLFW to use X11 via XWayland

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

# Match helhest_featherstone_diff.yaml: dt=1e-3, duration=2.0s
DT = 2e-3
DURATION = 5.0
T = int(DURATION / DT)  # 400 steps

# DOF indices: qpos = [x, y, z, qw, qx, qy, qz, theta_l, theta_r, theta_rear]
# ctrl = [vel_left, vel_right, vel_rear]
TARGET_CTRL = np.array([1.0, 6.0, 0.0], dtype=np.float32)
INIT_CTRL = np.array([2.0, 5.0, 0.0], dtype=np.float32)

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>

    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

      <!-- Fixed components (no joint = welded to chassis) -->
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

      <!-- Wheels (hinge joints around Y axis) -->
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01"

              rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01"

              rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.35 0.1 0.01"

              rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- velocity servos: torque = kv * (ctrl - joint_vel), matching k_d=100 -->
    <velocity name="left_act"  joint="left_wheel_j"  kv="100"/>
    <velocity name="right_act" joint="right_wheel_j" kv="100"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="100"/>
  </actuator>
</mujoco>
"""


def make_init_data(mx, mj_model):
    """Create initial MJX data with the robot in its default pose."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def rollout(mx, dx0, ctrl):
    """Simulate T steps with constant ctrl. Returns [T, 3] chassis xy positions."""

    def step_fn(carry, _):
        d = carry.replace(ctrl=ctrl)
        d = mjx.step(mx, d)
        return d, d.qpos[:2]  # track x, y only

    _, xy_traj = jax.lax.scan(step_fn, dx0, None, length=T)
    return xy_traj  # [T, 2]


def endpoint_loss(mx, dx0, ctrl, target_xy):
    xy_traj = rollout(mx, dx0, ctrl)
    delta = xy_traj[-1] - target_xy
    return 10.0 * jnp.dot(delta, delta)


def simulate_cpu(mj_model, ctrl_np, print_every=50):
    """Run on CPU with viewer and print chassis trajectory."""
    import mujoco.viewer

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    print(f"  {'step':>5}  {'x':>8}  {'y':>8}  {'z':>8}")
    import time as _time

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20.0
        for step in range(T):
            step_start = _time.time()
            mj_data.ctrl[:] = ctrl_np
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            if step % print_every == 0 or step == T - 1:
                x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                print(f"  {step:>5}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}")
            if not viewer.is_running():
                break
            # real-time pacing
            elapsed = _time.time() - step_start
            _time.sleep(max(0.0, mj_model.opt.timestep - elapsed))


def main():
    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    mx = mjx.put_model(mj_model)
    dx0 = make_init_data(mx, mj_model)

    print(f"qpos size: {mj_model.nq}, ctrl size: {mj_model.nu}, T={T}, dt={DT}")

    # --- Visualize target episode on CPU first ---
    print("Simulating target episode (CPU)...")
    simulate_cpu(mj_model, TARGET_CTRL)

    # --- Target episode ---
    target_ctrl = jnp.array(TARGET_CTRL)
    target_xy = rollout(mx, dx0, target_ctrl)[-1]
    print(f"Target final xy: ({target_xy[0]:.3f}, {target_xy[1]:.3f})")

    # --- Compile loss + grad ---
    ctrl = jnp.array(INIT_CTRL)

    # MJX constraint solver uses while_loop with dynamic convergence check,
    # which is incompatible with reverse-mode AD (jax.grad).
    # Forward-mode AD (jax.jacfwd / jvp) works through while_loop.
    # Cost: n_params forward passes per iteration (n=3 here, so 3x overhead).
    loss_fn = lambda c: endpoint_loss(mx, dx0, c, target_xy)
    value_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.jacfwd(loss_fn))  # forward-mode, compatible with while_loop

    print("Compiling value_fn...")
    t0 = time.perf_counter()
    loss = value_fn(ctrl)
    loss.block_until_ready()
    print(f"  value compile: {time.perf_counter() - t0:.2f}s")

    print("Compiling grad_fn (3x forward passes)...")
    t0 = time.perf_counter()
    grad = grad_fn(ctrl)
    grad.block_until_ready()
    print(f"  grad compile:  {time.perf_counter() - t0:.2f}s\n")

    # --- Adam optimizer (lr=1.0, matching endpoint_featherstone.py) ---
    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(ctrl)

    print(f"Optimizing: T={T}, dt={DT}, lr=1.0 (Adam, forward-mode AD)")
    for i in range(200):
        t0 = time.perf_counter()
        loss = value_fn(ctrl)
        grad = grad_fn(ctrl)
        loss.block_until_ready()
        grad.block_until_ready()
        t_iter = time.perf_counter() - t0

        updates, opt_state = optimizer.update(grad, opt_state)
        ctrl = optax.apply_updates(ctrl, updates)

        print(
            f"Iter {i:3d}: Loss={loss:.4f} | "
            f"ctrl=({ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f}) | "
            f"t={t_iter * 1000:.1f}ms"
        )

        if loss < 1e-4:
            print("Converged!")
            break


if __name__ == "__main__":
    main()
