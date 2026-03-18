"""Curling box trajectory optimization using MuJoCo MJX (JAX-based differentiable physics).

Comparable to examples/comparison/curling_box/curling_box_axion.py.

Optimizes the initial Y-velocity of a box sliding on a frictional ground plane
(mu=0.15). The box is in sustained ground contact throughout.
Uses reverse-mode AD (jax.grad): 1 backward pass per iteration.
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)  # force GLFW to use X11 via XWayland

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

DT = 3e-2
DURATION = 2.0
T = int(DURATION / DT)  # ~66 steps

INIT_VEL_Y = 1.0  # initial guess
TARGET_VEL_Y = 2.5  # target to recover

LEARNING_RATE = 1e-2
MAX_GRAD = 200.0

# Box: half-extents 0.2 m, density 100 kg/m³ → mass = 100 * (0.4)³ = 6.4 kg
# Inertia of solid cube: I = m*(a²+b²)/3 per axis, with half-extent a=b=c=0.2:
#   I = 6.4 * (0.04 + 0.04) / 3 ≈ 0.171 kg·m²
BOX_XML = f"""
<mujoco model="curling_box">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.15 0.1 0.01"/>
    <body name="box" pos="0 0 0.21">
      <freejoint/>
      <inertial mass="6.4" pos="0 0 0" diaginertia="0.171 0.171 0.171"/>
      <geom type="box" size="0.2 0.2 0.2"
            friction="0.15 0.1 0.01"
            solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_init_data(mx, mj_model):
    """Create MJX data with the box at its default position."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def rollout(mx, dx0, vy):
    """Simulate T steps with initial Y-velocity vy (scalar).
    Returns (T, 3) xyz positions of the box.
    """
    dx = dx0.replace(qvel=dx0.qvel.at[1].set(vy))

    def step_fn(carry, _):
        d = mjx.step(mx, carry)
        return d, d.qpos[:3]

    _, xyz_traj = jax.lax.scan(step_fn, dx, None, length=T)
    return xyz_traj  # (T, 3)


def box_loss(mx, dx0, vy, target_xyz_traj):
    xyz_traj = rollout(mx, dx0, vy)
    delta = xyz_traj - target_xyz_traj
    return jnp.sum(delta**2)


def simulate_cpu(mj_model, vy_np, label=""):
    """Run on CPU with viewer."""
    import time as _time

    import mujoco.viewer

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.qvel[1] = vy_np

    print(f"  {label} {'step':>5}  {'x':>8}  {'y':>8}  {'z':>8}")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0
        for step in range(T):
            step_start = _time.time()
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            if step % 10 == 0 or step == T - 1:
                x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                print(f"         {step:>5}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}")
            if not viewer.is_running():
                break
            elapsed = _time.time() - step_start
            _time.sleep(max(0.0, mj_model.opt.timestep - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and skip CPU viewer")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(BOX_XML)
    mx = mjx.put_model(mj_model)
    dx0 = make_init_data(mx, mj_model)

    print(f"T={T}, dt={DT}, params=1 (initial Y-velocity), reverse-mode AD (jax.grad)")

    if not args.save:
        print("Simulating target episode (CPU)...")
        simulate_cpu(mj_model, TARGET_VEL_Y, label="target")

    target_xyz_traj = jax.jit(rollout)(mx, dx0, jnp.array(TARGET_VEL_Y))
    target_xyz_traj.block_until_ready()
    print(
        f"Target final xyz: ({target_xyz_traj[-1, 0]:.3f}, {target_xyz_traj[-1, 1]:.3f}, {target_xyz_traj[-1, 2]:.3f})"
    )

    loss_fn = lambda v: box_loss(mx, dx0, v, target_xyz_traj)
    value_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.grad(loss_fn))

    print("Compiling value_fn + grad_fn (reverse-mode AD, 1 backward pass)...")
    t0 = time.perf_counter()
    vy = jnp.array(INIT_VEL_Y)
    loss = value_fn(vy)
    loss.block_until_ready()
    grad = grad_fn(vy)
    grad.block_until_ready()
    print(f"  compile: {time.perf_counter() - t0:.2f}s\n")

    print(f"Optimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (gradient descent, reverse-mode AD)")
    results = {
        "simulator": "MJX-grad",
        "problem": "curling_box",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }
    for i in range(30):
        t0 = time.perf_counter()
        loss = value_fn(vy)
        grad = grad_fn(vy)
        loss.block_until_ready()
        grad.block_until_ready()
        t_iter = time.perf_counter() - t0

        grad_clamped = jnp.clip(grad, -MAX_GRAD, MAX_GRAD)
        vy = vy - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | vy={float(vy):.4f} | "
            f"grad={float(grad):.4f} | t={t_iter * 1000:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter * 1000)

        # if loss < 1e-4:
        #     print("Converged!")
        #     break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")
    else:
        print(f"\nOptimized vy: {float(vy):.4f}  (target: {TARGET_VEL_Y})")
        print("\nSimulating optimized episode (CPU)...")
        simulate_cpu(mj_model, float(vy), label="optimized")


if __name__ == "__main__":
    main()
