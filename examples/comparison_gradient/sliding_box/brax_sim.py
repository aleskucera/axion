"""Sliding box optimization using Brax (JAX-based differentiable physics).

Optimizes the initial Y-velocity of a box sliding on a frictional ground plane
(mu=0.15) to match a target trajectory. Uses JAX reverse-mode AD through
Brax's positional pipeline.

Usage:
    python examples/comparison_gradient/sliding_box/brax.py
    python examples/comparison_gradient/sliding_box/brax.py --save results/brax.json
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import warnings

warnings.filterwarnings("ignore")

import brax.positional.pipeline as pipeline
import brax.io.mjcf as mjcf

from config import DT, DURATION, INIT_VEL_Y, TARGET_VEL_Y

T = int(DURATION / DT)

LEARNING_RATE = 1e-2
MAX_GRAD = 200.0

BOX_XML = f"""
<mujoco model="sliding_box">
  <option gravity="0 0 -9.81" timestep="{DT}"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.15 0.1 0.01"/>
    <body name="box" pos="0 0 0.21">
      <freejoint/>
      <inertial mass="6.4" pos="0 0 0" diaginertia="0.171 0.171 0.171"/>
      <geom type="box" size="0.2 0.2 0.2" friction="0.15 0.1 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_system():
    return mjcf.loads(BOX_XML)


def rollout(sys, vy):
    """Simulate T steps with initial Y-velocity vy (scalar).
    Returns (T,) y positions of the box.
    """
    # q: [x, y, z, qw, qx, qy, qz]; qd: [wx, wy, wz, vx, vy, vz]
    q0 = jnp.zeros(sys.q_size()).at[2].set(0.21).at[3].set(1.0)
    qd0 = jnp.zeros(sys.qd_size()).at[4].set(vy)
    state = pipeline.init(sys, q0, qd0)

    def step_fn(state, _):
        state = pipeline.step(sys, state, jnp.zeros(0))
        return state, state.x.pos[0, 1]  # y position

    _, y_traj = jax.lax.scan(step_fn, state, None, length=T)
    return y_traj  # (T,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    sys = make_system()

    # Compute target trajectory
    target_y = jax.jit(lambda v: rollout(sys, v))(jnp.array(float(TARGET_VEL_Y)))
    jax.block_until_ready(target_y)
    print(f"Target final y: {float(target_y[-1]):.3f}")

    @jax.jit
    def loss_and_grad(vy):
        def loss_fn(v):
            y = rollout(sys, v)
            return jnp.mean((y - target_y) ** 2)
        return jax.value_and_grad(loss_fn)(vy)

    vy = jnp.array(float(INIT_VEL_Y))
    print(f"\nOptimizing: T={T}, dt={DT}, params=1 (Brax, positional pipeline, JAX grad)")

    results = {
        "simulator": "Brax",
        "problem": "sliding_box",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(50):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(vy)
        jax.block_until_ready(grad)
        t_iter = (time.perf_counter() - t0) * 1000

        grad = jnp.clip(grad, -MAX_GRAD, MAX_GRAD)
        vy = vy - LEARNING_RATE * grad

        print(f"Iter {i:3d}: loss={float(loss):.4f} | vy={float(vy):.4f} | t={t_iter:.1f}ms")
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter)

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
