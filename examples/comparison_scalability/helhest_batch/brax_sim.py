"""Helhest batch benchmark — Brax (JAX/GPU), NUM_WORLDS parallel worlds via vmap.

Brax supports GPU-parallel batch simulation natively via jax.vmap. All worlds
share the same control parameters (K spline points); loss and gradients are
averaged over worlds. JIT compilation happens on the first iteration and is
included in time_ms[0].

NOTE: Brax's positional pipeline requires dt=0.01 for stable simulation of
this robot (at dt=0.05 the robot sinks through the ground). Cylinder wheels
are approximated as spheres. T=300 steps are used (vs T=60 for others) to
maintain the same 3s simulation duration.

Usage:
    python examples/comparison_scalability/helhest_batch/brax.py
    python examples/comparison_scalability/helhest_batch/brax.py --save results/brax.json
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import brax.positional.pipeline as pipeline
import brax.io.mjcf as mjcf

from config import NUM_WORLDS, TARGET_CTRL, INIT_CTRL, DURATION

DT = 1e-2  # positional pipeline stable at dt=0.01; dt=0.05 sinks through ground
T = int(DURATION / DT)  # 300 steps
K = 10
NU = 3

TRAJECTORY_WEIGHT = 10.0

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.677"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_vel"  joint="left_wheel_j"  kv="150"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="150"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="150"/>
  </actuator>
</mujoco>
"""


def make_interp_matrix(T: int, K: int) -> jnp.ndarray:
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return jnp.array(W)


def adam_step(grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat / (jnp.sqrt(v_hat) + eps), m, v, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    sys = mjcf.loads(HELHEST_XML)
    W = make_interp_matrix(T, K)
    q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
    qd0 = jnp.zeros(sys.qd_size())

    def single_rollout(params):
        expanded = W @ params  # (T, NU)
        state = pipeline.init(sys, q0, qd0)

        def step_fn(state, act):
            state = pipeline.step(sys, state, act)
            return state, state.x.pos[0, :2]

        _, xy_traj = jax.lax.scan(step_fn, state, expanded)
        return xy_traj  # (T, 2)

    # Compute target trajectory (same for all worlds)
    target_params = jnp.tile(jnp.array(TARGET_CTRL, dtype=jnp.float32), (K, 1))
    target_xy = jax.jit(single_rollout)(target_params)
    jax.block_until_ready(target_xy)
    print(f"Target final xy: ({float(target_xy[-1, 0]):.3f}, {float(target_xy[-1, 1]):.3f})")

    def single_loss(params):
        xy = single_rollout(params)
        return jnp.mean(jnp.sum((xy - target_xy) ** 2, axis=-1)) * TRAJECTORY_WEIGHT

    # Batched loss+grad over NUM_WORLDS worlds (same params, averaged gradients)
    batched_vg = jax.vmap(jax.value_and_grad(single_loss))

    @jax.jit
    def batch_step(batch_params):
        losses, grads = batched_vg(batch_params)
        return jnp.mean(losses), jnp.mean(grads, axis=0)

    params = jnp.tile(jnp.array(INIT_CTRL, dtype=jnp.float32), (K, 1))
    # All worlds start with the same params
    batch_params = jnp.stack([params] * NUM_WORLDS)

    m_adam = jnp.zeros_like(params)
    v_adam = jnp.zeros_like(params)
    t_adam = 0

    print(
        f"\nOptimizing: T={T}, dt={DT}, K={K}, num_worlds={NUM_WORLDS} "
        f"(Brax, positional pipeline, jax.vmap)"
    )

    results = {
        "simulator": "Brax",
        "problem": "helhest_batch",
        "dt": DT,
        "T": T,
        "K": K,
        "num_worlds": NUM_WORLDS,
        "iterations": [],
        "loss": [],
        "time_ms": [],
        "peak_gpu_mb": None,
    }

    peak_gpu_mb = 0.0
    for i in range(50):
        t0 = time.perf_counter()
        avg_loss, avg_grad = batch_step(batch_params)
        jax.block_until_ready(avg_grad)
        t_iter = (time.perf_counter() - t0) * 1000

        try:
            mem_stats = jax.devices("gpu")[0].memory_stats()
            used_mb = mem_stats.get("bytes_in_use", 0) / 1024 ** 2
            peak_gpu_mb = max(peak_gpu_mb, used_mb)
        except Exception:
            pass

        grad_norm = float(jnp.linalg.norm(avg_grad))
        if grad_norm > 1.0:
            avg_grad = avg_grad / grad_norm
        update, m_adam, v_adam, t_adam = adam_step(avg_grad, m_adam, v_adam, t_adam, lr=0.01)
        params = params - update
        batch_params = jnp.stack([params] * NUM_WORLDS)

        print(f"Iter {i:3d}: loss={float(avg_loss):.4f} | num_worlds={NUM_WORLDS} | t={t_iter:.0f}ms")
        results["iterations"].append(i)
        results["loss"].append(float(avg_loss))
        results["time_ms"].append(t_iter)

    results["peak_gpu_mb"] = peak_gpu_mb if peak_gpu_mb > 0 else None

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
