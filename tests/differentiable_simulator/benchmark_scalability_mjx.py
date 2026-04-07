"""Scalability benchmark: MJX forward+backward time and GPU memory vs batch size.

Short horizon (0.1s) to avoid OOM on laptop GPU.
"""

import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

DT = 2e-3
DURATION = 0.1
T = int(DURATION / DT)  # 50 steps

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="100"/>
    <velocity name="right_act" joint="right_wheel_j" kv="100"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="100"/>
  </actuator>
</mujoco>
"""


def benchmark_mjx(batch_size, warmup=2, repeats=3):
    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    mx = mjx.put_model(mj_model)

    # Create batched initial data
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    dx0_single = mjx.put_data(mj_model, mj_data)

    # Batch via vmap
    dx0 = jax.tree.map(lambda x: jnp.stack([x] * batch_size), dx0_single)

    ctrl = jnp.tile(jnp.array([5.0, 5.0, 0.0]), (batch_size, 1))

    def rollout_single(dx0_s, ctrl_s):
        def step_fn(d, _):
            d = d.replace(ctrl=ctrl_s)
            d = mjx.step(mx, d)
            return d, d.qpos[:2]
        d_final, xy = jax.lax.scan(step_fn, dx0_s, None, length=T)
        return jnp.sum(xy ** 2)

    def batched_loss(dx0_batch, ctrl_batch):
        return jnp.mean(jax.vmap(rollout_single)(dx0_batch, ctrl_batch))

    loss_and_grad = jax.jit(jax.value_and_grad(batched_loss, argnums=1))

    # Compile
    print(f"  Compiling batch={batch_size}...", end=" ", flush=True)
    t0 = time.perf_counter()
    loss, grad = loss_and_grad(dx0, ctrl)
    jax.block_until_ready((loss, grad))
    compile_time = time.perf_counter() - t0
    print(f"compiled in {compile_time:.1f}s")

    # Warmup
    for _ in range(warmup):
        loss, grad = loss_and_grad(dx0, ctrl)
        jax.block_until_ready((loss, grad))

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(dx0, ctrl)
        jax.block_until_ready((loss, grad))
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = np.median(times)

    # Memory
    try:
        mem_stats = jax.devices()[0].memory_stats()
        peak_mb = mem_stats.get("peak_bytes_in_use", 0) / (1024 * 1024)
    except Exception:
        peak_mb = float("nan")

    return median_ms, peak_mb


print("=" * 70)
print(f"MJX Helhest scalability: {T} steps at dt={DT}s ({DURATION:.1f}s horizon)")
print("=" * 70)
print(f"{'batch':>8} {'time_ms':>12} {'mem_MB':>12}")
print("-" * 35)

for batch in [1, 2, 4, 8, 16, 32, 64]:
    try:
        t_ms, mem_mb = benchmark_mjx(batch)
        print(f"{batch:8d} {t_ms:12.1f} {mem_mb:12.1f}")
    except Exception as e:
        print(f"{batch:8d}  FAILED: {e}")
        break
