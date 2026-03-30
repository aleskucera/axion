"""Curling box trajectory optimization using MuJoCo CPU finite differences.

Comparable to examples/comparison/curling_box/curling_box_axion.py.

Optimizes the initial Y-velocity (scalar parameter) of a sliding box.
Uses centered finite differences: 2 forward passes per gradient step.
Runs on CPU with standard mujoco (no JAX/GPU).
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

DT = 3e-2
DURATION = 2.0
T = int(DURATION / DT)  # ~66 steps

INIT_VEL_Y = 1.0
TARGET_VEL_Y = 2.5

LEARNING_RATE = 1e-2
MAX_GRAD = 200.0
FD_EPS = 1e-5

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


def rollout(mj_model, vy):
    """Simulate T steps with initial Y-velocity vy. Returns (T, 3) xyz."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.qvel[1] = vy
    xyz = np.zeros((T, 3))
    for t in range(T):
        mujoco.mj_step(mj_model, mj_data)
        xyz[t] = mj_data.qpos[:3]
    return xyz


def loss_fn(mj_model, vy, target_xyz):
    xyz = rollout(mj_model, vy)
    return np.sum((xyz - target_xyz) ** 2)


def grad_fd(mj_model, vy, target_xyz):
    """Centered finite differences: 2 forward passes."""
    f_plus = loss_fn(mj_model, vy + FD_EPS, target_xyz)
    f_minus = loss_fn(mj_model, vy - FD_EPS, target_xyz)
    return (f_plus - f_minus) / (2 * FD_EPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(BOX_XML)

    print(f"T={T}, dt={DT}, params=1 (initial Y-velocity), FD passes=2 (centered)")

    target_xyz = rollout(mj_model, TARGET_VEL_Y)
    print(
        f"Target final xyz: ({target_xyz[-1, 0]:.3f}, {target_xyz[-1, 1]:.3f}, {target_xyz[-1, 2]:.3f})"
    )

    vy = INIT_VEL_Y
    print(f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (gradient descent, CPU FD)")

    results = {
        "simulator": "MuJoCo-FD",
        "problem": "curling_box",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(30):
        t0 = time.perf_counter()
        loss = loss_fn(mj_model, vy, target_xyz)
        grad = grad_fd(mj_model, vy, target_xyz)
        t_iter = (time.perf_counter() - t0) * 1000

        grad_clamped = np.clip(grad, -MAX_GRAD, MAX_GRAD)
        vy = vy - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | vy={vy:.4f} | " f"grad={grad:.4f} | t={t_iter:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter)

        # if loss < 1e-4:
        #     print("Converged!")
        #     break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
