"""Minimal test of Genesis differentiable simulation with an articulated robot.

As close as possible to the official Genesis robot_grad.py template:

    controls = torch.zeros(n_steps, n_dofs, requires_grad=True)
    for t in range(n_steps):
        robot.control_dofs_force(controls[t])
        scene.step()
    loss = mse(robot.get_pos(), target)
    loss.backward()

Differences forced by missing URDF / API:
  - robot.urdf  → inline MJCF (Helhest, masses/inertia from real robot)
  - robot.get_pos() → robot.get_state().pos[0]  (get_pos() n/a in 0.4.x)
  - controls is a list of gs.tensors (plain torch.zeros not tracked by Genesis AD)

Two test modes (--mode full | chassis | both):

  full    — Helhest chassis + 3 revolute wheel joints (9 DOFs)
            Genesis 0.4.x: backward hangs indefinitely during Quadrants JIT
            compilation of the ABD backward kernel (func_factor_mass).
            Run with --mode full to confirm; use --timeout 60 to cap it.

  chassis — Chassis freejoint only, no child joints (6 DOFs)
            Genesis 0.4.x: backward compiles (~12s JIT) but control_dofs_force
            is NOT on the AD tape → controls.grad is None.
            Position grad_fn IS present (AliasBackward0); only initial-state
            setters (set_dofs_velocity / set_pos / set_quat) produce gradients.

Run:
    python examples/comparison/genesis/test_robot_grad.py --mode chassis
    python examples/comparison/genesis/test_robot_grad.py --mode full --timeout 60
    python examples/comparison/genesis/test_robot_grad.py --mode both --timeout 60
"""
import argparse
import multiprocessing
import os
import sys
import tempfile
import time

import numpy.typing  # must precede genesis import (genesis<=0.3.8 needs np.typing accessible)
import genesis as gs
import torch

N_STEPS = 5
TARGET_LIST = [0.5, 0.3, 0.37]

# ---------------------------------------------------------------------------
# MJCF definitions
# ---------------------------------------------------------------------------

FULL_HELHEST_MJCF = """
<mujoco model="helhest">
  <worldbody>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              contype="0" conaffinity="0"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

CHASSIS_ONLY_MJCF = """
<mujoco model="helhest_chassis">
  <worldbody>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# ---------------------------------------------------------------------------


def write_mjcf(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def build_scene(mjcf_path: str):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build()
    return scene, robot


def _worker(mjcf: str, result_queue: multiprocessing.Queue) -> None:
    """Runs inside a subprocess so a hang can be killed externally."""
    # genesis <=0.3.8 uses np.typing.ArrayLike without importing numpy.typing first.
    import numpy.typing  # noqa: F401  — must precede genesis import
    import genesis as gs  # noqa: F811
    gs.init(backend=gs.gpu, logging_level="warning")
    target = torch.tensor(TARGET_LIST, device=gs.device)

    path = write_mjcf(mjcf)
    scene, robot = build_scene(path)
    n_dofs = robot.n_dofs
    result_queue.put(("info", f"n_dofs={n_dofs}, n_links={robot.n_links}"))

    # Control: constant velocity applied to all DOFs each step.
    # Use gs.tensor so it registers with the Genesis AD bridge.
    # set_dofs_velocity IS on the grad tape; control_dofs_force is NOT.
    ctrl = gs.tensor([0.1] * n_dofs, requires_grad=True)
    optimizer = torch.optim.Adam([ctrl], lr=0.01)

    # --- Forward ---
    t0 = time.perf_counter()
    scene.reset()
    for _ in range(N_STEPS):
        robot.set_dofs_velocity(ctrl)
        scene.step()
    fwd_ms = (time.perf_counter() - t0) * 1000
    result_queue.put(("info", f"Forward OK  ({fwd_ms:.0f} ms)"))

    # Correct 0.3.8 API: robot.get_state().pos is a gs.Tensor with scene set
    # and grad_fn present. It also registers state in _queried_states so that
    # loss.backward() → Tensor.backward() override → scene._backward() can
    # find and seed the gradient via add_grad_from_state().
    state = robot.get_state()
    pos = state.pos   # gs.Tensor, shape [1,3] or [3], grad_fn=<AliasBackward0>
    result_queue.put(("info", f"state.pos type={type(pos).__name__} shape={pos.shape} grad_fn={getattr(pos,'grad_fn',None)}"))

    loss = torch.nn.functional.mse_loss(pos.squeeze(), target)
    result_queue.put(("info", f"loss={float(loss):.4f}"))

    # --- Backward via loss.backward() (triggers scene._backward() internally) ---
    result_queue.put(("info", "loss.backward() ..."))
    try:
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bwd_ms = (time.perf_counter() - t0) * 1000
        result_queue.put(("info", f"Backward OK ({bwd_ms:.0f} ms)"))

        if ctrl.grad is not None and ctrl.grad.norm().item() > 0:
            result_queue.put(("info", f"ctrl.grad norm = {ctrl.grad.norm().item():.6f}"))
            result_queue.put(("result", "PASS"))
        else:
            result_queue.put(("info", f"ctrl.grad = {ctrl.grad}"))
            result_queue.put(("result", "PARTIAL"))
    except Exception as e:
        first_line = str(e).split("\n")[0]
        result_queue.put(("info", f"ERROR: {first_line}"))
        result_queue.put(("result", "FAIL"))


def run_test(name: str, mjcf: str, timeout_s: int) -> str:
    """Returns 'PASS', 'PARTIAL', 'FAIL', or 'TIMEOUT'."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(mjcf, q), daemon=True)
    p.start()

    deadline = time.perf_counter() + timeout_s
    verdict = None
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            kind, msg = q.get(timeout=min(remaining, 1.0))
            print(f"  {msg}")
            if kind == "result":
                verdict = msg
                break
        except Exception:
            pass
        if not p.is_alive():
            break

    p.kill()
    p.join(timeout=3)

    if verdict is None:
        verdict = "TIMEOUT"
        print(f"  TIMEOUT (>{timeout_s}s) — backward hung (Quadrants ABD JIT deadlock)")

    print(f"  → {verdict}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "chassis", "both"],
        default="both",
        help="full=Helhest with wheels  chassis=chassis only  both=run both",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-test timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    results = {}

    if args.mode in ("full", "both"):
        results["full_helhest"] = run_test(
            "Full Helhest (freejoint + 3 revolute wheel joints)",
            FULL_HELHEST_MJCF,
            timeout_s=args.timeout,
        )

    if args.mode in ("chassis", "both"):
        results["chassis_only"] = run_test(
            "Chassis only (freejoint, no child joints)",
            CHASSIS_ONLY_MJCF,
            timeout_s=args.timeout,
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
    print()
    print("Legend:")
    print("  PASS    — backward completes AND gradient flows to controls")
    print("  PARTIAL — backward completes but control_dofs_force not on AD tape")
    print("  FAIL    — backward raises an exception")
    print("  TIMEOUT — backward hung (JIT deadlock or infinite compilation)")
