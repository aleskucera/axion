"""Control stability benchmark — MuJoCo CPU (explicit position actuator).

Uses MuJoCo's <position> actuator which applies torque = kp*(q_ref - q) - kd*qd
as a feed-forward force before each constraint solve — equivalent to explicit PD.

Usage:
    python examples/comparison/control_stability/mujoco.py \
        --experiment dt_sweep \
        --save examples/comparison/control_stability/results/mujoco_dt.json

    python examples/comparison/control_stability/mujoco.py \
        --experiment binary_search \
        --save examples/comparison/control_stability/results/mujoco_threshold.json
"""
import argparse
import json
import math
import pathlib
import string

from config import (DURATION, LINK_LENGTH, LINK_MASS, LINK_INERTIA, Q_INIT, Q_TARGET,
                    STABILITY_TOL, DT_SWEEP_KP, DT_SWEEP_KD, DT_VALUES, GAIN_SWEEP_DT,
                    KP_VALUES, BSEARCH_KP, BSEARCH_KD, BSEARCH_MAX, BSEARCH_TOL, BSEARCH_DIVERGE)
import mujoco
import numpy as np


def kd_from_kp(kp: float) -> float:
    return 2.0 * math.sqrt(kp * LINK_INERTIA)


_XML_TEMPLATE = string.Template(
    """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="$dt" iterations="20" ls_iterations="20"/>
  <worldbody>
    <body name="link" pos="0 0 2" euler="0 0 -90">
      <joint name="j0" type="hinge" axis="0 1 0" pos="$neg_hl 0 0"/>
      <inertial mass="$mass" pos="${hl} 0 0"
                diaginertia="$ixx $iyy $iyy"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 $link_len 0 0"
            rgba="0.6 0.4 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ctrl" joint="j0" kp="$kp" kv="$kd"
              gear="1" forcelimited="false"/>
  </actuator>
</mujoco>
"""
)


def run_one(dt: float, kp: float, kd: float) -> dict:
    T = max(1, int(DURATION / dt))
    hl = LINK_LENGTH / 2.0
    # Inertia of rod about its end (joint location)
    ixx_end = LINK_MASS * LINK_LENGTH**2 / 3.0
    ixx_cm = LINK_MASS * LINK_LENGTH**2 / 12.0  # about CoM
    iyy_cm = ixx_cm

    xml = _XML_TEMPLATE.substitute(
        dt=dt,
        hl=hl,
        neg_hl=-hl,
        link_len=LINK_LENGTH,
        mass=LINK_MASS,
        ixx=ixx_cm,
        iyy=iyy_cm,
        kp=kp,
        kd=kd,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Start hanging at rest, target = Q_TARGET
    data.qpos[0] = Q_INIT
    data.ctrl[0] = Q_TARGET  # target angle
    mujoco.mj_forward(model, data)

    times = []
    angles = []
    stable = True

    for _ in range(T):
        mujoco.mj_step(model, data)
        q = float(data.qpos[0])
        t = float(data.time)

        if not np.isfinite(q):
            stable = False
            times.append(t)
            angles.append(None)
            break

        if abs(q) > STABILITY_TOL:
            stable = False

        times.append(t)
        angles.append(q)

    return {
        "dt": dt,
        "T": T,
        "kp": kp,
        "kd": kd,
        "time": times,
        "joint_angle": angles,
        "stable": stable,
    }


def _is_stable_threshold(dt: float, kp: float, kd: float) -> bool:
    run = run_one(dt, kp, kd)
    angles = [a for a in run["joint_angle"] if a is not None]
    if not angles:
        return False
    return max(abs(a) for a in angles) < BSEARCH_DIVERGE


def find_threshold(kp: float, kd: float) -> dict:
    lo = 0.001
    hi = None

    probe = lo * 2.0
    n_evals = 0
    while probe <= BSEARCH_MAX:
        n_evals += 1
        stable = _is_stable_threshold(probe, kp, kd)
        print(f"  probe dt={probe:.4f}s → {'STABLE' if stable else 'UNSTABLE'}")
        if not stable:
            hi = probe
            break
        lo = probe
        probe *= 2.0

    if hi is None:
        return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": True}

    while hi - lo > BSEARCH_TOL:
        mid = (lo + hi) / 2.0
        n_evals += 1
        stable = _is_stable_threshold(mid, kp, kd)
        print(f"  bisect dt={mid:.4f}s → {'STABLE' if stable else 'UNSTABLE'}")
        if stable:
            lo = mid
        else:
            hi = mid

    return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=["dt_sweep", "gain_sweep", "binary_search"], required=True
    )
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    results = {
        "simulator": "MuJoCo",
        "experiment": args.experiment,
        "runs": [],
    }

    if args.experiment == "dt_sweep":
        kp, kd = DT_SWEEP_KP, DT_SWEEP_KD
        print(f"MuJoCo — dt_sweep (kp={kp}, kd={kd}):")
        for dt in DT_VALUES:
            print(f"  dt={dt:.3f}s ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max((abs(a) for a in run["joint_angle"] if a is not None), default=0)
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")
    elif args.experiment == "gain_sweep":
        dt = GAIN_SWEEP_DT
        print(f"MuJoCo — gain_sweep (dt={dt}s):")
        for kp in KP_VALUES:
            kd = kd_from_kp(kp)
            print(f"  kp={kp:5d}  kd={kd:.1f} ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max((abs(a) for a in run["joint_angle"] if a is not None), default=0)
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")
    else:  # binary_search
        kp, kd = BSEARCH_KP, BSEARCH_KD
        print(f"MuJoCo — binary_search (kp={kp}, kd={kd}):")
        threshold = find_threshold(kp, kd)
        results["max_stable_dt"] = threshold["max_stable_dt"]
        results["n_evals"] = threshold["n_evals"]
        results["hit_max"] = threshold["hit_max"]
        results["kp"] = kp
        results["kd"] = kd
        suffix = "hit BSEARCH_MAX" if threshold["hit_max"] else f"{threshold['n_evals']} evals"
        print(f"  => max_stable_dt = {threshold['max_stable_dt']:.4f}s ({suffix})")

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
