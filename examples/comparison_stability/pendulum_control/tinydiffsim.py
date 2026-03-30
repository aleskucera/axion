"""Control stability benchmark — tiny-differentiable-simulator (explicit PD).

Sets up a single-link pendulum as a fixed-base TinyMultiBody with one revolute
joint (Y-axis), then applies explicit PD torque each step:
    tau = kp * (Q_TARGET - q) - kd * qd

The simulator uses forward_dynamics (Featherstone ABA) + semi-implicit Euler
integration.  Because the torque is applied before the dynamics solve (open-loop
at each step), this is equivalent to explicit PD control — the same scheme as
MuJoCo, Genesis, and MJX in this benchmark suite.

Usage:
    python tinydiffsim.py --experiment binary_search --save results/tinydiffsim_threshold.json
    python tinydiffsim.py --experiment dt_sweep      --save results/tinydiffsim_dt.json
    python tinydiffsim.py --experiment gain_sweep    --save results/tinydiffsim_gain.json
"""
import argparse
import json
import math
import pathlib

import pytinydiffsim as pd

from config import (
    BSEARCH_KD,
    BSEARCH_KP,
    BSEARCH_MAX,
    BSEARCH_TOL,
    BSEARCH_DIVERGE,
    DURATION,
    DT_SWEEP_KD,
    DT_SWEEP_KP,
    DT_VALUES,
    GAIN_SWEEP_DT,
    KP_VALUES,
    LINK_INERTIA,
    LINK_LENGTH,
    LINK_MASS,
    Q_INIT,
    Q_TARGET,
    STABILITY_TOL,
)


def kd_from_kp(kp: float) -> float:
    return 2.0 * math.sqrt(kp * LINK_INERTIA)


def _build_pendulum() -> pd.TinyMultiBody:
    """Build a fixed-base pendulum with one revolute-Y joint.

    q = 0 : link pointing straight down (hanging at rest).
    Inertia parameters match MuJoCo's ixx_cm = LINK_MASS * LINK_LENGTH**2 / 12
    about the CoM, with CoM at -LINK_LENGTH/2 along the link axis (Z in body
    frame).  The parallel-axis theorem gives I_pivot = mL²/3, matching
    LINK_INERTIA from config.
    """
    urdf = pd.TinyUrdfStructures()

    # Massless fixed base
    base = pd.TinyUrdfLink()
    base.link_name = "base"
    ine_base = pd.TinyUrdfInertial()
    ine_base.mass = 0.0
    ine_base.inertia_xxyyzz = pd.Vector3(0.0, 0.0, 0.0)
    base.urdf_inertial = ine_base
    urdf.base_links = [base]

    # Pendulum link
    link = pd.TinyUrdfLink()
    link.link_name = "pendulum"
    link.parent_index = -1  # attach directly to base
    ine = pd.TinyUrdfInertial()
    ine.mass = LINK_MASS
    # Inertia about CoM for a uniform rod: I = mL²/12
    I_com = LINK_MASS * LINK_LENGTH ** 2 / 12.0
    ine.inertia_xxyyzz = pd.Vector3(I_com, I_com, 0.0)
    ine.origin_xyz = pd.Vector3(0.0, 0.0, -LINK_LENGTH / 2.0)
    link.urdf_inertial = ine
    urdf.links = [link]

    # Revolute joint around Y
    joint = pd.TinyUrdfJoint()
    joint.joint_name = "hinge"
    joint.joint_type = pd.JOINT_REVOLUTE_AXIS
    joint.parent_name = "base"
    joint.child_name = "pendulum"
    joint.joint_origin_xyz = pd.Vector3(0.0, 0.0, 0.0)
    joint.joint_origin_rpy = pd.Vector3(0.0, 0.0, 0.0)
    joint.joint_axis_xyz = pd.Vector3(0.0, 1.0, 0.0)
    urdf.joints = [joint]

    world = pd.TinyWorld()  # world not used for contact here
    mb = pd.TinyMultiBody(False)
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)

    # Zero joint damping — PD control is applied explicitly
    mb.links[0].damping = 0.0
    mb.links[0].stiffness = 0.0
    return mb


_GRAVITY = pd.Vector3(0.0, 0.0, -9.81)


def run_one(dt: float, kp: float, kd: float) -> dict:
    mb = _build_pendulum()
    mb.q[0] = Q_INIT
    mb.qd[0] = 0.0

    T = max(1, int(DURATION / dt))
    times = []
    angles = []
    stable = True

    for step in range(T):
        # Explicit PD torque applied before dynamics solve
        mb.tau[0] = kp * (Q_TARGET - mb.q[0]) - kd * mb.qd[0]

        pd.forward_kinematics(mb, mb.q, mb.qd)
        pd.forward_dynamics(mb, _GRAVITY)
        pd.integrate_euler(mb, dt)

        q = mb.q[0]
        t = (step + 1) * dt

        if not math.isfinite(q):
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
        "--experiment",
        choices=["dt_sweep", "gain_sweep", "binary_search"],
        required=True,
    )
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    results = {
        "simulator": "TinyDiffSim",
        "experiment": args.experiment,
        "runs": [],
    }

    if args.experiment == "dt_sweep":
        kp, kd = DT_SWEEP_KP, DT_SWEEP_KD
        print(f"TinyDiffSim — dt_sweep (kp={kp}, kd={kd}):")
        for dt in DT_VALUES:
            print(f"  dt={dt:.3f}s ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max(
                (abs(a) for a in run["joint_angle"] if a is not None), default=0
            )
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")

    elif args.experiment == "gain_sweep":
        dt = GAIN_SWEEP_DT
        print(f"TinyDiffSim — gain_sweep (dt={dt}s):")
        for kp in KP_VALUES:
            kd = kd_from_kp(kp)
            print(f"  kp={kp:5d}  kd={kd:.1f} ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max(
                (abs(a) for a in run["joint_angle"] if a is not None), default=0
            )
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")

    else:  # binary_search
        kp, kd = BSEARCH_KP, BSEARCH_KD
        print(f"TinyDiffSim — binary_search (kp={kp}, kd={kd}):")
        threshold = find_threshold(kp, kd)
        results["max_stable_dt"] = threshold["max_stable_dt"]
        results["n_evals"] = threshold["n_evals"]
        results["hit_max"] = threshold["hit_max"]
        results["kp"] = kp
        results["kd"] = kd
        suffix = (
            "hit BSEARCH_MAX"
            if threshold["hit_max"]
            else f"{threshold['n_evals']} evals"
        )
        print(f"  => max_stable_dt = {threshold['max_stable_dt']:.4f}s ({suffix})")

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
