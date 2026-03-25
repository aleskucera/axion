"""Control stability benchmark — Genesis (Taichi/GPU, explicit PD actuator).

Scene: single pendulum, pivot fixed at (0,0,2), 1m link, 1 kg.
       Start at Q_INIT=0 rad. Target: Q_TARGET=π/3 rad.
       Uses Genesis PD control (dof_stiffness=kp, dof_damping=kd).

Usage:
    python examples/comparison/control_stability/genesis.py \
        --experiment binary_search \
        --save examples/comparison/control_stability/results/genesis_threshold.json
"""
import argparse
import json
import math
import pathlib

import genesis as gs
import numpy as np

gs.init(backend=gs.gpu, logging_level="warning")

from config import (DURATION, LINK_LENGTH, LINK_MASS, LINK_INERTIA, Q_INIT, Q_TARGET,
                    STABILITY_TOL, DT_SWEEP_KP, DT_SWEEP_KD, DT_VALUES, GAIN_SWEEP_DT,
                    KP_VALUES, BSEARCH_KP, BSEARCH_KD, BSEARCH_MAX, BSEARCH_TOL, BSEARCH_DIVERGE)


def kd_from_kp(kp: float) -> float:
    return 2.0 * math.sqrt(kp * LINK_INERTIA)


def run_one(dt: float, kp: float, kd: float) -> dict:
    T = max(1, int(DURATION / dt))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=False,
        ),
        show_viewer=False,
    )

    # Pendulum: box link, hinged at one end to world
    pendulum = scene.add_entity(
        gs.morphs.Box(
            size=(LINK_LENGTH, 0.1, 0.1),
            pos=(LINK_LENGTH / 2, 0.0, 2.0),  # CoM at half-length from pivot
        ),
        material=gs.materials.Rigid(rho=LINK_MASS / (LINK_LENGTH * 0.1 * 0.1)),
    )

    scene.build()

    # Set PD gains on the revolute DOF
    pendulum.set_dofs_kp(np.array([kp]))
    pendulum.set_dofs_kv(np.array([kd]))

    # Initial position
    pendulum.set_dofs_position(np.array([Q_INIT]))
    pendulum.set_dofs_velocity(np.array([0.0]))

    times = []
    angles = []
    stable = True

    for step in range(T):
        pendulum.control_dofs_position(np.array([Q_TARGET]))
        scene.step()
        t = float((step + 1) * dt)
        q = float(pendulum.get_dofs_position()[0])

        if not np.isfinite(q):
            stable = False
            times.append(t)
            angles.append(None)
            break

        if abs(q) > STABILITY_TOL:
            stable = False

        times.append(t)
        angles.append(q)

    scene.destroy()

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

    results = {"simulator": "Genesis", "experiment": args.experiment, "runs": []}

    if args.experiment == "dt_sweep":
        kp, kd = DT_SWEEP_KP, DT_SWEEP_KD
        print(f"Genesis — dt_sweep (kp={kp}, kd={kd}):")
        for dt in DT_VALUES:
            print(f"  dt={dt:.3f}s ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max((abs(a) for a in run["joint_angle"] if a is not None), default=0)
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")

    elif args.experiment == "gain_sweep":
        dt = GAIN_SWEEP_DT
        print(f"Genesis — gain_sweep (dt={dt}s):")
        for kp in KP_VALUES:
            kd = kd_from_kp(kp)
            print(f"  kp={kp:5d}  kd={kd:.1f} ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_a = max((abs(a) for a in run["joint_angle"] if a is not None), default=0)
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_a:.2f} rad)")

    else:  # binary_search
        kp, kd = BSEARCH_KP, BSEARCH_KD
        print(f"Genesis — binary_search (kp={kp}, kd={kd}):")
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
