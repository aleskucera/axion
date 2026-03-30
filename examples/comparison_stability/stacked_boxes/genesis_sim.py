"""Stacked-boxes stability benchmark — Genesis (Taichi/GPU, forward only).

Usage:
    python examples/comparison/stacked_boxes/genesis.py \
        --save examples/comparison/stacked_boxes/results/genesis.json
"""
import argparse
import json
import pathlib

import genesis as gs
import numpy as np

gs.init(backend=gs.gpu, logging_level="warning")

from config import DURATION, DENSITY, HX1, HX2, HX3, Z1, Z2, Z3, STABILITY_TOL, KE, KD, KF, MU, BSEARCH_TOL, BSEARCH_MAX


def find_threshold() -> dict:
    lo = 0.001
    hi = None
    probe = lo * 2.0
    n_evals = 0
    while probe <= BSEARCH_MAX:
        n_evals += 1
        print(f"  probe dt={probe:.4f}s ...", end=" ", flush=True)
        r = run_one(probe)
        if r["stable"]:
            print("stable")
            lo = probe
            probe *= 2.0
        else:
            print("UNSTABLE")
            hi = probe
            break

    if hi is None:
        print(f"  Stable up to {lo:.4f}s (BSEARCH_MAX reached)")
        return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": True}

    while hi - lo > BSEARCH_TOL:
        mid = (lo + hi) / 2.0
        n_evals += 1
        print(f"  bisect dt={mid:.4f}s ...", end=" ", flush=True)
        r = run_one(mid)
        if r["stable"]:
            print("stable")
            lo = mid
        else:
            print("UNSTABLE")
            hi = mid

    print(f"  → max stable dt = {lo:.4f}s  (n_evals={n_evals})")
    return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": False}


def run_one(dt: float) -> dict:
    T = max(1, int(DURATION / dt))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=False,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    scene.add_entity(
        gs.morphs.Box(size=(2*HX1, 2*HX1, 2*HX1), pos=(0.0, 0.0, Z1)),
        material=gs.materials.Rigid(rho=DENSITY),
    )
    scene.add_entity(
        gs.morphs.Box(size=(2*HX2, 2*HX2, 2*HX2), pos=(0.0, 0.0, Z2)),
        material=gs.materials.Rigid(rho=DENSITY),
    )
    box_top = scene.add_entity(
        gs.morphs.Box(size=(2*HX3, 2*HX3, 2*HX3), pos=(0.0, 0.0, Z3)),
        material=gs.materials.Rigid(rho=DENSITY),
    )
    scene.build()

    times = []
    heights = []
    stable = True

    for step in range(T):
        scene.step()
        t = float((step + 1) * dt)
        pos = box_top.get_pos()
        h = float(pos[0, 2]) if pos.ndim > 1 else float(pos[2])

        if not np.isfinite(h):
            stable = False
            times.append(t)
            heights.append(None)
            break

        if abs(h - Z3) > STABILITY_TOL:
            stable = False

        times.append(t)
        heights.append(h)

    scene.destroy()

    return {
        "dt": dt,
        "T": T,
        "time": times,
        "top_box_height": heights,
        "stable": stable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    base = {
        "simulator": "Genesis",
        "problem": "stacked_boxes",
        "density": DENSITY,
        "hx1": HX1, "hx2": HX2, "hx3": HX3,
        "z_top_initial": Z3,
        "stability_tol": STABILITY_TOL,
    }

    print("Genesis — binary_search:")
    threshold = find_threshold()
    results = {**base, **threshold}

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
