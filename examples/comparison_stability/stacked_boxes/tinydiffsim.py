"""Stacked-boxes stability benchmark — tiny-differentiable-simulator (LCP/PGS contact).

Uses pytinydiffsim with floating-base TinyMultiBody rigid bodies and impulse-based
LCP contact resolved via Projected Gauss-Seidel (PGS).  Unlike penalty-spring
simulators this solver applies velocity-level impulses rather than stiff spring
forces, so the KE/KD contact-spring parameters from config have no direct analogue.

Note: the 512:1 mass ratio (bottom 96 kg, top 49152 kg) places extreme demands on
iterative impulse solvers.  PGS propagates corrections through multi-contact chains
only one pass at a time; with such mass disparities the solver cannot deliver
sufficient impulse to the lighter boxes through the heavier ones, causing the stack
to slowly sink regardless of time-step size.  The benchmark therefore reports
max_stable_dt = 0 (never stable) for this scenario, which is itself a meaningful
result showing where LCP-PGS methods break down.

Usage:
    python tinydiffsim.py --save results/tinydiffsim.json
"""
import argparse
import json
import math
import pathlib

import numpy as np
import pytinydiffsim as pd

from config import (
    BSEARCH_MAX,
    BSEARCH_TOL,
    DENSITY,
    DURATION,
    HX1,
    HX2,
    HX3,
    STABILITY_TOL,
    Z1,
    Z2,
    Z3,
)

# PGS solver iterations — more iterations help multi-contact chains but still
# cannot overcome the fundamental mass-ratio limitation at any tested dt.
PGS_ITERS = 100


def _mass(hx: float) -> float:
    return DENSITY * (2 * hx) ** 3


def _inertia(hx: float) -> float:
    m = _mass(hx)
    return m * (2 * hx) ** 2 / 6.0


def _make_ground(world) -> pd.TinyMultiBody:
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "ground"
    ine = pd.TinyUrdfInertial()
    ine.mass = 0.0
    ine.inertia_xxyyzz = pd.Vector3(0.0, 0.0, 0.0)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.PLANE_TYPE
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(False)
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    return mb


def _make_box(world, hx: float, mass: float, z: float) -> pd.TinyMultiBody:
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "link"
    ine = pd.TinyUrdfInertial()
    ine.mass = mass
    I = _inertia(hx)
    ine.inertia_xxyyzz = pd.Vector3(I, I, I)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.BOX_TYPE
    col.geometry.box.extents = pd.Vector3(hx, hx, hx)
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(True)  # floating
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    mb.set_position(pd.Vector3(0.0, 0.0, z))
    return mb


def run_one(dt: float) -> dict:
    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)
    world.friction = 0.1  # match config MU
    dispatcher = world.get_collision_dispatcher()
    solver = pd.TinyMultiBodyConstraintSolver()
    solver.pgs_iterations_ = PGS_ITERS

    ground = _make_ground(world)
    box1 = _make_box(world, HX1, _mass(HX1), Z1)
    box2 = _make_box(world, HX2, _mass(HX2), Z2)
    box3 = _make_box(world, HX3, _mass(HX3), Z3)

    T = max(1, int(DURATION / dt))
    times = []
    heights = []
    stable = True
    g = pd.Vector3(0.0, 0.0, -9.81)

    for step in range(T):
        for mb in (box1, box2, box3):
            pd.forward_dynamics(mb, g)
        contacts = world.compute_contacts_multi_body(
            [ground, box1, box2, box3], dispatcher
        )
        for cps in contacts:
            solver.resolve_collision(cps, dt)
        for mb in (box1, box2, box3):
            pd.integrate_euler(mb, dt)

        h = box3.base_X_world().translation.z
        t = (step + 1) * dt

        if not math.isfinite(h):
            stable = False
            times.append(t)
            heights.append(None)
            break

        if abs(h - Z3) > STABILITY_TOL:
            stable = False

        times.append(t)
        heights.append(h)

    return {
        "dt": dt,
        "T": T,
        "time": times,
        "top_box_height": heights,
        "stable": stable,
    }


def find_threshold() -> dict:
    """Binary-search for the largest stable timestep.

    Pre-checks the smallest reasonable dt first.  If even that is unstable,
    returns max_stable_dt = 0 immediately (never stable).
    """
    # Pre-check: is there any stable dt at all?
    lo_check = 1e-4
    print(f"  pre-check dt={lo_check:.1e}s ...", end=" ", flush=True)
    r = run_one(lo_check)
    if not r["stable"]:
        print("UNSTABLE — never stable for this scenario")
        return {"max_stable_dt": 0.0, "n_evals": 1, "hit_max": False}
    print("stable")

    lo = lo_check
    hi = None
    probe = lo * 2.0
    n_evals = 1  # count the pre-check

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    base = {
        "simulator": "TinyDiffSim",
        "problem": "stacked_boxes",
        "density": DENSITY,
        "hx1": HX1,
        "hx2": HX2,
        "hx3": HX3,
        "z_top_initial": Z3,
        "stability_tol": STABILITY_TOL,
        "pgs_iters": PGS_ITERS,
    }

    print("TinyDiffSim — binary_search:")
    threshold = find_threshold()
    results = {**base, **threshold}

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
