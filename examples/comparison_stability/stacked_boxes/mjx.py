"""Stacked-boxes stability benchmark — MuJoCo MJX (GPU, forward only).

Usage:
    python examples/comparison/stacked_boxes/mjx.py \
        --save examples/comparison/stacked_boxes/results/mjx.json
"""
import argparse
import json
import pathlib
import string

from config import DURATION, DENSITY, HX1, HX2, HX3, Z1, Z2, Z3, STABILITY_TOL, KE, KD, KF, MU, BSEARCH_TOL, BSEARCH_MAX
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np


def find_threshold() -> dict:
    lo = 0.0005
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

_XML_TEMPLATE = string.Template("""
<mujoco model="stacked_boxes">
  <option gravity="0 0 -9.81" timestep="$dt" iterations="20" ls_iterations="20"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1"
          friction="0.1 0.1 0.01" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    <body name="box1" pos="0 0 $z1">
      <freejoint/>
      <inertial mass="$m1" pos="0 0 0" diaginertia="$i1 $i1 $i1"/>
      <geom type="box" size="$hx1 $hx1 $hx1"
            friction="0.1 0.1 0.01" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    </body>
    <body name="box2" pos="0 0 $z2">
      <freejoint/>
      <inertial mass="$m2" pos="0 0 0" diaginertia="$i2 $i2 $i2"/>
      <geom type="box" size="$hx2 $hx2 $hx2"
            friction="0.1 0.1 0.01" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    </body>
    <body name="box3" pos="0 0 $z3">
      <freejoint/>
      <inertial mass="$m3" pos="0 0 0" diaginertia="$i3 $i3 $i3"/>
      <geom type="box" size="$hx3 $hx3 $hx3"
            friction="0.1 0.1 0.01" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
""")


def _mass(hx):
    return DENSITY * (2 * hx) ** 3


def _inertia(hx):
    return _mass(hx) * (2 * hx) ** 2 / 6.0


def run_one(dt: float) -> dict:
    T = max(1, int(DURATION / dt))
    xml = _XML_TEMPLATE.substitute(
        dt=dt,
        z1=Z1, z2=Z2, z3=Z3,
        hx1=HX1, hx2=HX2, hx3=HX3,
        m1=_mass(HX1), m2=_mass(HX2), m3=_mass(HX3),
        i1=_inertia(HX1), i2=_inertia(HX2), i3=_inertia(HX3),
    )
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mx = mjx.put_model(mj_model)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    dx0 = mjx.put_data(mj_model, mj_data)

    step_fn = jax.jit(lambda d: mjx.step(mx, d))
    dx = step_fn(dx0)
    dx.qpos.block_until_ready()  # warmup

    dx = dx0
    times = []
    heights = []
    stable = True

    for step in range(T):
        dx = step_fn(dx)
        dx.qpos.block_until_ready()
        t = float((step + 1) * dt)
        h = float(dx.qpos[16])  # box3 z = qpos[7*2 + 2]

        if not np.isfinite(h):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    base = {
        "simulator": "MJX",
        "problem": "stacked_boxes",
        "density": DENSITY,
        "hx1": HX1, "hx2": HX2, "hx3": HX3,
        "z_top_initial": Z3,
        "stability_tol": STABILITY_TOL,
    }

    print("MJX — binary_search:")
    threshold = find_threshold()
    results = {**base, **threshold}

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
