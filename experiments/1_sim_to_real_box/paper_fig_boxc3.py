"""Campaign-3 version of the paper's box_sim_to_real figure — SAME layout as
plot_paper_panels.make_fig1 (xy, z-rise, bar; serif, shared bottom legend),
with the representative held-out sample from the 14-run dataset.

Sample: ostrich9 (joint-median held-out run under both engines' identified
configs). Engines: Ostrich (identified constant-mu), MuJoCo (c3-identified =
frozen c1 with rear/tor from the c3 grid), Semi-Implicit (c3-identified:
h=0.25 ms, flat-optimum stiffness = c1 values, recalibrated cmd scale).

    .venv/bin/python experiments/1_sim_to_real_box/paper_fig_boxc3.py
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

import eval_campaign2 as ec
from common_box import DATA_DIR, RESULTS_DIR, load_gt
import examples.helhest_junior.replay_real as rr
from plot_best14 import _run_mj_c3

SAMPLE = "ostrich9"
OSTRICH_CFG = dict(k_p=10000.0, mu_front=0.6, mu_rear=0.6,
                   mu_long_front=0.8, mu_long_rear=1.2)
# held-out (10-run) means at identified configs (SI: si_c3_final.json)
BAR = {"Ostrich": (0.208, 0.05), "MuJoCo": (0.315, 0.002),
       "Semi-Implicit": (0.268, 0.00025)}

SIM_COLORS = {"Ostrich": ps.COLORS["Ostrich"], "MuJoCo": ps.COLORS["MuJoCo"],
              "Semi-Implicit": ps.COLORS["Semi-Implicit"]}
SIM_ORDER = ["Ostrich", "MuJoCo", "Semi-Implicit"]

# This is the paper's only figure* : it is included at 0.74\textwidth, not at
# \columnwidth. savefig crops with bbox_inches="tight", so DRAWN_IN is the
# CROPPED width, measured from the emitted PNG and pinned here; the script
# reports it on every run so a drift shows up. Re-running the simulations to
# converge it is expensive, hence --cache below.
PRINTED_IN = 0.74 * ps.TEXT_IN
FIG_W, FIG_H = 16.0, 3.2
DRAWN_IN = 13.79
S = ps.apply(drawn_in=DRAWN_IN, printed_in=PRINTED_IN)

_orig_init = rr.HelhestJuniorReplaySimulator.__init__
_PATCH_ACTIVE = [False]


def _patched_init(self, *a, **kw):
    # OSTRICH_CFG applies only while the Ostrich runner is active --
    # a blanket patch would silently override the other engines' configs.
    if _PATCH_ACTIVE[0]:
        kw.update(OSTRICH_CFG)
    _orig_init(self, *a, **kw)


rr.HelhestJuniorReplaySimulator.__init__ = _patched_init


def _display(name):
    if name == "Ostrich":
        return r"\textbf{Ostrich}"
    return {"Semi-Implicit": "Semi-Impl."}.get(name, name)


SI_C3 = dict(dt=0.00025, ke=8e4, kd=2e3)  # si_c3_final.json
SI_SCALE = 0.9193


def load_trajs(gt, cache=None):
    """Run the three engines. With `cache`, reuse a previous run's output.

    The cache stores exactly what the simulators produced, so a cached figure
    is identical to a freshly simulated one; it exists only so the layout can
    be iterated without paying for the GPU replay each time.
    """
    if cache is not None and cache.is_file():
        blob = json.loads(cache.read_text())
        if blob.get("sample") == SAMPLE:
            print(f"[cache] reusing trajectories from {cache}")
            return {k: (np.asarray(v[0]), np.asarray(v[1]))
                    for k, v in blob["trajs"].items()}
        print(f"[cache] {cache} is for sample {blob.get('sample')!r}, re-running")
    trajs = {}
    _PATCH_ACTIVE[0] = True
    try:
        so = ec._score_run(*ec.run_ostrich(gt, 0.937), gt)
    finally:
        _PATCH_ACTIVE[0] = False
    trajs["Ostrich"] = (np.asarray(so["sim_rel"]),
                        np.asarray(so["sim_t_aligned"]))
    sm = ec._score_run(*_run_mj_c3(gt), gt)
    trajs["MuJoCo"] = (np.asarray(sm["sim_rel"]),
                       np.asarray(sm["sim_t_aligned"]))
    ec.C1_SI = dict(ec.C1_SI, **SI_C3)
    ss = ec._score_run(*ec.run_semi_implicit(gt, SI_SCALE), gt)
    trajs["Semi-Implicit"] = (np.asarray(ss["sim_rel"]),
                              np.asarray(ss["sim_t_aligned"]))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"sample": SAMPLE,
             "trajs": {k: [v[0].tolist(), v[1].tolist()] for k, v in trajs.items()}}))
        print(f"[cache] wrote {cache}")
    return trajs


def panel_xy(ax, trajs, gt):
    box = gt["box"]
    cx, cy = box["center"][:2]
    hx, hy = box["half_extents"][:2]
    yb = box.get("yaw", 0.0)
    c, sn = np.cos(yb), np.sin(yb)
    R = np.array([[c, -sn], [sn, c]])
    cc = (np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
                    [-hx, -hy]]) @ R.T) + [cx, cy]
    ax.fill(cc[:, 0], cc[:, 1], color="gray", alpha=0.18, zorder=1)
    ax.plot(cc[:, 0], cc[:, 1], color="dimgray", lw=0.5 * ps.LW * S, ls="--",
            zorder=1)
    # label above the shape, parallel to the box side (rotation normalized
    # to [-90, 90] deg so the text stays upright). Anchored at the midpoint
    # of the two highest corners = visually centered above the obstacle.
    # Hand-tune with the offsets below (meters, world frame).
    LABEL_DX = 0.0
    LABEL_DY = 0.08
    top2 = cc[np.argsort(cc[:, 1])[-2:]]
    rot = (np.degrees(yb) + 90.0) % 180.0 - 90.0
    ax.text(top2[:, 0].mean() + LABEL_DX, cc[:, 1].max() + LABEL_DY,
            "obstacle", ha="center", va="bottom",
            rotation=rot, rotation_mode="anchor",
            fontsize=ps.PRINT["annot"] * S, color="dimgray", style="italic",
            zorder=2)

    real_x = np.asarray(gt["real"]["x"])
    real_y = np.asarray(gt["real"]["y"])
    ax.plot(real_x, real_y, "k--", lw=ps.LW * S, label="Real robot", zorder=10)
    for sim in SIM_ORDER:
        sr, _ = trajs[sim]
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(sr[:, 0], sr[:, 1], "-", color=SIM_COLORS[sim], lw=0.8 * ps.LW * S,
                label=_display(sim), zorder=zord)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-0.1, max(real_x.max(), cc[:, 0].max()) + 0.3)
    ax.set_ylim(-0.9, 0.6)
    # equal data scale so the (rotated) obstacle rectangle stays orthogonal
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=ps.GRID_MAJOR["alpha"], linewidth=ps.GRID_MAJOR["lw"] * S)


def panel_z(ax, trajs, gt):
    t_lo, t_hi = 0.5, float(np.asarray(gt["real"]["t"])[-1])
    real_t = np.asarray(gt["real"]["t"])
    real_z = np.asarray(gt["real"]["z"])
    m = (real_t >= t_lo) & (real_t <= t_hi)
    ax.plot(real_t[m], real_z[m], "k--", lw=ps.LW * S, label="Real robot",
            zorder=10)
    for sim in SIM_ORDER:
        sr, st = trajs[sim]
        sel = (st >= t_lo) & (st <= t_hi)
        z = sr[:, 2]
        bl = (st >= t_lo) & (st <= 2.0)
        baseline = float(np.mean(z[bl])) if bl.any() else 0.0
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(st[sel], z[sel] - baseline, "-", color=SIM_COLORS[sim],
                lw=0.8 * ps.LW * S, label=_display(sim), zorder=zord)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"base $z$ rise (m)")
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-0.07, 0.24)
    ax.grid(True, alpha=ps.GRID_MAJOR["alpha"], linewidth=ps.GRID_MAJOR["lw"] * S)


def panel_bar(ax):
    sims = ["Semi-Implicit", "MuJoCo", "Ostrich"]  # bottom-up -> Ostrich top
    xmax = 0.62
    y_pos = np.arange(len(sims))
    for y, sim in zip(y_pos, sims):
        err, dt = BAR[sim]
        ax.barh(y, err, color=SIM_COLORS[sim], height=0.5,
                edgecolor="black", linewidth=0.5 * ps.LW * S, zorder=3)
        ax.text(err + 0.02, y, rf" {err:.3f}  ($h={dt}$\,s)",
                va="center", ha="left", fontsize=ps.PRINT["annot"] * S)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_display(s) for s in sims])
    ax.set_xlabel(r"Combined pos. + yaw error (m)")
    ax.set_title("Accuracy over held-out runs (lower is better)",
                 pad=1.1 * ps.PRINT["title"] * S)
    ax.grid(True, axis="x", alpha=ps.GRID_MAJOR["alpha"],
            linewidth=ps.GRID_MAJOR["lw"] * S, zorder=0)
    ax.set_ylim(-0.5, len(sims) - 0.5)
    ax.set_xlim(0, xmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None,
                    help="JSON file to store/reuse the simulated trajectories, "
                         "so the layout can be iterated without re-simulating")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = load_gt(DATA_DIR / f"{SAMPLE}.json")
    trajs = load_trajs(gt, pathlib.Path(args.cache) if args.cache else None)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                             gridspec_kw={"width_ratios": [2, 2, 1.9],
                                          "wspace": 0.32})
    ax_xy, ax_z, ax_bar = axes
    panel_xy(ax_xy, trajs, gt)
    panel_z(ax_z, trajs, gt)
    panel_bar(ax_bar)

    handles, labels = ax_xy.get_legend_handles_labels()
    bbox_xy = ax_xy.get_position()
    bbox_z = ax_z.get_position()
    x_center = (bbox_xy.x0 + bbox_z.x1) / 2.0
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(x_center, 0.04),
               ncol=len(labels), fontsize=ps.PRINT["legend"] * S, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    out = pathlib.Path(args.out) if args.out else (
        RESULTS_DIR / "paper_panels" / "box_sim_to_real_c3.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    _px = plt.imread(out)
    _w, _h = _px.shape[1] / 300.0, _px.shape[0] / 300.0
    print(f"Wrote {out}  (cropped canvas {_w:.2f} x {_h:.2f} in -> printed "
          f"{PRINTED_IN:.2f} x {PRINTED_IN * _h / _w:.2f} in; DRAWN_IN is "
          f"{DRAWN_IN:.2f}, printed label {ps.PRINT['label'] * DRAWN_IN / _w:.2f} pt)")


if __name__ == "__main__":
    main()
