"""dt-vs-error figure from the c3 sweep (4 trajectories of the new SIV-A
dataset) — plateau-departure presentation.

Instead of an absolute usability threshold, each engine's plateau is the h
range where its combined error stays within DEPART_FACTOR of its own
small-h floor (the model-error floor, h-invariant). The plateau edge is
where integration error takes over. Dashed engine-colored floor segments
span each plateau; the arrow compares Ostrich's and MuJoCo's edges.

Usage:
    python experiments/2_dt_stability_box/plot_dt_sweep_c3.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"
DEPART_FACTOR = 2.0
CRASHED_MARKER_Y = 12.0

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ENGINES = {  # file key -> (label, color, marker, lw, zorder)
    "ostrich": (r"\textbf{Ostrich}", "#2196F3", "o", 2.0, 5),
    "mujoco": ("MuJoCo", "#E91E63", "s", 1.8, 4),
    "semi_implicit": ("Semi-Impl.", "#FF9800", "^", 1.8, 3),
}


SKIP = {}  # per-engine h values to exclude, e.g. {"ostrich": {0.005}}


def load(key):
    with open(RESULTS / f"dt_sweep_c3_{key}.json") as f:
        rows = json.load(f)["rows"]
    ok = [(r["h"], r["mean_combined"]) for r in rows
          if r["n_diverged"] == 0 and r["h"] not in SKIP.get(key, ())]
    crashed = [r["h"] for r in rows if r["n_diverged"] > 0]
    return sorted(ok), sorted(crashed)


def plateau_edge(ok):
    floor = min(e for _, e in ok)
    inside = [h for h, e in ok if e <= DEPART_FACTOR * floor]
    return floor, max(inside)


def main():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    edges = {}

    for key, (label, color, marker, lw, z) in ENGINES.items():
        ok, crashed = load(key)
        hs = [h for h, _ in ok]
        errs = [e for _, e in ok]
        ax.plot(hs, errs, color=color, marker=marker, linewidth=lw,
                markersize=6, label=label, zorder=z)

        floor, edge = plateau_edge(ok)
        edges[key] = edge
        # plateau-extent bracket below the curve (annotation layer, offset
        # from the data so it never hides markers)
        by = floor / 1.45
        ax.plot([hs[0], edge], [by, by], color=color, linewidth=1.6,
                alpha=0.9, zorder=2)
        for x in (hs[0], edge):
            ax.plot([x, x], [by / 1.12, by * 1.12], color=color,
                    linewidth=1.6, alpha=0.9, zorder=2)

        if crashed:
            xs = [hs[-1]] + crashed
            ys = [errs[-1]] + [CRASHED_MARKER_Y] * len(crashed)
            ax.plot(xs, ys, color=color, linestyle="--", linewidth=1.2,
                    alpha=0.55, zorder=2)
            ax.plot(crashed, [CRASHED_MARKER_Y] * len(crashed), "x",
                    color=color, markersize=10, markeredgewidth=2.2, zorder=6)

    # plateau-edge arrow: MuJoCo -> Ostrich
    mj, ax_edge = edges["mujoco"], edges["ostrich"]
    ratio = ax_edge / mj
    ax.set_ylim(0.026, 25)
    arrow_y = 0.045
    ax.annotate("", xy=(ax_edge, arrow_y), xytext=(mj, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3,
                                shrinkA=2, shrinkB=2), zorder=7)
    ax.text(np.sqrt(mj * ax_edge), arrow_y * 0.86,
            rf"$\sim{ratio:.0f}\times$ larger usable $h$",
            ha="center", va="top", fontsize=10, fontweight="bold", zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"timestep $h$ [s]")
    ax.set_ylabel(r"combined error [m]")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", ncol=1, frameon=False)
    fig.tight_layout()
    out = RESULTS / "dt_sweep_c3.png"
    fig.savefig(out, dpi=265)
    print(f"saved {out}, edges: {edges}, ratio {ratio:.1f}x")


if __name__ == "__main__":
    main()
