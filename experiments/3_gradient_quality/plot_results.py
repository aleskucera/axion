"""Plot Experiment 3: gradient-quality convergence.

Two-panel figure:
  Left : RMSE vs iteration  (per-step gradient quality — fair head-to-head)
  Right: RMSE vs wall-clock (time-to-solution — Axion's adjoint advantage)

Usage:
    python experiments/3_gradient_quality/plot_results.py
    python experiments/3_gradient_quality/plot_results.py --save results/gradient_quality.png
    GT_STEM=acceleration python experiments/3_gradient_quality/plot_results.py
"""
import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STYLES = {
    "Axion":         {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MuJoCo":        {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "MJX":           {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "TinyDiffSim":   {"color": "#607D8B", "marker": "D", "lw": 1.8, "zorder": 3},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 2},
}
LABELS = {
    "Axion":         r"\textbf{Axion}",
    "MJX":           "MJX",
    "MuJoCo":        "MuJoCo",
    "TinyDiffSim":   "TinyDiffSim",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = ["Axion", "MJX", "TinyDiffSim", "Semi-Implicit"]


def load_run(path):
    d = json.loads(path.read_text())
    rmse = np.array(d.get("rmse_m", []), dtype=float)
    times_ms = np.array(d.get("time_ms", []), dtype=float)
    # Running-best so the curves are monotonic (optimization progress story)
    rmse_best = np.minimum.accumulate(rmse) if len(rmse) else rmse
    iters = np.arange(len(rmse))
    cum_s = np.cumsum(times_ms) / 1000.0 if len(times_ms) else np.zeros_like(rmse)
    return {
        "sim": d.get("simulator", path.stem),
        "rmse": rmse,
        "rmse_best": rmse_best,
        "iters": iters,
        "cum_s": cum_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    gt_stem = os.environ.get("GT_STEM", "")
    suffix = f"_{gt_stem}" if gt_stem else ""

    # Load one run per simulator
    runs = {}
    name_map = {
        "axion": "Axion", "mjx": "MJX",
        "semi_implicit": "Semi-Implicit", "tinydiffsim": "TinyDiffSim",
    }
    for stem, sim_name in name_map.items():
        path = RESULTS_DIR / f"{stem}{suffix}.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        runs[sim_name] = load_run(path)

    if not runs:
        print("No results found. Run run_experiment.sh first.")
        return

    fig, (ax_it, ax_wc) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    fig.subplots_adjust(wspace=0.30)

    for sim in SIM_ORDER:
        if sim not in runs:
            continue
        r = runs[sim]
        st = STYLES[sim]
        ax_it.plot(
            r["iters"], r["rmse_best"],
            color=st["color"], marker=st["marker"], linewidth=st["lw"],
            markersize=4, markevery=max(1, len(r["iters"]) // 15),
            label=LABELS[sim], zorder=st["zorder"],
        )
        ax_wc.plot(
            r["cum_s"], r["rmse_best"],
            color=st["color"], marker=st["marker"], linewidth=st["lw"],
            markersize=4, markevery=max(1, len(r["cum_s"]) // 15),
            label=LABELS[sim], zorder=st["zorder"],
        )

    ax_it.set_xlabel("Iteration")
    ax_it.set_ylabel("Running-best RMSE (m)")
    ax_it.grid(True, alpha=0.35, linewidth=0.6)

    ax_wc.set_xlabel("Wall-clock time (s)")
    ax_wc.set_ylabel("Running-best RMSE (m)")
    ax_wc.set_xscale("log")
    ax_wc.grid(True, which="both", alpha=0.35, linewidth=0.6)
    ax_wc.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    handles, labels = ax_it.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles), fontsize=12,
        frameon=False, columnspacing=1.5, handlelength=1.5,
    )

    # --- Save ---
    save_arg = args.save or f"gradient_quality{suffix}.png"
    out = RESULTS_DIR / save_arg if "/" not in save_arg else pathlib.Path(save_arg)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / f"gradient_quality{suffix}.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
