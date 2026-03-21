"""Plot scalability of Helhest optimization: time and memory vs number of worlds.

Usage:
    python examples/comparison/helhest_scalability/plot_results.py
    python examples/comparison/helhest_scalability/plot_results.py --show
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STYLES = {
    "Axion":      {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MJX-jacfwd": {"color": "#FF5722", "marker": "s", "lw": 1.5, "zorder": 3},
    "MJX-grad":   {"color": "#FF8A65", "marker": "^", "lw": 1.5, "zorder": 3},
}
LABELS = {
    "Axion":      r"\textbf{Axion}",
    "MJX-jacfwd": "MJX-jacfwd",
    "MJX-grad":   "MJX-grad",
}
SIM_ORDER = list(STYLES.keys())


def load_results() -> dict[str, dict[int, dict]]:
    """Returns {simulator: {num_worlds: data}}."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        sim = data.get("simulator")
        nw = data.get("num_worlds")
        if sim is None or nw is None:
            continue
        out.setdefault(sim, {})[nw] = data
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    all_results = load_results()
    if not all_results:
        print("No results found. Run run_sweep.sh first.")
        return

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.35)

    for sim in SIM_ORDER:
        if sim not in all_results:
            continue
        data_by_worlds = all_results[sim]
        worlds = sorted(data_by_worlds.keys())
        style = STYLES[sim]
        label = LABELS[sim]

        times = [data_by_worlds[n]["median_time_ms"] for n in worlds]
        mems  = [data_by_worlds[n]["peak_gpu_mb"] for n in worlds]

        ax_time.plot(worlds, times, color=style["color"], marker=style["marker"],
                     linewidth=style["lw"], markersize=5, label=label, zorder=style["zorder"])
        ax_mem.plot(worlds, mems, color=style["color"], marker=style["marker"],
                    linewidth=style["lw"], markersize=5, label=label, zorder=style["zorder"])

    for ax in (ax_time, ax_mem):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of worlds")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.2, linewidth=0.4)

    ax_time.set_ylabel("Median time per iteration (ms)")
    ax_time.set_title("Wall-clock time vs worlds", pad=4)

    ax_mem.set_ylabel("Peak GPU memory (MB)")
    ax_mem.set_title("GPU memory vs worlds", pad=4)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), frameon=False, columnspacing=1.5)

    out = RESULTS_DIR / "scalability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
