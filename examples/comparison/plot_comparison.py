"""Compare gradient-based optimization across differentiable simulators.

Reads benchmark results from examples/comparison/results/*.json and generates
a 2×2 figure: loss curves and per-iteration timing for ball throw and helhest.

Usage:
    # First collect data (one command per simulator):
    python examples/comparison/ball_throw/ball_throw_axion.py        --save examples/comparison/results/ball_throw_axion.json
    python examples/comparison/ball_throw/ball_throw_xpbd.py         --save examples/comparison/results/ball_throw_xpbd.json
    python examples/comparison/ball_throw/ball_throw_featherstone.py --save examples/comparison/results/ball_throw_featherstone.json
    python examples/comparison/ball_throw/ball_throw_mjx.py          --save examples/comparison/results/ball_throw_mjx.json
    julia examples/comparison/ball_throw/ball_throw_dojo.jl          --save examples/comparison/results/ball_throw_dojo.json

    python examples/comparison/helhest/helhest_axion.py        --save examples/comparison/results/helhest_axion.json
    python examples/comparison/helhest/helhest_xpbd.py         --save examples/comparison/results/helhest_xpbd.json
    python examples/comparison/helhest/helhest_featherstone.py --save examples/comparison/results/helhest_featherstone.json
    python examples/comparison/helhest/helhest_mjx.py          --save examples/comparison/results/helhest_mjx.json
    # python examples/comparison/helhest/helhest_nimble.py        --save examples/comparison/results/helhest_nimble.json
    julia examples/comparison/helhest/helhest_dojo.jl          --save examples/comparison/results/helhest_dojo.json

    # Then plot:
    python examples/comparison/plot_comparison.py
    python examples/comparison/plot_comparison.py --show   # open interactive window
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

STYLES = {
    "Axion": {"color": "#2196F3", "marker": "o", "label": "Axion (Warp, implicit AD)"},
    "XPBD": {"color": "#FF9800", "marker": "v", "label": "XPBD (Warp, BPTT)"},
    "Featherstone": {"color": "#9C27B0", "marker": "D", "label": "Featherstone (Warp, BPTT)"},
    "MJX": {"color": "#FF5722", "marker": "s", "label": "MJX (JAX, forward-mode AD)"},
    "Dojo": {"color": "#4CAF50", "marker": "^", "label": "Dojo (Julia, BPTT)"},
    "Nimble": {"color": "#795548", "marker": "P", "label": "Nimble (DART, implicit AD)"},
}


def load_results(problem: str) -> dict:
    results = {}
    for path in sorted(RESULTS_DIR.glob(f"{problem}_*.json")):
        data = json.loads(path.read_text())
        sim = data.get("simulator", path.stem.replace(f"{problem}_", "").capitalize())
        results[sim] = data
    return results


def plot_loss(ax, results: dict, title: str):
    for sim, data in sorted(results.items()):
        style = STYLES.get(sim, {"color": "gray", "marker": ".", "label": sim})
        iters = data["iterations"]
        loss = data["loss"]
        every = max(1, len(iters) // 8)
        ax.semilogy(
            iters,
            loss,
            color=style["color"],
            marker=style["marker"],
            markevery=every,
            markersize=5,
            linewidth=2,
            label=style["label"],
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_timing(ax, results: dict, title: str):
    sims = sorted(results.keys())
    colors = [STYLES.get(s, {"color": "gray"})["color"] for s in sims]
    labels = [STYLES.get(s, {"label": s})["label"] for s in sims]

    # Exclude first iteration (compilation / warmup)
    medians = []
    p25, p75 = [], []
    for sim in sims:
        t = np.array(results[sim]["time_ms"])
        t = t[1:] if len(t) > 1 else t  # drop warmup
        medians.append(np.median(t))
        p25.append(np.percentile(t, 25))
        p75.append(np.percentile(t, 75))

    x = np.arange(len(sims))
    bars = ax.bar(x, medians, color=colors, width=0.5, zorder=3)
    ax.errorbar(
        x,
        medians,
        yerr=[np.array(medians) - np.array(p25), np.array(p75) - np.array(medians)],
        fmt="none",
        color="black",
        capsize=5,
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("Time per iteration (ms, median ± IQR)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    for bar, val in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR), help="Directory containing result JSON files"
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    ball_results = load_results("ball_throw")
    helhest_results = load_results("helhest")

    if not ball_results and not helhest_results:
        print("No results found. Run the benchmark scripts first (see module docstring).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Differentiable Simulator Comparison", fontsize=14, fontweight="bold")

    if ball_results:
        plot_loss(axes[0, 0], ball_results, "Ball Throw — Loss vs Iterations")
        plot_timing(axes[1, 0], ball_results, "Ball Throw — Time per Iteration")
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No ball_throw results",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
            color="gray",
        )
        axes[1, 0].text(
            0.5,
            0.5,
            "No ball_throw results",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            color="gray",
        )

    if helhest_results:
        plot_loss(axes[0, 1], helhest_results, "Helhest — Loss vs Iterations")
        plot_timing(axes[1, 1], helhest_results, "Helhest — Time per Iteration")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No helhest results",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            color="gray",
        )
        axes[1, 1].text(
            0.5,
            0.5,
            "No helhest results",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            color="gray",
        )

    plt.tight_layout()

    out = results_dir / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
