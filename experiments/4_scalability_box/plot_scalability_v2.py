"""Revised scalability figure: throughput as the headline, memory as support.

Panel (a): optimization throughput (world-iterations/s) vs #worlds, every
engine at its best memory configuration (MJX with jax.checkpoint per step).
Panel (b): peak GPU memory (NVML) vs #worlds. OOM points marked.

    .venv/bin/python experiments/4_scalability_box/plot_scalability_v2.py
"""
import collections
import glob
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"

# palette matches Figs. 2-3: Ostrich blue, MuJoCo/MJX red, Semi-Implicit orange
SERIES = {
    "ostrich": ("Ostrich", "#2196F3", "-", "o"),
    "ostrich_ckpt": ("Ostrich + checkpoint", "#2196F3", ":", "v"),
    "mjx_ckpt_step": ("MJX + checkpoint", "#E91E63", "-", "s"),
    "mjx_ckpt_none": ("MJX plain BPTT", "#E91E63", ":", "P"),
    "semi_implicit": ("Semi-Implicit", "#FF9800", "-", "^"),
}
# last world count that FAILED with OOM (annotation target)
OOM = {"mjx_ckpt_none": 8, "mjx_ckpt_step": 8192, "semi_implicit": 1024, "ostrich": 16384}


def load():
    data = collections.defaultdict(dict)
    for f in glob.glob(str(RES / "*.json")):
        m = re.match(r".*/([a-z_]+?)_(\d+)\.json", f)
        if not m:
            continue
        name, w = m.group(1), int(m.group(2))
        d = json.load(open(f))
        data[name][w] = d
    return data


def main():
    data = load()
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.size": 16,
        "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 13, "axes.spines.top": False,
        "axes.spines.right": False})
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(8.2, 4.0))

    for key, (label, color, ls, mk) in SERIES.items():
        if key not in data:
            continue
        ws = sorted(data[key])
        thr = [w / (data[key][w]["median_time_ms"] / 1000.0) for w in ws]
        mss = 5.5 if key == "mjx_ckpt_none" else 3.5
        mew = dict(markeredgecolor="white", markeredgewidth=0.9) if key == "mjx_ckpt_none" else {}
        ax_t.plot(ws, thr, ls, c=color, marker=mk, ms=mss, lw=1.7, label=label, **mew)
        # memory: prefer NVML series file, else NVML field, else tracked
        mem_key = {"ostrich": "ostrich_nvml", "semi_implicit": "si_nvml"}.get(key, key)
        src = data.get(mem_key, data[key])
        wm = sorted(src)
        mem = [src[w].get("peak_gpu_mb_nvml") or src[w].get("peak_gpu_mb")
               for w in wm]
        ax_m.plot(wm, mem, ls, c=color, marker=mk, ms=mss, lw=1.7, label=label, **mew)

        if key in OOM:
            for ax, ys in ((ax_t, thr), (ax_m, mem)):
                ax.plot([OOM[key]], [ys[-1] if ax is ax_t else ys[-1]], "x",
                        c=color, ms=9, mew=2)
            # per-series label offsets so the three OOM labels stay clear of
            # curves and of each other
            _off = {"mjx_ckpt_none": (6, -18), "mjx_ckpt_step": (8, -18),
                    "semi_implicit": (-14, 10), "ostrich": (6, -20)}.get(key, (6, -16))
            ax_t.annotate("OOM", (OOM[key], thr[-1]), textcoords="offset points",
                          xytext=_off, fontsize=11, color=color)

    for ax in (ax_t, ax_m):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax_t.set_ylabel("throughput [world-iterations / s]")
    ax_t.set_title("(a) throughput (fwd+bwd)", fontsize=16)
    ax_m.set_ylabel("peak GPU memory [MB, NVML]")
    ax_m.set_title("(b) peak GPU memory", fontsize=16)
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
               frameon=False, columnspacing=1.2, handletextpad=0.5)

    fig.tight_layout(rect=(0, 0.16, 1, 0.98))
    out = RES / "scalability_v2_paper.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
