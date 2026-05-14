from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "data" / "logs" / "multirollouts"

# Publication-style typography / layout (matches plot_hdf5log_from_example.py academic mode).
BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LINEWIDTH = 2.5
GRID_ALPHA = 0.3
MODE = "autoregressive"

LEGEND_AXION = "Axion simulator"
LEGEND_NEURAL = f"Neural prediction ({MODE})"

# #for INITIAL_STATE = (0.5, -0.3, 1.0, -2.0)
# LOGS = [
#     "AxionEngine_example_2026-05-12_17-57-17.h5",
#     "GPTEngine_example_2026-05-12_17-42-59.h5",
#     "GPTEngine_example_2026-05-12_18-18-08.h5", # with COM shifting
# ]

# # for INITIAL_STATE = (-0.5704, 2.8907, -3.6530, -7.6918)
# LOGS = [
#     "AxionEngine_example_2026-05-12_18-33-26.h5",   
#     "GPTEngine_example_2026-05-12_18-34-39.h5", # model mse 298
#     "GPTEngine_example_2026-05-12_22-46-27.h5",  # model mse 299
#     "WarmupGPTEngine_example_2026-05-12_23-03-33.h5", # model 299
#     "GPTEngine_example_2026-05-12_23-20-59.h5" # model 32
# ]

# for initial state INITIAL_STATE = (0.5, -0.3, 1.0, -2.0)
# LOGS = [
#     "AxionEngine_example_2026-05-14_15-13-26.h5",
#     "GPTEngine_example_2026-05-14_15-12-59.h5"
# ]

# with contacts and INITIAL_STATE = (0,0,0,0)
LOGS = [
    "AxionEngine_example_2026-05-16_20-34-19.h5",
    #"TeacherForcedGPTEngine_example_2026-05-16_20-32-45.h5" # best from sweep
    "GPTEngine_example_2026-05-16_20-31-57.h5"  # best from sweep
]

def _apply_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": AXES_LABELS_FONTSIZE,
            "axes.titlesize": AXES_LABELS_FONTSIZE + 1,
            "xtick.labelsize": AXES_TICKS_FONTSIZE,
            "ytick.labelsize": AXES_TICKS_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
        }
    )


def _format_x_label_for_plot(x_label: str) -> str:
    if x_label == "Step":
        return r"Time step $t$ [-]"
    if x_label == "Time [s]":
        return r"Time $t$ [s]"
    return x_label


def _state_ylabel(idx: int) -> str:
    return (
        r"$q_0$   [rad]",
        r"$q_1$   [rad]",
        r"$u_0$   [$\mathrm{rad}\cdot\mathrm{s}^{-1}$]",
        r"$u_1$   [$\mathrm{rad}\cdot\mathrm{s}^{-1}$]",
    )[idx]


def _plot_legend_label(source_label: str) -> str:
    """Map HDF5 script_name / file stem to fixed legend text."""
    s = source_label.lower()
    if "axionengine" in s:
        return LEGEND_AXION
    if "gptengine" in s:
        return LEGEND_NEURAL
    return source_label


def resolve_hdf5_path(entry: str | Path, log_dir: Path) -> Path:
    """Bare filenames resolve under ``data/logs/``; absolute or multi-part paths stay as-is."""
    p = Path(entry)
    if p.is_absolute() or len(p.parts) > 1:
        return p.resolve()
    return (log_dir / p.name).resolve()


def load_pendulum_state_log(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Missing top-level group 'data' in {path}")
        grp = f["data"]
        if "states" not in grp or "time" not in grp:
            raise KeyError(f"Expected data/states and data/time in {path}")
        states = np.asarray(grp["states"][:], dtype=np.float64)
        time = np.asarray(grp["time"][:], dtype=np.float64).ravel()
        label = path.stem
        if "script_name" in grp.attrs:
            label = str(grp.attrs["script_name"])

    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"states must have shape (T, 4); got {states.shape} in {path}")
    if time.shape[0] != states.shape[0]:
        raise ValueError(
            f"time length {time.shape[0]} != states rows {states.shape[0]} in {path}"
        )
    return time, states, label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PendulumStateLogger HDF5 logs as a 2×2 grid of time series "
            "(q0, q1, qd0, qd1) vs time step index (same x-axis convention as "
            "plot_hdf5log_from_example.py without --dt)."
        )
    )
    parser.add_argument(
        "hdf5_paths",
        nargs="*",
        type=str,
        help=(
            "HDF5 files to compare. If omitted, uses LOGS in this script. "
            "Bare filenames resolve under repo data/logs/."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the figure to this path (e.g. compare.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (use with --save).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _apply_matplotlib_style()

    log_dir = _DEFAULT_LOG_DIR
    entries = args.hdf5_paths if args.hdf5_paths else LOGS
    paths = [resolve_hdf5_path(e, log_dir) for e in entries]

    series: list[tuple[np.ndarray, np.ndarray, str]] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"HDF5 not found: {p}")
        series.append(load_pendulum_state_log(p))

    # Step index on x-axis (matches plot_hdf5log_from_example._build_time_axis when dt is None).
    max_steps = max(states.shape[0] for _, states, _ in series)
    x_step_min = 0.0
    x_step_max = float(max_steps - 1) if max_steps > 0 else 0.0

    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Comparison of next state values: Axion ground truth vs {MODE} rollout of neural network"
    )
    axes_flat = np.asarray(axes).ravel()
    x_disp = _format_x_label_for_plot("Step")
    for k in range(4):
        ax = axes_flat[k]
        for idx, (_time, states, source_label) in enumerate(series):
            color = cmap(idx % 10)
            n = states.shape[0]
            step_axis = np.arange(n, dtype=float)
            legend = _plot_legend_label(source_label)
            linestyle = "--" if legend == LEGEND_NEURAL else "-"
            ax.plot(
                step_axis,
                states[:, k],
                color=color,
                label=legend,
                linewidth=LINEWIDTH,
                linestyle=linestyle,
            )
        ax.set_ylabel(_state_ylabel(k))
        ax.set_xlabel(x_disp)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc="best")
    axes_flat[0].set_xlim(x_step_min, x_step_max)
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
