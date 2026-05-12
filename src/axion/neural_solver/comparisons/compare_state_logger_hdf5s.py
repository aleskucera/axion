from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "data" / "logs"

# #for INITIAL_STATE = (0.5, -0.3, 1.0, -2.0)
# LOGS = [
#     "AxionEngine_example_2026-05-12_17-57-17.h5",
#     "GPTEngine_example_2026-05-12_17-42-59.h5",
#     "GPTEngine_example_2026-05-12_18-18-08.h5", # with COM shifting
# ]

# for INITIAL_STATE = (-0.5704, 2.8907, -3.6530, -7.6918)
LOGS = [
    "AxionEngine_example_2026-05-12_18-33-26.h5",
    "GPTEngine_example_2026-05-12_18-34-39.h5"
]

COORD_LABELS = ["q0 (rad)", "q1 (rad)", "qd0 (rad/s)", "qd1 (rad/s)"]


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
            "Plot PendulumStateLogger HDF5 logs: four stacked time series "
            "(q0, q1, qd0, qd1) with simulation time on the x-axis."
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
    log_dir = _DEFAULT_LOG_DIR
    entries = args.hdf5_paths if args.hdf5_paths else LOGS
    paths = [resolve_hdf5_path(e, log_dir) for e in entries]

    series: list[tuple[np.ndarray, np.ndarray, str]] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"HDF5 not found: {p}")
        series.append(load_pendulum_state_log(p))

    t_min = min(float(t.min()) for t, _, _ in series)
    t_max = max(float(t.max()) for t, _, _ in series)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10),
        sharex="all",
        constrained_layout=True,
    )

    for idx, (time, states, label) in enumerate(series):
        color = cmap(idx % 10)
        for k in range(4):
            axes[k].plot(time, states[:, k], color=color, label=label, linewidth=1.2)

    for k, ax in enumerate(axes):
        ax.set_ylabel(COORD_LABELS[k])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("simulation time (s)")
    axes[0].set_xlim(t_min, t_max)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
