from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_HDF5_PATH = (
    Path(__file__).resolve().parents[4]
    / "data/logs/AxioneEngineWithNeuralLambdas_example_2026-04-17_16-16-55.h5"
)
DEFAULT_LAMBDA_SLICE = slice(0, 21)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot real-vs-predicted next states and selected next lambdas "
            "from an AxioneEngineWithNeuralLambdas HDF5 log."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=DEFAULT_HDF5_PATH,
        help="Path to AxioneEngineWithNeuralLambdas log (.h5).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional simulation time step. If omitted, x-axis is step index.",
    )
    parser.add_argument(
        "--lambda-start",
        type=int,
        default=DEFAULT_LAMBDA_SLICE.start,
        help="First lambda index to plot (inclusive).",
    )
    parser.add_argument(
        "--lambda-stop",
        type=int,
        default=DEFAULT_LAMBDA_SLICE.stop,
        help="Last lambda index to plot (exclusive).",
    )
    return parser.parse_args()


def _load_and_validate(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    required = (
        "next_states",
        "predicted_next_states",
        "next_lambdas",
        "predicted_next_lambdas",
    )

    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f:
            raise KeyError(f"Missing top-level group 'data' in {hdf5_path}")
        group = h5f["data"]
        missing = [key for key in required if key not in group]
        if missing:
            raise KeyError(
                f"Missing required dataset(s) in {hdf5_path}: {', '.join(missing)}"
            )

        next_states = np.asarray(group["next_states"][:]).squeeze()
        predicted_next_states = np.asarray(group["predicted_next_states"][:]).squeeze()
        next_lambdas = np.asarray(group["next_lambdas"][:]).squeeze()
        predicted_next_lambdas = np.asarray(group["predicted_next_lambdas"][:]).squeeze()

    if next_states.ndim != 2 or predicted_next_states.ndim != 2:
        raise ValueError(
            "State arrays must become shape (T, 4) after squeezing; "
            f"got {next_states.shape} and {predicted_next_states.shape}"
        )
    if next_lambdas.ndim != 2 or predicted_next_lambdas.ndim != 2:
        raise ValueError(
            "Lambda arrays must become shape (T, N) after squeezing; "
            f"got {next_lambdas.shape} and {predicted_next_lambdas.shape}"
        )
    if next_states.shape != predicted_next_states.shape:
        raise ValueError(
            "next_states and predicted_next_states shape mismatch: "
            f"{next_states.shape} vs {predicted_next_states.shape}"
        )
    if next_lambdas.shape != predicted_next_lambdas.shape:
        raise ValueError(
            "next_lambdas and predicted_next_lambdas shape mismatch: "
            f"{next_lambdas.shape} vs {predicted_next_lambdas.shape}"
        )
    if next_states.shape[0] != next_lambdas.shape[0]:
        raise ValueError(
            "State and lambda sequence lengths differ: "
            f"{next_states.shape[0]} vs {next_lambdas.shape[0]}"
        )
    if next_states.shape[1] != 4:
        raise ValueError(f"Expected 4 state dimensions, got {next_states.shape[1]}")

    return next_states, predicted_next_states, next_lambdas, predicted_next_lambdas


def _build_time_axis(length: int, dt: float | None) -> tuple[np.ndarray, str]:
    if dt is None:
        return np.arange(length), "Step"
    if dt <= 0:
        raise ValueError(f"--dt must be positive when provided, got {dt}")
    return np.arange(length, dtype=float) * dt, "Time [s]"


def _plot_states(time_axis: np.ndarray, x_label: str, real: np.ndarray, pred: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Next State: Simulator vs Neural Prediction")

    for idx, ax in enumerate(axes.flat):
        ax.plot(time_axis, real[:, idx], label="Simulator next state", linewidth=1.8)
        ax.plot(time_axis, pred[:, idx], label="Predicted next state", linewidth=1.6, linestyle="--")
        ax.set_title(f"state[{idx}]")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()


def _plot_lambdas(
    time_axis: np.ndarray,
    x_label: str,
    real: np.ndarray,
    pred: np.ndarray,
    lambda_start: int,
    lambda_stop: int,
) -> None:
    total_lambdas = real.shape[1]
    start = max(0, lambda_start)
    stop = min(total_lambdas, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda slice [{lambda_start}:{lambda_stop}] for total {total_lambdas} lambdas"
        )

    selected = list(range(start, stop))
    n = len(selected)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Next Lambdas: Simulator vs Neural Prediction (indices {start}:{stop})")

    for local_idx, lambda_idx in enumerate(selected):
        ax = axes_arr[local_idx]
        ax.plot(
            time_axis,
            real[:, lambda_idx],
            label="Simulator next lambda",
            linewidth=1.6,
        )
        ax.plot(
            time_axis,
            pred[:, lambda_idx],
            label="Predicted next lambda",
            linewidth=1.4,
            linestyle="--",
        )
        ax.set_title(f"lambda[{lambda_idx}]")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for extra_idx in range(n, len(axes_arr)):
        axes_arr[extra_idx].set_visible(False)

    fig.tight_layout()


def _plot_mae_summary(
    next_states: np.ndarray,
    predicted_next_states: np.ndarray,
    next_lambdas: np.ndarray,
    predicted_next_lambdas: np.ndarray,
) -> None:
    state_mae = float(np.mean(np.abs(next_states - predicted_next_states)))
    lambda_mae = float(np.mean(np.abs(next_lambdas - predicted_next_lambdas)))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["State prediction MAE", "Lambda prediction MAE"]
    values = [state_mae, lambda_mae]
    bars = ax.bar(labels, values, width=0.55)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Overall Prediction Error Summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()


def main() -> None:
    args = _parse_args()
    next_states, predicted_next_states, next_lambdas, predicted_next_lambdas = _load_and_validate(
        args.hdf5
    )
    time_axis, x_label = _build_time_axis(next_states.shape[0], args.dt)

    _plot_states(time_axis, x_label, next_states, predicted_next_states)
    _plot_lambdas(
        time_axis=time_axis,
        x_label=x_label,
        real=next_lambdas,
        pred=predicted_next_lambdas,
        lambda_start=args.lambda_start,
        lambda_stop=args.lambda_stop,
    )
    _plot_mae_summary(
        next_states=next_states,
        predicted_next_states=predicted_next_states,
        next_lambdas=next_lambdas,
        predicted_next_lambdas=predicted_next_lambdas,
    )
    plt.show()


if __name__ == "__main__":
    main()
