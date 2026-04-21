from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

MODEL_INFO = "MSEModel"
COMPARISON_CSV_PATH = Path(__file__).resolve().parent / "comparison_40k.csv"
DEFAULT_HDF5_PATH = (
    Path(__file__).resolve().parents[4]
    / "data/logs/AxioneEngineWithNeuralLambdas_example_2026-04-20_21-41-08.h5"
)
DEFAULT_LAMBDA_SLICE = slice(0, 21)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot real-vs-predicted next states and selected next lambdas "
            "from an AxioneEngineWithNeuralLambdas HDF5 log. "
            "If the file also contains lambda_activity and "
            "lambda_activity_ground_truth, a fourth figure compares predicted vs "
            "simulator activity labels (same lambda index slice as --lambda-start/--lambda-stop)."
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
    parser.add_argument(
        "--csv",
        type=Path,
        default=COMPARISON_CSV_PATH,
        help=(
            "Append one summary row to this CSV after a successful run "
            "(creates file with header if missing)."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not append to the comparison CSV.",
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


def _try_load_lambda_activity_pair(
    hdf5_path: Path, t_len: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load lambda_activity and lambda_activity_ground_truth if both exist under data/.
    Returns None if either dataset is missing. Validates shape (T, C) and T == t_len.
    """
    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f:
            return None
        group = h5f["data"]
        if "lambda_activity" not in group or "lambda_activity_ground_truth" not in group:
            return None
        pred = np.asarray(group["lambda_activity"][:]).squeeze()
        gt = np.asarray(group["lambda_activity_ground_truth"][:]).squeeze()

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            "lambda_activity and lambda_activity_ground_truth must be 2D (T, C) after squeeze; "
            f"got shapes {pred.shape} and {gt.shape}"
        )
    if pred.shape != gt.shape:
        raise ValueError(
            "lambda_activity and lambda_activity_ground_truth shape mismatch: "
            f"{pred.shape} vs {gt.shape}"
        )
    if pred.shape[0] != t_len:
        raise ValueError(
            "lambda_activity length does not match trajectory length: "
            f"{pred.shape[0]} vs {t_len}"
        )
    return pred, gt


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


def _plot_lambda_activity_labels(
    time_axis: np.ndarray,
    x_label: str,
    pred: np.ndarray,
    gt: np.ndarray,
    lambda_start: int,
    lambda_stop: int,
) -> None:
    total_channels = pred.shape[1]
    start = max(0, lambda_start)
    stop = min(total_channels, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda activity slice [{lambda_start}:{lambda_stop}] "
            f"for total {total_channels} channels"
        )

    selected = list(range(start, stop))
    n = len(selected)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"Lambda activity: predicted vs ground truth (indices {start}:{stop})"
    )

    for local_idx, ch_idx in enumerate(selected):
        ax = axes_arr[local_idx]
        ax.plot(
            time_axis,
            pred[:, ch_idx],
            label="Predicted activity",
            linewidth=1.6,
        )
        ax.plot(
            time_axis,
            gt[:, ch_idx],
            label="Simulator GT",
            linewidth=1.4,
            linestyle="--",
        )
        ax.set_title(f"channel[{ch_idx}]")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Label")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for extra_idx in range(n, len(axes_arr)):
        axes_arr[extra_idx].set_visible(False)

    fig.tight_layout()


def _compute_prediction_metrics(
    next_states: np.ndarray,
    predicted_next_states: np.ndarray,
    next_lambdas: np.ndarray,
    predicted_next_lambdas: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Returns (state_mae, lambda_mae, state_total_abs, lambda_total_abs,
             total_abs_error, total_squared_error).
    total_abs_error is sum of absolute errors over all state and lambda elements.
    total_squared_error is sum of squared errors over all state and lambda elements.
    """
    state_abs_err = np.abs(next_states - predicted_next_states)
    lambda_abs_err = np.abs(next_lambdas - predicted_next_lambdas)

    state_mae = float(np.mean(state_abs_err))
    lambda_mae = float(np.mean(lambda_abs_err))
    state_total = float(np.sum(state_abs_err))
    lambda_total = float(np.sum(lambda_abs_err))
    total_abs_error = state_total + lambda_total

    state_sq = np.sum((next_states - predicted_next_states) ** 2)
    lambda_sq = np.sum((next_lambdas - predicted_next_lambdas) ** 2)
    total_squared_error = float(state_sq + lambda_sq)

    return (
        state_mae,
        lambda_mae,
        state_total,
        lambda_total,
        total_abs_error,
        total_squared_error,
    )


def _append_comparison_csv(
    csv_path: Path,
    log_filename: str,
    model_info: str,
    trajectory_timesteps: int,
    state_mae: float,
    lambda_mae: float,
    total_abs_error: float,
    total_squared_error: float,
) -> None:
    fieldnames = (
        "log_file",
        "model_info",
        "trajectory_timesteps",
        "state_mae",
        "lambda_mae",
        "total_absolute_error",
        "total_squared_error",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "log_file": log_filename,
                "model_info": model_info,
                "trajectory_timesteps": trajectory_timesteps,
                "state_mae": state_mae,
                "lambda_mae": lambda_mae,
                "total_absolute_error": total_abs_error,
                "total_squared_error": total_squared_error,
            }
        )


def _plot_mae_summary(
    state_mae: float,
    lambda_mae: float,
    state_total: float,
    lambda_total: float,
) -> None:
    fig, (ax_mae, ax_total) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Overall Prediction Error Summary")

    mae_labels = ["State MAE", "Lambda MAE"]
    mae_values = [state_mae, lambda_mae]
    mae_bars = ax_mae.bar(mae_labels, mae_values, width=0.45)
    for bar, value in zip(mae_bars, mae_values):
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )
    ax_mae.set_ylabel("Mean Absolute Error")
    ax_mae.set_title("MAE")
    ax_mae.grid(True, axis="y", alpha=0.3)

    total_labels = ["State total acc. abs. error", "Lambda total acc. abs. error"]
    total_values = [state_total, lambda_total]
    total_bars = ax_total.bar(total_labels, total_values, width=0.45)
    for bar, value in zip(total_bars, total_values):
        ax_total.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    ax_total.set_ylabel("Total Accumulated Absolute Error")
    ax_total.set_title("Total Accumulated Absolute Error")
    ax_total.grid(True, axis="y", alpha=0.3)

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
    (
        state_mae,
        lambda_mae,
        state_total_abs,
        lambda_total_abs,
        total_abs_error,
        total_squared_error,
    ) = _compute_prediction_metrics(
        next_states,
        predicted_next_states,
        next_lambdas,
        predicted_next_lambdas,
    )
    _plot_mae_summary(state_mae, lambda_mae, state_total_abs, lambda_total_abs)
    if not args.no_csv:
        _append_comparison_csv(
            args.csv,
            log_filename=args.hdf5.name,
            model_info=MODEL_INFO,
            trajectory_timesteps=int(next_states.shape[0]),
            state_mae=state_mae,
            lambda_mae=lambda_mae,
            total_abs_error=total_abs_error,
            total_squared_error=total_squared_error,
        )
    activity_pair = _try_load_lambda_activity_pair(args.hdf5, next_states.shape[0])
    if activity_pair is not None:
        lambda_activity, lambda_activity_gt = activity_pair
        _plot_lambda_activity_labels(
            time_axis=time_axis,
            x_label=x_label,
            pred=lambda_activity,
            gt=lambda_activity_gt,
            lambda_start=args.lambda_start,
            lambda_stop=args.lambda_stop,
        )
    plt.show()


if __name__ == "__main__":
    main()
