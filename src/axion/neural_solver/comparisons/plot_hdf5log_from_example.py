from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

NO_CONTACT_MODELS_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-24_14-01-52.h5"
]

MTL_JUMP_MODELS_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-06-11.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-06-39.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-07-02.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-07-23.h5",
]

CONTACT_MTL_MODELS_LAMBDA_REGR_ONLY_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-25-41.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-26-10.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-26-32.h5",
]

CONTACT_MTL_MODELS_CONDITIONED_LAMBDA_REGR_ONLY_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-53-39.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-04.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-26.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-49.h5",
]

NO_CONTACT_MODELS_INFO = [
    "pure mse, w_state = 500",
    "mse with lambda in log space, w_state = 2",
    "residual pure",
    "residual + mse on last timestep, w_state = 500",
    "residual + mse on whole window T, w_state = 500"
]

MTL_JUMP_MODELS_MODEL_INFOS = [
    "with 0.01*(1-y_{cls}) * SmoothL1(jump_pred, 0) term",
    "y_{cls} * SmoothL1(jump_pred, jump_gt)",
    "y_{cls} * MSE(jump_pred, jump_gt)",
    "y_{cls} * SmoothL1(jump_pred, jump_gt), 200 epoch",
]

CONTACT_MTL_MODELS_LAMBDA_REGR_ONLY_MODEL_INFOS = [
    "contact_mtl, asinh, mse",
    "contact_mtl, asinh + ouput normal, mse",
    "contact_mtl, no tranform, mse",
]

CONTACT_MTL_MODELS_CONDITIONED_LAMBDA_REGR_ONLY_MODEL_INFOS = [
    "contact_mtl, pure mse",
    "contact_mtl, asinh, mse",
    "contact_mtl, asinh + output normal, mse",
    "contact_mtl, asinh + output normal, mse, 1M dataset"
]

ID = 0
MODEL_INFO = NO_CONTACT_MODELS_INFO[ID]
COMPARISON_CSV_PATH =  None #Path(__file__).resolve().parent / "contact_mtl_conditioned_lambda_regr.csv" # None
DEFAULT_HDF5_PATH = Path(__file__).resolve().parents[4] / "data/logs" / NO_CONTACT_MODELS_HDF5_LOG_FILE_NAMES[ID]
DEFAULT_LAMBDA_SLICE = slice(0, 9)
ANALYZE_INCOMPLETE_MTL = False
ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY = False
ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY = False
# Must match `jump_target_scale` / MTLModel.jump_target_scale for the checkpoint that produced the log.
DEFAULT_JUMP_TARGET_SCALE = 100.0
SIM_COLOR = "tab:blue"
PRED_COLOR = "tab:orange"

def _parse_optional_path(value: str) -> Path | None:
    if value.lower() in {"none", "null"}:
        return None
    return Path(value)

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
        type=_parse_optional_path,
        default=COMPARISON_CSV_PATH,
        help=(
            "Append one summary row to this CSV after a successful run "
            "(creates file with header if missing). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not append to the comparison CSV.",
    )
    return parser.parse_args()


def _load_and_validate(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    required = [
        "next_states",
        "next_lambdas",
        "predicted_next_lambdas",
    ]
    require_predicted_states = not (
        ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
        or ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
    )
    if require_predicted_states:
        required.append("predicted_next_states")

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
        predicted_next_states = (
            np.asarray(group["predicted_next_states"][:]).squeeze()
            if "predicted_next_states" in group
            else None
        )
        next_lambdas = np.asarray(group["next_lambdas"][:]).squeeze()
        predicted_next_lambdas = np.asarray(group["predicted_next_lambdas"][:]).squeeze()

    if next_states.ndim != 2:
        raise ValueError(
            f"next_states must become shape (T, 4) after squeezing; got {next_states.shape}"
        )
    if predicted_next_states is not None and predicted_next_states.ndim != 2:
        raise ValueError(
            "predicted_next_states must become shape (T, 4) after squeezing; "
            f"got {predicted_next_states.shape}"
        )
    if next_lambdas.ndim != 2 or predicted_next_lambdas.ndim != 2:
        raise ValueError(
            "Lambda arrays must become shape (T, N) after squeezing; "
            f"got {next_lambdas.shape} and {predicted_next_lambdas.shape}"
        )
    if predicted_next_states is not None and next_states.shape != predicted_next_states.shape:
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


def _load_lambda_jump(hdf5_path: Path, t_len: int, n_lambda: int) -> np.ndarray:
    """Load data/lambda_jump and validate shape (T, N)."""
    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f or "lambda_jump" not in h5f["data"]:
            raise KeyError(
                f"Missing data/lambda_jump in {hdf5_path} (required for ANALYZE_INCOMPLETE_MTL)."
            )
        jump = np.asarray(h5f["data"]["lambda_jump"][:]).squeeze()

    if jump.ndim != 2:
        raise ValueError(
            "lambda_jump must be 2D (T, N) after squeeze; "
            f"got shape {jump.shape}"
        )
    if jump.shape[0] != t_len or jump.shape[1] != n_lambda:
        raise ValueError(
            "lambda_jump shape mismatch: "
            f"got {jump.shape}, expected ({t_len}, {n_lambda})"
        )
    return jump


def _reconstruct_incomplete_mtl_pred_lambdas(
    next_lambdas: np.ndarray,
    lambda_activity_pred: np.ndarray,
    lambda_jump: np.ndarray,
    jump_target_scale: float,
) -> np.ndarray:
    """
    Artificial next-lambda prediction for incomplete MTL logs:
    where predicted binary activity is true, use simulator next lambda + scaled neural jump;
    otherwise keep value as NaN so inactive predictions are not plotted.
    """
    if lambda_activity_pred.shape != next_lambdas.shape or lambda_jump.shape != next_lambdas.shape:
        raise ValueError(
            "Shape mismatch in incomplete-MTL reconstruction: "
            f"next_lambdas {next_lambdas.shape}, activity {lambda_activity_pred.shape}, "
            f"jump {lambda_jump.shape}"
        )
    active = (lambda_activity_pred >= 0.5)
    scaled_jump = lambda_jump.astype(np.float64, copy=False) * float(jump_target_scale)
    reconstructed = np.full_like(next_lambdas, np.nan, dtype=np.float64)
    reconstructed[active] = (
        next_lambdas.astype(np.float64, copy=False)[active] + scaled_jump[active]
    )
    return reconstructed


def _mask_predicted_lambdas_by_gt_activity(
    predicted_next_lambdas: np.ndarray,
    lambda_activity_ground_truth: np.ndarray,
) -> np.ndarray:
    """
    For conditioned lambda-regression analysis: keep predicted next lambda where the
    simulator GT activity label is active (>= 0.5); set to NaN elsewhere.
    """
    if predicted_next_lambdas.shape != lambda_activity_ground_truth.shape:
        raise ValueError(
            "Shape mismatch masking predictions by GT activity: "
            f"predicted_next_lambdas {predicted_next_lambdas.shape}, "
            f"lambda_activity_ground_truth {lambda_activity_ground_truth.shape}"
        )
    out = predicted_next_lambdas.astype(np.float64, copy=True)
    inactive = lambda_activity_ground_truth < 0.5
    out[inactive] = np.nan
    return out


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
        ax.plot(
            time_axis,
            real[:, idx],
            label="Simulator next state",
            linewidth=1.8,
            color=SIM_COLOR,
        )
        ax.plot(
            time_axis,
            pred[:, idx],
            label="Predicted next state",
            linewidth=1.6,
            linestyle="--",
            color=PRED_COLOR,
        )
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
    jump_raw: np.ndarray | None = None,
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
            color=SIM_COLOR,
        )
        pred_series = pred[:, lambda_idx]
        valid = np.isfinite(pred_series)
        if ANALYZE_INCOMPLETE_MTL:
            ax.scatter(
                time_axis[valid],
                pred_series[valid],
                label="Predicted next lambda (active only)",
                s=22.0,
                marker="o",
                color=PRED_COLOR,
            )
            if jump_raw is not None:
                for t_idx in np.where(valid)[0]:
                    gt_lambda = real[t_idx, lambda_idx]
                    scaled_jump = jump_raw[t_idx, lambda_idx] * DEFAULT_JUMP_TARGET_SCALE
                    ax.annotate(
                        f"{gt_lambda:.2f}+{scaled_jump:.2f}={pred_series[t_idx]:.2f}",
                        (time_axis[t_idx], pred_series[t_idx]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                        color="black",
                    )
        elif (
            ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
            and not ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
            and not ANALYZE_INCOMPLETE_MTL
        ):
            ax.scatter(
                time_axis[valid],
                pred_series[valid],
                label="Predicted next lambda (GT-active only)",
                s=22.0,
                marker="o",
                color=PRED_COLOR,
            )
        else:
            ax.plot(
                time_axis,
                pred_series,
                label="Predicted next lambda",
                linewidth=1.4,
                linestyle="--",
                color=PRED_COLOR,
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
            color=PRED_COLOR,
        )
        ax.plot(
            time_axis,
            gt[:, ch_idx],
            label="Simulator GT",
            linewidth=1.4,
            linestyle="--",
            color=SIM_COLOR,
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
    predicted_next_states: np.ndarray | None,
    next_lambdas: np.ndarray,
    predicted_next_lambdas: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Returns (state_mae, lambda_mae, state_total_abs, lambda_total_abs,
             total_abs_error, total_squared_error).
    total_abs_error is sum of absolute errors over all state and lambda elements.
    total_squared_error is sum of squared errors over all state and lambda elements.
    """
    if predicted_next_states is not None:
        state_abs_err = np.abs(next_states - predicted_next_states)
        state_mae = float(np.mean(state_abs_err))
        state_total = float(np.sum(state_abs_err))
        state_sq = np.sum((next_states - predicted_next_states) ** 2)
    else:
        state_mae = float("nan")
        state_total = 0.0
        state_sq = 0.0
    lambda_diff = next_lambdas - predicted_next_lambdas
    lambda_valid = np.isfinite(lambda_diff)
    lambda_abs_err = np.abs(lambda_diff)

    if np.any(lambda_valid):
        lambda_mae = float(np.mean(lambda_abs_err[lambda_valid]))
        lambda_total = float(np.sum(lambda_abs_err[lambda_valid]))
        lambda_sq = np.sum((lambda_diff[lambda_valid]) ** 2)
    else:
        lambda_mae = float("nan")
        lambda_total = 0.0
        lambda_sq = 0.0
    total_abs_error = state_total + lambda_total

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
    lambda_total_abs_error: float,
    total_abs_error: float,
    total_squared_error: float,
) -> None:
    fieldnames = (
        "log_file",
        "model_info",
        "trajectory_timesteps",
        "state_mae",
        "lambda_mae",
        "lambda_total_absolute_error",
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
                "lambda_total_absolute_error": lambda_total_abs_error,
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

    state_label = "State MAE" if not math.isnan(state_mae) else "State MAE (N/A)"
    mae_labels = [state_label, "Lambda MAE"]
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
    t_len = int(next_states.shape[0])
    activity_pair = _try_load_lambda_activity_pair(args.hdf5, t_len)
    lambda_jump_arr = None

    if ANALYZE_INCOMPLETE_MTL:
        if activity_pair is None:
            raise ValueError(
                "ANALYZE_INCOMPLETE_MTL is True but the HDF5 log is missing "
                "lambda_activity and/or lambda_activity_ground_truth under data/."
            )
        lambda_activity_pred, _ = activity_pair
        lambda_jump_arr = _load_lambda_jump(args.hdf5, t_len, next_lambdas.shape[1])
        predicted_next_lambdas = _reconstruct_incomplete_mtl_pred_lambdas(
            next_lambdas,
            lambda_activity_pred,
            lambda_jump_arr,
            DEFAULT_JUMP_TARGET_SCALE,
        )
    elif (
        ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
        and not ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
        and not ANALYZE_INCOMPLETE_MTL
    ):
        if activity_pair is None:
            raise ValueError(
                "ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY requires "
                "lambda_activity and lambda_activity_ground_truth under data/."
            )
        _, lambda_activity_gt = activity_pair
        predicted_next_lambdas = _mask_predicted_lambdas_by_gt_activity(
            predicted_next_lambdas,
            lambda_activity_gt,
        )

    time_axis, x_label = _build_time_axis(next_states.shape[0], args.dt)

    if predicted_next_states is not None:
        _plot_states(time_axis, x_label, next_states, predicted_next_states)
    _plot_lambdas(
        time_axis=time_axis,
        x_label=x_label,
        real=next_lambdas,
        pred=predicted_next_lambdas,
        lambda_start=args.lambda_start,
        lambda_stop=args.lambda_stop,
        jump_raw=lambda_jump_arr if ANALYZE_INCOMPLETE_MTL else None,
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
    if not args.no_csv and args.csv is not None:
        _append_comparison_csv(
            args.csv,
            log_filename=args.hdf5.name,
            model_info=MODEL_INFO,
            trajectory_timesteps=int(next_states.shape[0]),
            state_mae=state_mae,
            lambda_mae=lambda_mae,
            lambda_total_abs_error=lambda_total_abs,
            total_abs_error=total_abs_error,
            total_squared_error=total_squared_error,
        )
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
