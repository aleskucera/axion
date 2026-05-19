from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axion.neural_solver.comparisons.plot_hdf5log_from_example import _load_and_validate
from axion.neural_solver.train.trained_models.selected_trained_models import (
    CONTACT_MODELS,
)

# ---------------------------------------------------------------------------
# USER-FACING KNOBS - edit these
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]

# Folder containing AxionEngineWithNeuralLambdas *.h5 recordings.
# Filenames must match `run_AxionEngineNeuralLambdas_multiple_times.py`:
#   {run_idx:02d}_{_safe_model_log_stem(model)}.h5  with run_idx starting at 1.
LOG_DIR = REPO_ROOT / "data" / "logs" / "axionEngineWithNeuralLambdasComparison" / "run2"

# One slot per entry, in this order, on the x-axis — same list/order as the batch
# runner's NEURAL_MODELS so every model is shown (missing logs → no bar for that slot).
NEURAL_MODELS = CONTACT_MODELS

# Figure 1: total lambda L1 trajectory error (one bar per model).
OUTPUT_PATH_TOTAL: Path | None = (
    REPO_ROOT
    / "data"
    / "logs"
    / "axionEngineWithNeuralLambdasComparison"
    / "multi_model_lambda_total.png"
)

# Figure 2: same metrics split by canonical λ channel groups (three bars per model).
OUTPUT_PATH_BY_CHANNEL: Path | None = (
    REPO_ROOT
    / "data"
    / "logs"
    / "axionEngineWithNeuralLambdasComparison"
    / "multi_model_lambda_by_channel.png"
)

# If True, call plt.show() (blocking). If False, close figures after saving.
SHOW_FIG = True

FIGURE_DPI = 150

# Zeros / NaNs are clipped for plotting only (log y-axis).
Y_LOG_FLOOR = 1e-12

# Match plot_error_from_multirollouts.py (typography / grid).
BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 1
LINEWIDTH = 2.5  # For any line plots if added later.
GRID_ALPHA = 0.3

# Bar widths: figure 1 (total λ) vs figure 2 (grouped channels) use separate scales.
# Figure 2 keeps the narrower grouped columns; only the total-λ plot uses the larger scale.
BAR_WIDTH_SCALE_TOTAL = 5.0
BAR_TOTAL_MAX_W = 0.94
BAR_WIDTH_SCALE_BY_CHANNEL = 2.5
BAR_BY_CHANNEL_MAX_W = 0.88 / 3.0

# Y-axis label for both figures (log-scaled values).
Y_LABEL_LAMBDA_ERROR = (
    r"Cumulated total $\lambda$ error "
    r"($\log_{10}$)"
)

# Grouping for figure 2 is chosen from λ width via `_lambda_group_slices`:
#   24-λ: joint | ctrl(2) | contact;   22-λ: joint | normal | friction (no ctrl in vector).

# Requires plot_hdf5log_from_example module globals ANALYZE_* to stay False so
# predicted_next_states are present in the logs (same as comparison rollouts).
# ---------------------------------------------------------------------------


def _apply_plot_style() -> None:
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


def _safe_model_log_stem(model_path: Path) -> str:
    """Same stem helper as examples/double_pendulum/run_AxionEngineNeuralLambdas_multiple_times.py."""
    return "__".join(Path(model_path).parts).replace(" ", "_").replace("/", "__")


def _lambda_group_slices(
    n_ch: int,
) -> tuple[slice, slice, slice] | None:
    """
    Return (joint, contact_normal, friction) slices for known λ widths.

    - 24-λ (full Pendulum logger): joint 0–9, normal 12–15, friction 16–23.
    - 22-λ (compact, no control block): joint 0–9, normal 10–13, friction 14–21.
    """
    if n_ch >= 24:
        return (
            slice(0, 10),
            slice(12, 16),
            slice(16, 24),
        )
    if n_ch == 22:
        return (
            slice(0, 10),
            slice(10, 14),
            slice(14, 22),
        )
    return None


def _lambda_l1_sum(
    next_lambdas: np.ndarray,
    pred_lambdas: np.ndarray,
    channel_slice: slice | None = None,
) -> float:
    """Sum of absolute errors over time and selected channels (finite diff entries only)."""
    if channel_slice is not None:
        gt = next_lambdas[:, channel_slice]
        pr = pred_lambdas[:, channel_slice]
    else:
        gt = next_lambdas
        pr = pred_lambdas
    diff = gt - pr
    valid = np.isfinite(diff)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(np.abs(diff[valid])))


def _log_floor_from_values(*arrays: np.ndarray) -> float:
    parts = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()
        parts.append(a[np.isfinite(a) & (a > 0)])
    positive = np.concatenate(parts) if parts else np.array([], dtype=float)
    if positive.size:
        return min(Y_LOG_FLOOR, float(np.min(positive) * 1e-3))
    return Y_LOG_FLOOR


def _clip_positive_for_log(values: np.ndarray, floor: float) -> np.ndarray:
    """Finite missing (NaN) stays NaN (no bar). Non-positive finite → floor for log plot."""
    a = np.asarray(values, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask_finite = np.isfinite(a)
    mask_pos = mask_finite & (a > 0)
    mask_zero_or_neg = mask_finite & (a <= 0)
    out[mask_pos] = a[mask_pos]
    out[mask_zero_or_neg] = floor
    return out


def _model_axis_labels(n: int) -> list[str]:
    """Model A, Model B, … Model Z, then Model 27, Model 28, …"""
    out: list[str] = []
    for i in range(n):
        if i < 26:
            out.append(f"Model {chr(ord('A') + i)}")
        else:
            out.append(f"Model {i + 1}")
    return out


def main() -> None:
    log_dir = LOG_DIR.expanduser().resolve()
    if not log_dir.is_dir():
        print(f"ERROR: not a directory: {log_dir}", file=sys.stderr)
        return

    h5_paths = sorted(p for p in log_dir.glob("*.h5") if p.is_file())
    if not h5_paths:
        print(f"WARNING: no .h5 files under {log_dir}", file=sys.stderr)

    n_models = len(NEURAL_MODELS)
    lambda_totals: list[float] = []
    joint_errs: list[float] = []
    normal_errs: list[float] = []
    friction_errs: list[float] = []

    for run_idx, neural_model in enumerate(NEURAL_MODELS, start=1):
        mp = Path(neural_model)
        stem = f"{run_idx:02d}_{_safe_model_log_stem(mp)}"
        h5_path = log_dir / f"{stem}.h5"

        if not h5_path.is_file():
            print(
                f"MISSING log for model [{run_idx}/{n_models}] {mp.as_posix()}: "
                f"expected {h5_path.name}",
                file=sys.stderr,
            )
            lambda_totals.append(float("nan"))
            joint_errs.append(float("nan"))
            normal_errs.append(float("nan"))
            friction_errs.append(float("nan"))
            continue

        try:
            _next_states, pred_states, next_lambdas, pred_lambdas = _load_and_validate(
                h5_path
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"SKIP {h5_path.name}: {exc}", file=sys.stderr)
            lambda_totals.append(float("nan"))
            joint_errs.append(float("nan"))
            normal_errs.append(float("nan"))
            friction_errs.append(float("nan"))
            continue

        if pred_states is None:
            print(
                f"SKIP {h5_path.name}: missing predicted_next_states",
                file=sys.stderr,
            )
            lambda_totals.append(float("nan"))
            joint_errs.append(float("nan"))
            normal_errs.append(float("nan"))
            friction_errs.append(float("nan"))
            continue
        if pred_lambdas is None:
            print(
                f"SKIP {h5_path.name}: missing predicted_next_lambdas",
                file=sys.stderr,
            )
            lambda_totals.append(float("nan"))
            joint_errs.append(float("nan"))
            normal_errs.append(float("nan"))
            friction_errs.append(float("nan"))
            continue

        n_ch = int(next_lambdas.shape[1])

        lambda_totals.append(_lambda_l1_sum(next_lambdas, pred_lambdas, None))

        groups = _lambda_group_slices(n_ch)
        if groups is None:
            print(
                f"NOTE {h5_path.name}: λ width {n_ch} — total error plotted; "
                "channel breakdown only supported for 22 or ≥24 channels (skipping groups).",
                file=sys.stderr,
            )
            joint_errs.append(float("nan"))
            normal_errs.append(float("nan"))
            friction_errs.append(float("nan"))
        else:
            sj, sn, sf = groups
            joint_errs.append(_lambda_l1_sum(next_lambdas, pred_lambdas, sj))
            normal_errs.append(_lambda_l1_sum(next_lambdas, pred_lambdas, sn))
            friction_errs.append(_lambda_l1_sum(next_lambdas, pred_lambdas, sf))

    if n_models == 0:
        print("ERROR: NEURAL_MODELS is empty.", file=sys.stderr)
        return

    any_finite = np.any(np.isfinite(np.asarray(lambda_totals, dtype=float)))
    if not any_finite:
        print("ERROR: no valid recordings to plot (all models missing or skipped).", file=sys.stderr)
        return

    _apply_plot_style()

    n = n_models
    labels = _model_axis_labels(n)
    x = np.arange(n, dtype=float)

    lambda_arr = np.asarray(lambda_totals, dtype=float)
    joint_arr = np.asarray(joint_errs, dtype=float)
    normal_arr = np.asarray(normal_errs, dtype=float)
    friction_arr = np.asarray(friction_errs, dtype=float)

    # --- Figure 1: total lambda error ---
    floor_total = _log_floor_from_values(lambda_arr)
    lambda_plot = _clip_positive_for_log(lambda_arr, floor_total)
    width_total_base = min(0.65, 0.8 / max(n, 1))
    width_total = min(BAR_TOTAL_MAX_W, BAR_WIDTH_SCALE_TOTAL * width_total_base)
    fig1, ax1 = plt.subplots(figsize=(max(8.0, 0.45 * n + 2.0), 5.5))
    ax1.bar(
        x,
        lambda_plot,
        width_total,
        color="tab:orange",
        zorder=2,
    )
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel(Y_LABEL_LAMBDA_ERROR)
    ax1.set_title(
        r"Comparison of total $\lambda$ prediction error across multiple neural models"
    )
    ax1.grid(True, axis="y", which="both", alpha=GRID_ALPHA, linestyle=":")
    fig1.tight_layout()

    if OUTPUT_PATH_TOTAL is not None:
        out1 = OUTPUT_PATH_TOTAL.expanduser().resolve()
        out1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out1, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved figure: {out1}")

    # --- Figure 2: joint / contact normal / friction ---
    floor_ch = _log_floor_from_values(joint_arr, normal_arr, friction_arr)
    jp = _clip_positive_for_log(joint_arr, floor_ch)
    normal_plot = _clip_positive_for_log(normal_arr, floor_ch)
    fp = _clip_positive_for_log(friction_arr, floor_ch)

    width_group_base = min(0.22, 0.65 / max(n, 1))
    # Three bars at x-w, x, x+w → outer span 3w; keep < ~0.88 category width.
    w = min(BAR_WIDTH_SCALE_BY_CHANNEL * width_group_base, BAR_BY_CHANNEL_MAX_W)
    fig2, ax2 = plt.subplots(figsize=(max(9.0, 0.55 * n + 2.0), 5.5))
    ax2.bar(
        x - w,
        jp,
        w,
        label="Joint",
        color="tab:blue",
        zorder=2,
    )
    ax2.bar(
        x,
        normal_plot,
        w,
        label="Contact normal",
        color="tab:green",
        zorder=2,
    )
    ax2.bar(
        x + w,
        fp,
        w,
        label="Friction",
        color="tab:red",
        zorder=2,
    )
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel(Y_LABEL_LAMBDA_ERROR)
    ax2.set_title(
        r"Comparison of total $\lambda$ prediction error across multiple neural models"

    )
    ax2.grid(True, axis="y", which="both", alpha=GRID_ALPHA, linestyle=":")
    ax2.legend(loc="upper right")
    fig2.tight_layout()

    if OUTPUT_PATH_BY_CHANNEL is not None:
        out2 = OUTPUT_PATH_BY_CHANNEL.expanduser().resolve()
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out2, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved figure: {out2}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


if __name__ == "__main__":
    main()
