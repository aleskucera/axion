#!/usr/bin/env python3
"""
Analyze `data/engine_comparison*.hdf5`.

Computes, for each run, the total number of Newton iterations (sum of `iter_count`)
for selected engines and plots them as a bar chart.

Additionally, for each run, creates a line plot where:
- x-axis = simulation step index
- y-axis = iter_count
- one line per engine

Also creates temporal plots for hardcoded slices of:
- converged lambda components
- converged state components (from `converged_body_vel` flattened)

Example:
    python src/axion/neural_solver/comparisons/analyze_engine_test.py \
        --hdf5 data/engine_comparison_20260424_153000.hdf5 \
        --axion-group axion_engine \
        --hybrid-group hybrid_gpt_engine_neural_warm_start_forces \
        --repeated-group repeated_axion_engine
"""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_H5_PATH = REPO_ROOT / "data" / "engine_comparison_20260424_164345.hdf5"
LAMBDA_SLICE = (0, 10)
STATE_SLICE = (0, 2)


def _sorted_run_keys(h5_group: h5py.Group) -> list[str]:
    keys = [k for k in h5_group.keys() if k.startswith("run_")]
    keys.sort()
    return keys


def _extract_run_index(run_key: str) -> int:
    # Expected format: run_000, run_001, ...
    try:
        return int(run_key.split("_", 1)[1])
    except Exception:
        return -1


def _iter_count_series_for_run(run_group: h5py.Group) -> np.ndarray:
    if "iter_count" not in run_group:
        raise KeyError(f"Missing dataset 'iter_count' in group {run_group.name}")
    iters = np.asarray(run_group["iter_count"][:])
    # Expected shape is (num_steps, 1) -> squeeze to (num_steps,)
    return iters.squeeze()


def _total_iters_for_series(iters: np.ndarray) -> int:
    return int(np.sum(iters))


def _validate_engine_group_exists(
    f: h5py.File, engine_group_name: str, h5_path: pathlib.Path
) -> h5py.Group:
    if engine_group_name not in f:
        available = ", ".join(sorted(f.keys()))
        raise KeyError(
            f"Engine group '{engine_group_name}' not found in {h5_path}. "
            f"Top-level groups: {available}"
        )
    return f[engine_group_name]


def _get_dataset_or_fallback(run_group: h5py.Group, preferred: str, fallback: str) -> np.ndarray:
    if preferred in run_group:
        return np.asarray(run_group[preferred][:])
    if fallback in run_group:
        return np.asarray(run_group[fallback][:])
    raise KeyError(
        f"Neither '{preferred}' nor '{fallback}' exists in {run_group.name}. "
        f"Available datasets: {sorted(run_group.keys())}"
    )


def _flatten_feature_series(arr: np.ndarray) -> np.ndarray:
    # Convert (T, ...features...) -> (T, F)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot total Newton iter_count per run and iter_count vs step per run "
            "(for selected engine outputs)."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default=str(DEFAULT_H5_PATH),
        help=f"Path to engine comparison HDF5 (default: {DEFAULT_H5_PATH})",
    )
    parser.add_argument("--axion-group", type=str, default="axion_engine")
    parser.add_argument(
        "--hybrid-group",
        type=str,
        default="hybrid_gpt_engine_neural_warm_start_forces",
    )
    parser.add_argument(
        "--repeated-group", type=str, default="repeated_axion_engine"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="If set, save plot to this path (e.g. data/iter_totals.png).",
    )
    parser.add_argument(
        "--iter-plots-dir",
        type=str,
        default="",
        help="Directory to save per-run iter_count-vs-step plots "
        "(default: <hdf5_dir>/iter_count_vs_step_per_run/<YYYYMMDD_HHMMSS>).",
    )
    parser.add_argument(
        "--temporal-plots-dir",
        type=str,
        default="",
        help="Directory to save temporal lambda/state plots "
        "(default: <hdf5_dir>/temporal_variables_per_run/<YYYYMMDD_HHMMSS>).",
    )
    args = parser.parse_args()

    h5_path = pathlib.Path(args.hdf5)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.iter_plots_dir:
        iter_plots_dir = pathlib.Path(args.iter_plots_dir)
    else:
        iter_plots_dir = h5_path.parent / "iter_count_vs_step_per_run" / run_stamp
    iter_plots_dir.mkdir(parents=True, exist_ok=True)
    if args.temporal_plots_dir:
        temporal_plots_dir = pathlib.Path(args.temporal_plots_dir)
    else:
        temporal_plots_dir = h5_path.parent / "temporal_variables_per_run" / run_stamp
    temporal_plots_dir.mkdir(parents=True, exist_ok=True)

    engine_groups = [
        (args.axion_group, "axion_engine"),
        (args.hybrid_group, "hybrid_gpt_engine_neural_warm_start_forces"),
        (args.repeated_group, "repeated_axion_engine"),
    ]

    # Load iter_count series per run for each engine.
    # all_series[engine_group_name][run_idx] -> (num_steps,) ndarray
    all_series: dict[str, dict[int, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        for engine_group_name, _ in engine_groups:
            eng_grp = _validate_engine_group_exists(f, engine_group_name, h5_path)
            series_for_runs: dict[int, np.ndarray] = {}
            for run_key in _sorted_run_keys(eng_grp):
                run_idx = _extract_run_index(run_key)
                series_for_runs[run_idx] = _iter_count_series_for_run(
                    eng_grp[run_key]
                )
            all_series[engine_group_name] = series_for_runs

    run_sets = [set(s.keys()) for s in all_series.values()]
    run_indices = sorted(set.intersection(*run_sets)) if run_sets else []
    if not run_indices:
        available = {k: sorted(v.keys()) for k, v in all_series.items()}
        raise RuntimeError(
            "No overlapping run indices across selected engine groups. "
            f"Run indices per group: {available}"
        )

    # --- Bar plot of total iterations ---
    totals_by_engine: dict[str, list[int]] = {}
    for engine_group_name, _ in engine_groups:
        totals_by_engine[engine_group_name] = [
            _total_iters_for_series(all_series[engine_group_name][i])
            for i in run_indices
        ]

    x = np.arange(len(run_indices))
    engine_count = len(engine_groups)
    width = 0.9 / engine_count

    plt.figure(figsize=(12, 4.5))
    offsets = np.linspace(-0.45 + width / 2, 0.45 - width / 2, engine_count)
    for (engine_group_name, _), off in zip(engine_groups, offsets):
        plt.bar(
            x + off,
            totals_by_engine[engine_group_name],
            width=width,
            label=engine_group_name,
        )
    plt.xticks(x, [str(i) for i in run_indices], rotation=0)
    plt.ylabel("Total Newton iterations (sum iter_count)")
    plt.xlabel("Run index")
    plt.title(f"Total iter_count per run ({engine_count} engines)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
    else:
        plt.show()

    # --- Per-run iter_count vs simulation-step line plots ---
    for run_idx in run_indices:
        series_by_engine = {
            engine_group_name: np.ravel(all_series[engine_group_name][run_idx])
            for engine_group_name, _ in engine_groups
        }

        num_steps = int(
            min(arr.shape[0] for arr in series_by_engine.values())
        )
        steps = np.arange(num_steps)

        plt.figure(figsize=(12, 4.5))
        for engine_group_name, _ in engine_groups:
            arr = series_by_engine[engine_group_name]
            plt.plot(steps, arr[:num_steps], label=engine_group_name)
        plt.xlabel("Simulation step")
        plt.ylabel("iter_count")
        plt.title(f"iter_count vs step (run_{run_idx:03d})")
        plt.grid(True, axis="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = iter_plots_dir / f"iter_count_vs_step_run_{run_idx:03d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # --- Per-run temporal plots for lambda/state slices ---
    with h5py.File(h5_path, "r") as f:
        axion_grp = _validate_engine_group_exists(f, args.axion_group, h5_path)
        hybrid_grp = _validate_engine_group_exists(f, args.hybrid_group, h5_path)
        repeated_grp = _validate_engine_group_exists(f, args.repeated_group, h5_path)

        for run_idx in run_indices:
            run_key = f"run_{run_idx:03d}"
            axion_run = axion_grp[run_key]
            hybrid_run = hybrid_grp[run_key]
            repeated_run = repeated_grp[run_key]

            axion_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    axion_run, "converged_constr_force", "constr_force"
                )
            )
            hybrid_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    hybrid_run, "converged_constr_force", "constr_force"
                )
            )
            repeated_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    repeated_run, "converged_constr_force", "constr_force"
                )
            )
            hybrid_pred_lambda = None
            if "hybrid_predicted_next_lambdas" in hybrid_run:
                hybrid_pred_lambda = _flatten_feature_series(
                    np.asarray(hybrid_run["hybrid_predicted_next_lambdas"][:])
                )

            axion_state = _flatten_feature_series(
                _get_dataset_or_fallback(axion_run, "converged_body_vel", "body_vel")
            )
            hybrid_state = _flatten_feature_series(
                _get_dataset_or_fallback(hybrid_run, "converged_body_vel", "body_vel")
            )
            repeated_state = _flatten_feature_series(
                _get_dataset_or_fallback(
                    repeated_run, "converged_body_vel", "body_vel"
                )
            )
            # Hybrid neural state prediction in solver coordinates after FK warm start.
            hybrid_pred_state = _flatten_feature_series(
                _get_dataset_or_fallback(
                    hybrid_run, "hybrid_predicted_next_body_vel", "init_guess_body_vel"
                )
            )

            lambda_steps = min(
                axion_lambda.shape[0],
                hybrid_lambda.shape[0],
                repeated_lambda.shape[0],
                hybrid_pred_lambda.shape[0] if hybrid_pred_lambda is not None else np.inf,
            )
            state_steps = min(
                axion_state.shape[0],
                hybrid_state.shape[0],
                repeated_state.shape[0],
                hybrid_pred_state.shape[0],
            )
            lambda_hi = min(
                LAMBDA_SLICE[1],
                axion_lambda.shape[1],
                hybrid_lambda.shape[1],
                repeated_lambda.shape[1],
                hybrid_pred_lambda.shape[1] if hybrid_pred_lambda is not None else np.inf,
            )
            state_hi = min(
                STATE_SLICE[1],
                axion_state.shape[1],
                hybrid_state.shape[1],
                repeated_state.shape[1],
                hybrid_pred_state.shape[1],
            )

            steps_lambda = np.arange(int(lambda_steps))
            for idx in range(LAMBDA_SLICE[0], int(lambda_hi)):
                plt.figure(figsize=(12, 4.5))
                plt.plot(
                    steps_lambda, axion_lambda[: int(lambda_steps), idx], label="Axion converged"
                )
                plt.plot(
                    steps_lambda,
                    repeated_lambda[: int(lambda_steps), idx],
                    label="RepeatedAxion converged",
                )
                plt.plot(
                    steps_lambda,
                    hybrid_lambda[: int(lambda_steps), idx],
                    label="Hybrid converged",
                )
                if hybrid_pred_lambda is not None:
                    plt.plot(
                        steps_lambda,
                        hybrid_pred_lambda[: int(lambda_steps), idx],
                        label="Hybrid neural prediction",
                    )
                plt.xlabel("Simulation step")
                plt.ylabel(f"lambda[{idx}]")
                plt.title(f"Temporal lambda[{idx}] (run_{run_idx:03d})")
                plt.grid(True, axis="both", alpha=0.25)
                plt.legend()
                plt.tight_layout()
                out_path = temporal_plots_dir / f"lambda_{idx:03d}_run_{run_idx:03d}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}")

            steps_state = np.arange(int(state_steps))
            for idx in range(STATE_SLICE[0], int(state_hi)):
                plt.figure(figsize=(12, 4.5))
                plt.plot(
                    steps_state, axion_state[: int(state_steps), idx], label="Axion converged"
                )
                plt.plot(
                    steps_state,
                    repeated_state[: int(state_steps), idx],
                    label="RepeatedAxion converged",
                )
                plt.plot(
                    steps_state,
                    hybrid_state[: int(state_steps), idx],
                    label="Hybrid converged",
                )
                plt.plot(
                    steps_state,
                    hybrid_pred_state[: int(state_steps), idx],
                    label="Hybrid neural prediction",
                )
                plt.xlabel("Simulation step")
                plt.ylabel(f"state_vel_component[{idx}]")
                plt.title(f"Temporal state component[{idx}] (run_{run_idx:03d})")
                plt.grid(True, axis="both", alpha=0.25)
                plt.legend()
                plt.tight_layout()
                out_path = temporal_plots_dir / f"state_{idx:03d}_run_{run_idx:03d}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
