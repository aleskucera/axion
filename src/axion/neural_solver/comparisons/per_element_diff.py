#!/usr/bin/env python3
"""
Per-element absolute difference between HybridGPTEngine initial guesses and
AxionEngine converged values.

For each of (body_pose, body_vel, constr_force), the script:
  1. Loads init_guess_* from HybridGPTEngine (warm-start OFF and ON) and the
     converged value from AxionEngine for every simulation step.
  2. Flattens trailing dimensions so each step becomes a 1-D vector of elements.
  3. Computes the absolute difference |init_guess[t,j] - final[t,j]| per step.
  4. Averages that absolute difference across all steps in the run.
  5. Plots element index vs mean absolute difference, overlaying both Hybrid
     variants in distinct colors, one figure per run.
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_H5_PATH = REPO_ROOT / "data" / "engine_comparison.hdf5"

DATASET_PAIRS = [
    ("init_guess_body_pose", "body_pose", "body_pose"),
    ("init_guess_body_vel", "body_vel", "body_vel"),
    ("init_guess_constr_force", "constr_force", "constr_force"),
]

def _sorted_run_keys(h5_group: h5py.Group) -> list[str]:
    keys = [k for k in h5_group.keys() if k.startswith("run_")]
    keys.sort()
    return keys


def _mean_abs_diff(init_guess: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Mean absolute difference per element, averaged over the step axis.

    Parameters
    ----------
    init_guess, final : arrays of shape ``(num_steps, 1, ...)``

    Returns
    -------
    1-D array of length ``prod(trailing_dims)``.
    """
    num_steps = init_guess.shape[0]
    ig = init_guess.reshape(num_steps, -1)
    fn = final.reshape(num_steps, -1)
    return np.abs(ig - fn).mean(axis=0)


def _plot_run(
    run_key: str,
    hybrid_off_run: h5py.Group,
    hybrid_on_run: h5py.Group,
    axion_run: h5py.Group,
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for ax, (ig_name, final_name, label) in zip(axes, DATASET_PAIRS):
        ig_off = np.asarray(hybrid_off_run[ig_name][:])
        ig_on = np.asarray(hybrid_on_run[ig_name][:])
        fn = np.asarray(axion_run[final_name][:])

        diff_off = _mean_abs_diff(ig_off, fn)
        diff_on = _mean_abs_diff(ig_on, fn)

        num_elements = diff_off.shape[0]
        x = np.arange(num_elements)

        if num_elements <= 20:
            width = 0.45
            ax.bar(
                x - width / 2,
                diff_off,
                width=width,
                alpha=0.9,
                color="tab:blue",
                label="hybrid warm-start OFF" if ax is axes[0] else None,
            )
            ax.bar(
                x + width / 2,
                diff_on,
                width=width,
                alpha=0.9,
                color="tab:orange",
                label="hybrid warm-start ON" if ax is axes[0] else None,
            )
            ax.set_xticks(x)
        else:
            # For larger vectors, keep grouped bars (side-by-side) so the colors
            # remain visually separable.
            width = 0.45
            ax.bar(
                x - width / 2,
                diff_off,
                width=width,
                linewidth=0.8,
                alpha=0.95,
                color="tab:blue",
                label="hybrid warm-start OFF" if ax is axes[0] else None,
            )
            ax.bar(
                x + width / 2,
                diff_on,
                width=width,
                linewidth=0.8,
                alpha=0.95,
                color="tab:orange",
                label="hybrid warm-start ON" if ax is axes[0] else None,
            )

        ax.set_xlabel("Element index")
        ax.set_ylabel("Mean absolute difference")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Per-element mean |init_guess - converged|: Hybrid (warm-start OFF/ON) vs AxionEngine"
        f"  ({run_key})",
        fontsize=13,
    )
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-element mean absolute difference between HybridGPTEngine "
            "initial guesses (warm-start OFF/ON) and AxionEngine converged values."
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
        "--hybrid-off-group",
        type=str,
        default="hybrid_gpt_engine_use_warm_start_forces_false",
        help="Hybrid group name for warm-start OFF.",
    )
    parser.add_argument(
        "--hybrid-on-group",
        type=str,
        default="hybrid_gpt_engine_use_warm_start_forces_true",
        help="Hybrid group name for warm-start ON.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=15,
        help="Number of runs to plot (default: 3, from the beginning).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save plots (default: <hdf5_dir>/per_element_diff).",
    )
    args = parser.parse_args()

    h5_path = pathlib.Path(args.hdf5)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = h5_path.parent / "per_element_diff"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        hybrid_off_grp = f[args.hybrid_off_group]
        hybrid_on_grp = f[args.hybrid_on_group]
        axion_grp = f[args.axion_group]

        # Plot only run groups that exist in all three required outputs.
        off_keys = set(_sorted_run_keys(hybrid_off_grp))
        on_keys = set(_sorted_run_keys(hybrid_on_grp))
        ax_keys = set(_sorted_run_keys(axion_grp))
        common_keys = sorted(off_keys & on_keys & ax_keys)[: args.num_runs]

        run_keys = common_keys
        for run_key in run_keys:
            out_path = output_dir / f"per_element_diff_{run_key}.png"
            _plot_run(
                run_key,
                hybrid_off_grp[run_key],
                hybrid_on_grp[run_key],
                axion_grp[run_key],
                out_path,
            )


if __name__ == "__main__":
    main()
