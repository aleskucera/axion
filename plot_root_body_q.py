#!/usr/bin/env python3
"""
Script to plot comparison of root_body_q from NeRD env vs axion (NerdEngine).
- pendulum_nerdEnv_root_body_q.csv: root body pose from original NeRD environment.
- pendulum_root_body_q_NerdEngine.csv: root body pose from axion NerdEngine.

Loads both CSVs and plots comparison of all root_body_q columns (NeRD env vs axion).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Number of first time steps to plot
NUM_STEPS = 50
# Ignore first data point of every time evolution (same as plot_contacts)
SKIP_FIRST_ROW = True

# NeRD env CSV columns -> axion CSV columns (same quantity)
NERD_TO_AXION_COLUMNS = [
    ("pos_x", "root_body_q_0"),
    ("pos_y", "root_body_q_1"),
    ("pos_z", "root_body_q_2"),
    ("quat_x", "root_body_q_3"),
    ("quat_y", "root_body_q_4"),
    ("quat_z", "root_body_q_5"),
    ("quat_w", "root_body_q_6"),
]


def _load_and_align(
    nerd_path: Path,
    axion_path: Path,
    num_steps: int,
    skip_first: bool,
):
    """Load both CSVs, trim to same length, optionally skip first row. Return (steps, nerd_df, axion_df)."""
    nerd_df = pd.read_csv(nerd_path).head(num_steps)
    axion_df = pd.read_csv(axion_path).head(num_steps)
    if skip_first:
        nerd_df = nerd_df.iloc[1:]
        axion_df = axion_df.iloc[1:]
    n = min(len(nerd_df), len(axion_df))
    nerd_df = nerd_df.iloc[:n]
    axion_df = axion_df.iloc[:n]
    steps = nerd_df["step"].values
    return steps, nerd_df, axion_df


def _column_pairs_to_plot(nerd_df, axion_df):
    """Return list of (display_name, nerd_col, axion_col) for all column pairs present in both CSVs."""
    pairs = []
    for nerd_col, axion_col in NERD_TO_AXION_COLUMNS:
        if nerd_col in nerd_df.columns and axion_col in axion_df.columns:
            pairs.append((nerd_col, nerd_col, axion_col))
    return pairs


def plot_root_body_q_comparison(
    nerd_env_csv_path: str,
    axion_csv_path: str,
    num_steps: int = NUM_STEPS,
    skip_first_row: bool = SKIP_FIRST_ROW,
):
    """
    Compare root_body_q from NeRD env and axion CSVs; plot all columns (NeRD vs axion).

    Args:
        nerd_env_csv_path: Path to pendulum_nerdEnv_root_body_q.csv (NeRD environment).
        axion_csv_path: Path to pendulum_root_body_q_NerdEngine.csv (axion NerdEngine).
        num_steps: Number of first time steps to use.
        skip_first_row: If True, ignore the first data point of each evolution.
    """
    nerd_path = Path(nerd_env_csv_path)
    axion_path = Path(axion_csv_path)

    if not nerd_path.exists():
        raise FileNotFoundError(f"NeRD env root_body_q CSV not found: {nerd_path}")
    if not axion_path.exists():
        raise FileNotFoundError(f"Axion root_body_q CSV not found: {axion_path}")

    steps, nerd_df, axion_df = _load_and_align(
        nerd_path, axion_path, num_steps, skip_first_row
    )
    column_pairs = _column_pairs_to_plot(nerd_df, axion_df)

    if not column_pairs:
        print("No matching columns found in both CSVs. Nothing to plot.")
        return

    n_sub = len(column_pairs)
    fig, axes = plt.subplots(n_sub, 1, figsize=(12, 2.5 * n_sub), sharex=True)
    if n_sub == 1:
        axes = [axes]

    for ax, (display_name, nerd_col, axion_col) in zip(axes, column_pairs):
        ax.plot(
            steps,
            nerd_df[nerd_col].values,
            label="NeRD env",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            steps,
            axion_df[axion_col].values,
            label="axion (NerdEngine)",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_ylabel(display_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle(
        "root_body_q comparison (NeRD env vs axion NerdEngine)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.show()
    plt.show()


if __name__ == "__main__":
    nerd_csv = "src/axion/core/pendulum_nerdEnv_root_body_q.csv"
    axion_csv = "src/axion/core/pendulum_root_body_q_NerdEngine.csv"

    plot_root_body_q_comparison(
        nerd_env_csv_path=nerd_csv,
        axion_csv_path=axion_csv,
        num_steps=NUM_STEPS,
        skip_first_row=SKIP_FIRST_ROW,
    )
