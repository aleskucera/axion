#!/usr/bin/env python3
"""
Script to plot temporal evolution of contact info from four CSV files.
- pendulum_contacts_NerdEngine.csv: contacts as passed into the predictor (engine-side).
- pendulum_model_inputs_contacts.csv: contact inputs as fed to the NeRD model (predictor-side).
- pendulum_nerdEnv_contacts.csv: contacts from the original NeRD environment.
- pendulum_nerdEnv_contacts_raw.csv: raw contacts from the original NeRD environment.

Each CSV is plotted in a separate figure with subplots per contact group.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# Number of first time steps to plot (settable parameter)
NUM_STEPS = 50


def _group_contact_columns(columns):
    """Group column names by contact quantity (contact_depth, contact_normal, contact_point_0, etc.)."""
    groups = defaultdict(list)
    for col in columns:
        if col == "step":
            continue
        if col.startswith("contact_point_0_"):
            groups["contact_point_0"].append(col)
        elif col.startswith("contact_point_1_"):
            groups["contact_point_1"].append(col)
        else:
            # contact_depth_0, contact_normal_0, contact_thickness_0, contact_mask_0
            base = col.rsplit("_", 1)[0]
            groups[base].append(col)
    # Sort columns within each group by numeric suffix
    def sort_key(c):
        try:
            return int(c.split("_")[-1])
        except ValueError:
            return 0

    for key in groups:
        groups[key].sort(key=sort_key)
    return dict(groups)


def _plot_contact_csv(csv_path: Path, title: str, num_steps: int, fig_axes=None):
    """
    Plot all contact columns from one CSV in subplots (one subplot per contact group).
    """
    df = pd.read_csv(csv_path)
    df = df.head(num_steps)
    # Ignore the first data point of every time evolution
    df = df.iloc[1:]
    steps = df["step"].values
    contact_cols = [c for c in df.columns if c != "step"]
    groups = _group_contact_columns(contact_cols)
    if not groups:
        print(f"No contact columns found in {csv_path}")
        return

    group_order = [
        "contact_mask",
        "contact_depth",
        "contact_thickness",
        "contact_normal",
        "contact_point_0",
        "contact_point_1",
    ]
    # Drop missing groups (e.g. engine CSV has no contact_mask)
    group_order = [g for g in group_order if g in groups]
    n_sub = len(group_order)

    if fig_axes is None:
        fig, axes = plt.subplots(n_sub, 1, figsize=(12, 2.5 * n_sub), sharex=True)
        if n_sub == 1:
            axes = [axes]
    else:
        fig, axes = fig_axes

    for ax, group_name in zip(axes, group_order):
        cols = groups[group_name]
        for col in cols:
            ax.plot(steps, df[col].values, label=col, linewidth=1.0, alpha=0.8)
        ax.set_ylabel(group_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, axes


def plot_contacts(
    nerd_engine_csv_path: str,
    model_inputs_contacts_csv_path: str,
    nerd_env_contacts_csv_path: str,
    nerd_env_contacts_raw_csv_path: str,
    num_steps: int = NUM_STEPS,
):
    """
    Load all four contact CSVs and plot each in a separate figure.

    Args:
        nerd_engine_csv_path: Path to pendulum_contacts_NerdEngine.csv (engine-side contacts).
        model_inputs_contacts_csv_path: Path to pendulum_model_inputs_contacts.csv (model input contacts).
        nerd_env_contacts_csv_path: Path to pendulum_nerdEnv_contacts.csv (original NeRD environment contacts).
        nerd_env_contacts_raw_csv_path: Path to pendulum_nerdEnv_contacts_raw.csv (raw NeRD env contacts).
        num_steps: Number of first time steps to plot (X).
    """
    nerd_engine_csv_path = Path(nerd_engine_csv_path)
    model_inputs_contacts_csv_path = Path(model_inputs_contacts_csv_path)
    nerd_env_contacts_csv_path = Path(nerd_env_contacts_csv_path)
    nerd_env_contacts_raw_csv_path = Path(nerd_env_contacts_raw_csv_path)

    if not nerd_engine_csv_path.exists():
        raise FileNotFoundError(f"Engine contacts CSV not found: {nerd_engine_csv_path}")
    if not model_inputs_contacts_csv_path.exists():
        raise FileNotFoundError(
            f"Model inputs contacts CSV not found: {model_inputs_contacts_csv_path}"
        )
    if not nerd_env_contacts_csv_path.exists():
        raise FileNotFoundError(
            f"NeRD env contacts CSV not found: {nerd_env_contacts_csv_path}"
        )
    if not nerd_env_contacts_raw_csv_path.exists():
        raise FileNotFoundError(
            f"NeRD env contacts raw CSV not found: {nerd_env_contacts_raw_csv_path}"
        )

    print(f"Plotting first {num_steps} steps from each CSV (separate figures).")

    # Figure 1: Engine-side contacts (pendulum_contacts_NerdEngine.csv)
    fig1, _ = _plot_contact_csv(
        nerd_engine_csv_path,
        title="Contact info: engine-side (pendulum_contacts_NerdEngine.csv)",
        num_steps=num_steps,
    )
    fig1.show()

    # Figure 2: Model input contacts (pendulum_model_inputs_contacts.csv)
    fig2, _ = _plot_contact_csv(
        model_inputs_contacts_csv_path,
        title="Contact info: model inputs (pendulum_model_inputs_contacts.csv)",
        num_steps=num_steps,
    )
    fig2.show()

    # Figure 3: NeRD env contacts (pendulum_nerdEnv_contacts.csv)
    fig3, _ = _plot_contact_csv(
        nerd_env_contacts_csv_path,
        title="Contact info: NeRD env (pendulum_nerdEnv_contacts.csv)",
        num_steps=num_steps,
    )
    fig3.show()

    # Figure 4: NeRD env contacts raw (pendulum_nerdEnv_contacts_raw.csv)
    fig4, _ = _plot_contact_csv(
        nerd_env_contacts_raw_csv_path,
        title="Contact info: NeRD env raw (pendulum_nerdEnv_contacts_raw.csv)",
        num_steps=num_steps,
    )
    fig4.show()

    plt.show()


if __name__ == "__main__":
    nerd_engine_csv = "src/axion/core/pendulum_contacts_NerdEngine.csv"
    model_inputs_contacts_csv = "src/axion/nn_prediction/pendulum_model_inputs_contacts.csv"
    nerd_env_contacts_csv = "src/axion/nn_prediction/pendulum_nerdEnv_contacts.csv"
    nerd_env_contacts_raw_csv = "src/axion/nn_prediction/pendulum_nerdEnv_contacts_raw.csv"

    plot_contacts(
        nerd_engine_csv_path=nerd_engine_csv,
        model_inputs_contacts_csv_path=model_inputs_contacts_csv,
        nerd_env_contacts_csv_path=nerd_env_contacts_csv,
        nerd_env_contacts_raw_csv_path=nerd_env_contacts_raw_csv,
        num_steps=NUM_STEPS,
    )
