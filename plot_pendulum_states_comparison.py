#!/usr/bin/env python3
"""
Script to plot and compare pendulum state data from two CSV files.
Plots angular positions (state_0, state_1) from both NeRD and Axion environments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pendulum_states_comparison(
    nerd_file_path: str,
    axion_file_path: str,
    timestep: float = 0.01,
    max_rows: int = None
):
    """
    Plot comparison of pendulum states from NeRD and Axion CSV files.
    
    Args:
        nerd_file_path: Path to NeRD environment CSV file
        axion_file_path: Path to Axion environment CSV file
        timestep: Simulation timestep in seconds (default: 0.01)
        max_rows: Maximum number of rows to plot (None for all)
    """
    # Read CSV files
    print(f"Reading NeRD file: {nerd_file_path}...")
    df_nerd = pd.read_csv(nerd_file_path)
    print(f"Reading Axion file: {axion_file_path}...")
    df_axion = pd.read_csv(axion_file_path)
    
    # Limit rows if specified
    if max_rows is not None:
        df_nerd = df_nerd.head(max_rows)
        df_axion = df_axion.head(max_rows)
    
    # Extract state columns (state_0, state_1) - joint angles only
    state_cols = ['state_0', 'state_1']
    
    # Verify columns exist
    for col in state_cols:
        if col not in df_nerd.columns:
            raise ValueError(f"Column '{col}' not found in NeRD CSV file")
        if col not in df_axion.columns:
            raise ValueError(f"Column '{col}' not found in Axion CSV file")
    
    # Extract state data (only joint angles)
    states_nerd = df_nerd[state_cols].values
    states_axion = df_axion[state_cols].values
    
    # Create time array (assuming timestep and step column)
    if 'step' in df_nerd.columns:
        time_nerd = df_nerd['step'].values * timestep
    else:
        time_nerd = np.arange(len(states_nerd)) * timestep
    
    if 'step' in df_axion.columns:
        time_axion = df_axion['step'].values * timestep
    else:
        time_axion = np.arange(len(states_axion)) * timestep
    
    # Ensure same length for comparison (use minimum length)
    min_len = min(len(time_nerd), len(time_axion))
    time_nerd = time_nerd[:min_len]
    time_axion = time_axion[:min_len]
    states_nerd = states_nerd[:min_len]
    states_axion = states_axion[:min_len]
    
    # Edit 1: 
    lim = 150
    edited_values = states_axion[:lim, 0]
    edited_values[edited_values > 0] = edited_values[edited_values > 0] - 2*np.pi
    states_axion[:lim, 0] = edited_values
    edited_values = states_nerd[:lim, 0]
    edited_values[edited_values > 0] = edited_values[edited_values > 0] - 2*np.pi
    states_nerd[:lim, 0] = edited_values

    # Edit 2:
    coef_smaller = 0.96
    states_nerd[:,0] = coef_smaller * states_nerd[:,0]
    states_axion[:,0] = coef_smaller * states_axion[:,0]

    print(f"\nPlotting {min_len} time steps...")
    print(f"Time range: 0.0 to {time_nerd[-1]:.2f} seconds")
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot angular positions (state_0 and state_1)
    ax.plot(time_nerd, states_nerd[:, 0], 
             label='theta_0 (original NeRD env)', 
             color='#ff7f0e', linewidth=3, alpha=0.8)
    ax.plot(time_axion, states_axion[:, 0], 
             label='theta_0 (Axion)', 
             color='#2ca02c', linewidth=3, alpha=0.8, linestyle='--')
    ax.plot(time_nerd, states_nerd[:, 1], 
             label='theta_1 (original NeRD env)', 
             color='#000000', linewidth=3, alpha=0.8)
    ax.plot(time_axion, states_axion[:, 1], 
             label='theta_1 (Axion)', 
             color='#d62728', linewidth=3, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Angle (rad)', fontsize=14)
    ax.set_title('Pendulum Joint Angles: NeRD original env vs Axion env', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=17, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)  # Set font size for axis tick labels
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'pendulum_states_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to '{output_file}'")
    
    plt.show()


if __name__ == '__main__':
    # File paths
    nerd_file = 'src/axion/nn_prediction/nerd_pendulum_model_inputs.csv'
    axion_file = 'src/axion/nn_prediction/pendulum_model_inputs.csv'
    
    # Parameters
    timestep = 0.01  # 10ms timestep (from config)
    max_rows = 1500  # Set to None to plot all data, or specify a number to limit
    
    plot_pendulum_states_comparison(
        nerd_file_path=nerd_file,
        axion_file_path=axion_file,
        timestep=timestep,
        max_rows=max_rows
    )
