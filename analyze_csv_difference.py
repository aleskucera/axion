#!/usr/bin/env python3
"""
Script to analyze differences between two CSV files.
Compares element-wise differences and plots columns with nonzero differences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_csv_difference(
    file1_path: str,
    file2_path: str,
    max_rows: int = 100,
    epsilon: float = 1e-10
):
    """
    Analyze differences between two CSV files.
    
    Args:
        file1_path: Path to first CSV file
        file2_path: Path to second CSV file
        max_rows: Maximum number of rows to compare (default: 100)
        epsilon: Threshold for considering differences as nonzero (default: 1e-10)
    """
    # Read CSV files
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"Reading {file2_path}...")
    df2 = pd.read_csv(file2_path)
    
    # Limit to first max_rows rows
    df1 = df1.head(max_rows)
    df2 = df2.head(max_rows)
    
    # Get common columns (excluding 'step' if present)
    common_cols = [col for col in df1.columns if col in df2.columns]
    if 'step' in common_cols:
        common_cols.remove('step')
    
    print(f"\nFile 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    print(f"Common columns: {len(common_cols)}")
    
    # Find columns only in one file
    only_in_file1 = set(df1.columns) - set(df2.columns)
    only_in_file2 = set(df2.columns) - set(df1.columns)
    
    if only_in_file1:
        print(f"\nColumns only in file 1: {only_in_file1}")
    if only_in_file2:
        print(f"Columns only in file 2: {only_in_file2}")
    
    # Compute differences for common columns and store values
    column_values = {}
    nonzero_columns = []
    
    # Extract file names for legend
    file1_name = Path(file1_path).stem
    file2_name = Path(file2_path).stem
    
    for col in common_cols:
        # Get the data for this column from both dataframes
        col1 = df1[col].values
        col2 = df2[col].values
        
        # Ensure same length (pad with NaN if needed)
        min_len = min(len(col1), len(col2))
        col1 = col1[:min_len]
        col2 = col2[:min_len]
        
        # Compute element-wise difference
        diff = col1 - col2
        
        # Check if any differences exceed epsilon
        max_diff = np.max(np.abs(diff))
        if max_diff > epsilon:
            column_values[col] = (col1, col2)
            nonzero_columns.append(col)
            print(f"Column '{col}': max difference = {max_diff:.6e}")
    
    print(f"\nFound {len(nonzero_columns)} columns with differences > {epsilon}")
    
    if not nonzero_columns:
        print("No significant differences found!")
        return
    
    # Plot differences
    n_cols = len(nonzero_columns)
    if n_cols == 0:
        return
    
    # Determine subplot layout
    n_plots_per_row = 3
    n_rows = (n_cols + n_plots_per_row - 1) // n_plots_per_row
    
    fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(15, 5 * n_rows))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(nonzero_columns):
        ax = axes[idx]
        col1, col2 = column_values[col]
        x = np.arange(len(col1))
        
        # Plot individual values from both files
        ax.plot(x, col1, label=file1_name, linewidth=1.5, alpha=0.7)
        ax.plot(x, col2, label=file2_name, linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Compute and show max difference in title
        diff = col1 - col2
        max_diff = np.max(np.abs(diff))
        
        ax.set_xlabel('Row index')
        ax.set_ylabel('Value')
        ax.set_title(f'{col}\n(max diff: {max_diff:.6e})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('csv_differences.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to 'csv_differences.png'")
    plt.show()


if __name__ == '__main__':
    # File paths
    file1 = 'src/axion/nn_prediction/nerd_pendulum_model_inputs.csv'
    file2 = 'src/axion/nn_prediction/pendulum_model_inputs.csv'
    
    # Parameters
    max_rows = 2000
    epsilon = 1e-4
    
    analyze_csv_difference(file1, file2, max_rows=max_rows, epsilon=epsilon)

