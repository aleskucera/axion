import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Axion Debugger")
st.title("Axion Physics Debugger 🔍")

# --- SIDEBAR: DATA LOADING ---
st.sidebar.header("1. Select Simulation")
# Search in data/ folder for h5 files
data_files = glob.glob("data/logs/*.h5")
if not data_files:
    # Fallback to current dir if data/ is empty
    data_files = glob.glob("*.h5")

if not data_files:
    st.error("No .h5 files found in `data/logs` or current directory.")
    st.stop()

selected_file = st.sidebar.selectbox("File", data_files, index=0)

# --- SIDEBAR: WORLD FILTERING ---
st.sidebar.markdown("---")
st.sidebar.header("2. World Selection")


# Function to get max world count efficiently
@st.cache_data
def get_world_count(file_path):
    with h5py.File(file_path, "r") as f:
        # Check first available timestep
        steps = sorted([k for k in f.keys() if k.startswith("timestep_")])
        if not steps:
            return 0
        g = f[steps[0]]
        if "residual_norm_landscape_data" in g:
            # Metadata is [grid_res, plot_scale, steps, num_worlds]
            meta = g["residual_norm_landscape_data/pca_metadata"][()]
            return int(meta[3])
    return 0


total_worlds = get_world_count(selected_file)

if total_worlds > 0:
    st.sidebar.caption(f"Total Worlds in File: {total_worlds}")

    # Range Input
    w_min = st.sidebar.number_input("Min World", 0, total_worlds - 1, 0)
    w_max = st.sidebar.number_input("Max World", 0, total_worlds - 1, min(10, total_worlds - 1))

    if w_min > w_max:
        st.sidebar.error("Min World cannot be greater than Max World")
else:
    st.sidebar.warning("Could not determine world count.")
    w_min, w_max = 0, 0


# --- DATA LOADING ---
@st.cache_data
def load_timestep_table(file_path, w_start, w_end):
    """Loads and filters data, returning only rows within the requested world range."""
    if not Path(file_path).exists():
        return None

    data_rows = []

    with h5py.File(file_path, "r") as f:
        steps = sorted([k for k in f.keys() if k.startswith("timestep_")])

        for step in steps:
            g = f[step]
            if "residual_norm_landscape_data" not in g:
                continue

            sub = g["residual_norm_landscape_data"]
            if "simulation_dims" not in sub:
                continue

            # trajectory_residuals shape: (Iterations, Worlds, Dofs)
            # We need the norms of the LAST iteration for the worlds in range

            # Optimization: Only load the slice we need
            # This requires 'trajectory_residuals' to be chunked reasonably well
            # But h5py slicing is generally faster than loading everything

            # Slice: Last Iteration (-1), World Range (w_start:w_end+1), All Dofs (:)
            # Note: trajectory_residuals is (Iter, World, Dof)
            last_h_slice = sub["trajectory_residuals"][-1, w_start : w_end + 1, :]

            # Compute norms
            norms = np.linalg.norm(last_h_slice, axis=1)

            for i, val in enumerate(norms):
                data_rows.append(
                    {
                        "Timestep": int(step.split("_")[1]),
                        "World": w_start + i,
                        "Final Residual": float(val),
                    }
                )

    return pd.DataFrame(data_rows)


df = load_timestep_table(selected_file, w_min, w_max)

if df is None or df.empty:
    st.warning(f"No valid data found in range [{w_min}-{w_max}].")
    st.stop()

# --- TABLE DISPLAY ---
st.info("Sort table by 'Final Residual' to find broken steps. Click a row to inspect.")

# Simple Table
selection = st.dataframe(
    df.sort_values("Final Residual", ascending=False),
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    column_config={
        "Final Residual": st.column_config.NumberColumn(format="%.4e"),
        "Timestep": st.column_config.NumberColumn(format="%d"),
        "World": st.column_config.NumberColumn(format="%d"),
    },
    height=400,
)

# --- SELECTION HANDLING ---
if selection.selection.rows:
    # dataframe is sorted, need to find the correct row from the sorted index
    sorted_df = df.sort_values("Final Residual", ascending=False)
    selected_idx = selection.selection.rows[0]
    selected_row = sorted_df.iloc[selected_idx]
else:
    # Default to worst offender in range
    sorted_df = df.sort_values("Final Residual", ascending=False)
    selected_row = sorted_df.iloc[0]

sel_step = int(selected_row["Timestep"])
sel_world = int(selected_row["World"])

st.markdown(f"### 📍 Inspecting: Timestep `{sel_step}`, World `{sel_world}`")


# --- DETAIL LOADING ---
@st.cache_data
def load_detailed_data(file_path, step, world_idx):
    with h5py.File(file_path, "r") as f:
        g = f[f"timestep_{step:04d}/residual_norm_landscape_data"]

        grid = g["residual_norm_grid"][:, :, world_idx]
        alphas = g["pca_alphas"][world_idx, :]
        betas = g["pca_betas"][world_idx, :]
        traj_2d = g["trajectory_2d_projected"][:, world_idx, :]

        residuals = g["trajectory_residuals"][:, world_idx, :]
        lambdas = g["body_lambda_history"][:, world_idx, :]
        dims = g["simulation_dims"][()]

    return grid, alphas, betas, traj_2d, residuals, lambdas, dims


grid, alphas, betas, traj_2d, h_hist, lambda_hist, dims = load_detailed_data(
    selected_file, sel_step, sel_world
)

# --- PLOTS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Loss Landscape")

    # Auto-Range Logic for Colors
    grid_valid = grid[grid > 1e-12]
    if len(grid_valid) > 0:
        p05 = np.percentile(grid_valid, 5)
        p95 = np.percentile(grid_valid, 95)
        log_min_def = float(np.log10(p05))
        log_max_def = float(np.log10(p95))
    else:
        log_min_def, log_max_def = -6.0, 0.0

    c_min, c_max = st.slider(
        "Color Scale (Log10)", min_value=-12.0, max_value=5.0, value=(log_min_def, log_max_def)
    )

    # Log and Clip
    grid_log = np.log10(np.clip(grid.T, 1e-12, None))

    fig_land = go.Figure()
    fig_land.add_trace(
        go.Contour(
            z=grid_log,
            x=alphas,
            y=betas,
            colorscale="Plasma",
            zmin=c_min,
            zmax=c_max,
            contours=dict(coloring="heatmap"),
            colorbar=dict(title="Log(Res)", len=0.5),
        )
    )

    fig_land.add_trace(
        go.Scatter(
            x=traj_2d[:, 0],
            y=traj_2d[:, 1],
            mode="lines+markers",
            line=dict(color="white"),
            name="Path",
        )
    )

    # Markers
    fig_land.add_trace(
        go.Scatter(
            x=[traj_2d[0, 0]],
            y=[traj_2d[0, 1]],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Start",
        )
    )
    fig_land.add_trace(
        go.Scatter(
            x=[traj_2d[-1, 0]],
            y=[traj_2d[-1, 1]],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            name="End",
        )
    )

    fig_land.update_layout(height=500, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_land, use_container_width=True)

with col2:
    st.subheader("Max Residual per Iteration")

    N_u, N_j, N_n, N_f = dims
    s_d = slice(0, N_u)
    s_j = slice(N_u, N_u + N_j)
    s_n = slice(N_u + N_j, N_u + N_j + N_n)
    s_f = slice(N_u + N_j + N_n, N_u + N_j + N_n + N_f)

    # Calculate Max(|h|) for each component
    iters = np.arange(h_hist.shape[0])

    # Compute max absolute error per iteration
    max_d = np.max(np.abs(h_hist[:, s_d]), axis=1) if N_u > 0 else np.zeros(len(iters))
    max_j = np.max(np.abs(h_hist[:, s_j]), axis=1) if N_j > 0 else np.zeros(len(iters))
    max_n = np.max(np.abs(h_hist[:, s_n]), axis=1) if N_n > 0 else np.zeros(len(iters))
    max_f = np.max(np.abs(h_hist[:, s_f]), axis=1) if N_f > 0 else np.zeros(len(iters))

    fig_max = go.Figure()
    fig_max.add_trace(go.Scatter(x=iters, y=max_d, name="Dynamics", line=dict(dash="solid")))
    fig_max.add_trace(go.Scatter(x=iters, y=max_j, name="Joints", line=dict(dash="dash")))
    fig_max.add_trace(go.Scatter(x=iters, y=max_n, name="Contacts", line=dict(width=3)))
    fig_max.add_trace(go.Scatter(x=iters, y=max_f, name="Friction", line=dict(dash="dot")))

    fig_max.update_layout(
        yaxis_type="log",
        yaxis_title="Max(|h|)",
        xaxis_title="Newton Iteration",
        height=500,
        margin=dict(t=20, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_max, use_container_width=True)
