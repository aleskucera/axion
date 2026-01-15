import glob
import os
import time
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
# Search in data/logs/ folder first, then data/, then current
search_paths = ["data/logs/*.h5", "data/*.h5", "*.h5"]
data_files = []
for p in search_paths:
    data_files.extend(glob.glob(p))
    if data_files:
        break

if not data_files:
    st.error("No .h5 files found in `data/logs`, `data/`, or current directory.")
    st.stop()

selected_file = st.sidebar.selectbox("File", sorted(data_files, reverse=True), index=0)

# Manual Refresh Button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Check modification time for auto-refresh
current_mtime = os.path.getmtime(selected_file)

# --- SIDEBAR: WORLD FILTERING ---
st.sidebar.markdown("---")
st.sidebar.header("2. World Selection")


# Function to get max world count efficiently
@st.cache_data
def get_world_count(file_path, _mtime):
    with h5py.File(file_path, "r") as f:
        # Check first available timestep
        steps = sorted([k for k in f.keys() if k.startswith("timestep_")])
        if not steps:
            return 0
        g = f[steps[0]]
        
        # Try new history data location
        if "history_data" in g:
            # h_history shape: (Iter, World, Dof)
            if "h_history" in g["history_data"]:
                return g["history_data"]["h_history"].shape[1]

        # Try legacy residual_norm_landscape_data
        if "residual_norm_landscape_data" in g:
            if "pca_metadata" in g["residual_norm_landscape_data"]:
                # Metadata is [grid_res, plot_scale, steps, num_worlds]
                meta = g["residual_norm_landscape_data"]["pca_metadata"][()]
                return int(meta[3])
            if "trajectory_residuals" in g["residual_norm_landscape_data"]:
                return g["residual_norm_landscape_data"]["trajectory_residuals"].shape[1]
                
    return 0


total_worlds = get_world_count(selected_file, current_mtime)

if total_worlds > 0:
    st.sidebar.caption(f"Total Worlds in File: {total_worlds}")

    # Range Input
    w_min = st.sidebar.number_input("Min World", 0, total_worlds - 1, 0)
    w_max = st.sidebar.number_input("Max World", 0, total_worlds - 1, min(10, total_worlds - 1))

    if w_min > w_max:
        st.sidebar.error("Min World cannot be greater than Max World")
else:
    st.sidebar.warning("Could not determine world count. File might be empty or format unknown.")
    w_min, w_max = 0, 0


# --- DATA LOADING ---
@st.cache_data
def load_timestep_table(file_path, w_start, w_end, _mtime):
    """Loads and filters data, returning only rows within the requested world range."""
    if not Path(file_path).exists():
        return None

    data_rows = []

    with h5py.File(file_path, "r") as f:
        steps = sorted([k for k in f.keys() if k.startswith("timestep_")])

        for step in steps:
            g = f[step]
            
            # 1. Try Loading from History Data (New Standard)
            if "history_data" in g and "h_history" in g["history_data"]:
                # h_history: (Iter, World, Dof)
                # We want the norm of the last iteration for the selected worlds
                # Slice: Last Iter (-1), Worlds (w_start:w_end+1), All Dofs (:)
                last_h = g["history_data"]["h_history"][-1, w_start : w_end + 1, :]
                norms = np.linalg.norm(last_h, axis=1)
                
            # 2. Fallback to Legacy Data
            elif "residual_norm_landscape_data" in g and "trajectory_residuals" in g["residual_norm_landscape_data"]:
                last_h = g["residual_norm_landscape_data"]["trajectory_residuals"][-1, w_start : w_end + 1, :]
                norms = np.linalg.norm(last_h, axis=1)
            else:
                continue

            for i, val in enumerate(norms):
                data_rows.append(
                    {
                        "Timestep": int(step.split("_")[1]),
                        "World": w_start + i,
                        "Final Residual": float(val),
                    }
                )

    return pd.DataFrame(data_rows)


df = load_timestep_table(selected_file, w_min, w_max, current_mtime)

if df is None or df.empty:
    st.warning(f"No valid data found in range [{w_min}-{w_max}].")
    st.stop()

# --- SIDEBAR: TIMESTEP SELECTION ---
with st.sidebar:
    st.markdown("---")
    st.header("3. Select Timestep")
    st.caption("Sort by 'Final Residual' to find broken steps.")

    # Sort default so worst steps are at the top
    df_sorted = df.sort_values("Final Residual", ascending=False)

    selection = st.dataframe(
        df_sorted,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "Final Residual": st.column_config.NumberColumn(format="%.2e"),
            "Timestep": st.column_config.NumberColumn(format="%d"),
            "World": st.column_config.NumberColumn(format="%d"),
        },
        height=300,  # Fixed height for sidebar
    )

# --- SELECTION HANDLING ---
if selection.selection.rows:
    # Get the row index from the *sorted* dataframe (since the user clicked on sorted view)
    selected_idx = selection.selection.rows[0]
    selected_row = df_sorted.iloc[selected_idx]
else:
    # Default to the worst offender if nothing selected
    selected_row = df_sorted.iloc[0]

sel_step = int(selected_row["Timestep"])
sel_world = int(selected_row["World"])

# Show what is currently selected in the main view
st.success(
    f"**Inspecting:** Timestep `{sel_step}` | World `{sel_world}` | Residual `{selected_row['Final Residual']:.2e}`"
)


# --- DETAIL LOADING ---
@st.cache_data
def load_history_data(file_path, step, world_idx, _mtime):
    with h5py.File(file_path, "r") as f:
        step_key = f"timestep_{step:04d}"
        if step_key not in f:
            return None, None
            
        g = f[step_key]
        
        if "history_data" in g:
            sub = g["history_data"]
            h_hist = sub["h_history"][:, world_idx, :]
            dims = sub["simulation_dims"][()]
            return h_hist, dims
            
        # Fallback
        if "residual_norm_landscape_data" in g:
            sub = g["residual_norm_landscape_data"]
            h_hist = sub["trajectory_residuals"][:, world_idx, :]
            dims = sub["simulation_dims"][()]
            return h_hist, dims
            
    return None, None

h_hist, dims = load_history_data(selected_file, sel_step, sel_world, current_mtime)

@st.cache_data
def load_linesearch_data(file_path, step, newton_iter, world_idx, _mtime):
    with h5py.File(file_path, "r") as f:
        # Path: timestep_XXXX/newton_iteration_YY/linesearch
        key = f"timestep_{step:04d}/newton_iteration_{newton_iter:02d}/linesearch"
        
        if key not in f:
            return None
        
        g = f[key]
        # steps: (N_steps,)
        steps = g["steps"][:]
        # batch_h_norm_sq: (N_steps, N_worlds)
        batch_norms_sq = g["batch_h_norm_sq"][:, world_idx]
        # minimal_index: (N_worlds,)
        min_idx = g["minimal_index"][world_idx]
        
        return steps, np.sqrt(batch_norms_sq), min_idx

# --- NEWTON ITERATION SELECTOR ---
# We need to know how many iterations happened.
# h_hist has shape (iters + 1, dofs) because it includes initial state (iter 0) + results of N iterations.
# Or does it? Let's check engine.py/engine_data.py
# Engine loop runs 'max_newton_iters' times.
# copy_state_to_history is called inside loop (i=0..N-1) and once at end (i=N).
# So h_hist size is N+1.
# Iteration 0 corresponds to "Initial Guess" state?
# Usually, loop i corresponds to "Newton Iteration i".
# logger.log_linesearch_step(self, i) is called inside loop.
# So valid iterations for linesearch are 0 to max_iter-1.

if h_hist is not None:
    max_iter_idx = h_hist.shape[0] - 2 # (N+1 states means N steps, indices 0..N-1)
    if max_iter_idx < 0: max_iter_idx = 0
else:
    max_iter_idx = 0

with st.sidebar:
    st.markdown("---")
    st.header("4. Newton Iteration")
    selected_newton_iter = st.slider(
        "Select Iteration", 
        0, 
        max_iter_idx, 
        0,
        help="Select which Newton step to inspect (Linesearch & Linear Solve)."
    )

# --- PLOTS ---
col1, col2 = st.columns([1, 1])

# --- COL 1: LINESEARCH LANDSCAPE ---
with col1:
    st.subheader("Linesearch Landscape (Merit Function)")
    
    ls_data = load_linesearch_data(selected_file, sel_step, selected_newton_iter, sel_world, current_mtime)
    
    if ls_data is not None:
        ls_steps, ls_norms, ls_min_idx = ls_data
        
        fig_ls = go.Figure()
        
        # Plot the curve
        fig_ls.add_trace(
            go.Scatter(
                x=ls_steps,
                y=ls_norms,
                mode="lines+markers",
                name="Merit Function",
                line=dict(color="blue"),
                marker=dict(size=6)
            )
        )
        
        # Highlight the selected step
        chosen_step = ls_steps[ls_min_idx]
        chosen_norm = ls_norms[ls_min_idx]
        
        fig_ls.add_trace(
            go.Scatter(
                x=[chosen_step],
                y=[chosen_norm],
                mode="markers",
                name="Chosen Step",
                marker=dict(color="red", size=12, symbol="star"),
                text=[f"Step: {chosen_step:.2e}<br>Norm: {chosen_norm:.2e}"],
                hoverinfo="text"
            )
        )
        
        fig_ls.update_layout(
            xaxis_title="Step Size (alpha)",
            yaxis_title="Residual Norm ||h(alpha)||",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            margin=dict(t=0, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_ls, use_container_width=True)
    else:
        st.info("No linesearch data found for this iteration. Ensure `log_linesearch_data` is True.")


# --- COL 2: MAX RESIDUAL PER ITERATION ---
with col2:
    st.subheader("Convergence History (Max Residual)")

    if h_hist is not None and dims is not None:
        # Handle backward compatibility for files without N_ctrl in dims
        if len(dims) == 5:
            N_u, N_j, N_ctrl, N_n, N_f = dims
        else:
            N_u, N_j, N_n, N_f = dims
            N_ctrl = 0

        s_d = slice(0, N_u)
        s_j = slice(N_u, N_u + N_j)
        s_ctrl = slice(N_u + N_j, N_u + N_j + N_ctrl)
        s_n = slice(N_u + N_j + N_ctrl, N_u + N_j + N_ctrl + N_n)
        s_f = slice(
            N_u + N_j + N_ctrl + N_n, N_u + N_j + N_ctrl + N_n + N_f
        )

        # Calculate Max(|h|) for each component
        iters = np.arange(h_hist.shape[0])

        def compute_stats(slice_obj, offset):
            if slice_obj.start == slice_obj.stop:
                return np.zeros(len(iters)), np.zeros(len(iters)), np.zeros(len(iters))
            
            data = h_hist[:, slice_obj]
            # L2 Norm
            norms = np.linalg.norm(data, axis=1)
            # Max Absolute Value
            abs_data = np.abs(data)
            max_vals = np.max(abs_data, axis=1)
            # Global Index of Max Value
            max_idxs = np.argmax(abs_data, axis=1) + offset
            return norms, max_vals, max_idxs

        norm_d, max_d, idx_d = compute_stats(s_d, s_d.start)
        norm_j, max_j, idx_j = compute_stats(s_j, s_j.start)
        norm_ctrl, max_ctrl, idx_ctrl = compute_stats(s_ctrl, s_ctrl.start)
        norm_n, max_n, idx_n = compute_stats(s_n, s_n.start)
        norm_f, max_f, idx_f = compute_stats(s_f, s_f.start)

        # Total Norm
        total_norm = np.linalg.norm(h_hist, axis=1)

        fig_max = go.Figure()

        def add_trace(name, x, y, max_vals, max_idxs, line_style=None, width=None):
            hover_text = [
                f"Iter: {i}<br>Norm: {y[i]:.2e}<br>Max: {max_vals[i]:.2e}<br>Idx: {int(max_idxs[i])}"
                for i in range(len(x))
            ]
            fig_max.add_trace(
                go.Scatter(
                    x=x, 
                    y=y, 
                    name=name, 
                    line=dict(dash=line_style, width=width),
                    text=hover_text,
                    hoverinfo="text+name"
                )
            )

        add_trace("Dynamics", iters, norm_d, max_d, idx_d, line_style="solid")
        add_trace("Joints", iters, norm_j, max_j, idx_j, line_style="dash")
        add_trace("Control", iters, norm_ctrl, max_ctrl, idx_ctrl, line_style="longdash")
        add_trace("Contacts", iters, norm_n, max_n, idx_n, width=3)
        add_trace("Friction", iters, norm_f, max_f, idx_f, line_style="dot")
        
        # Add Total Norm Trace
        fig_max.add_trace(
            go.Scatter(
                x=iters,
                y=total_norm,
                name="Total Norm",
                line=dict(color="black", width=2),
                text=[f"Iter: {i}<br>Total Norm: {total_norm[i]:.2e}" for i in range(len(iters))],
                hoverinfo="text+name"
            )
        )
        
        # Add marker for current selected iteration
        if selected_newton_iter < len(iters):
             fig_max.add_vline(x=selected_newton_iter, line_width=1, line_dash="dash", line_color="grey")

        fig_max.update_layout(
            yaxis_type="log",
            yaxis_title="L2 Norm ||h||",
            xaxis_title="Newton Iteration",
            height=500,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_max, width="stretch")
    else:
        st.warning("No history data available.")


# --- LINEAR SOLVER CONVERGENCE ---
st.markdown("---")
st.subheader("Linear Solver Convergence")

@st.cache_data
def load_linear_history(file_path, step, newton_iter, _mtime):
    """Loads linear solver history for a specific Newton iteration."""
    with h5py.File(file_path, "r") as f:
        # Check if linear solver history exists for this newton iter
        key = f"timestep_{step:04d}/newton_iteration_{newton_iter:02d}/linear_solver_history"
        if key not in f:
            return None
        
        g = f[key]
        return g["residual_sq_history"][:]

lin_hist = load_linear_history(selected_file, sel_step, selected_newton_iter, current_mtime)

if lin_hist is not None:
    # lin_hist shape: (Linear_Iters + 1, Num_Worlds)
    
    # Extract for selected world
    r_sq_world = lin_hist[:, sel_world]
    r_norm_world = np.sqrt(r_sq_world)
    
    lin_iters = np.arange(len(r_norm_world))
    
    fig_lin = go.Figure()
    fig_lin.add_trace(
        go.Scatter(
            x=lin_iters,
            y=r_norm_world,
            mode="lines+markers",
            name=f"World {sel_world}",
        )
    )
    
    fig_lin.update_layout(
        title=f"Linear Solve (Newton Iter {selected_newton_iter})",
        xaxis_title="Linear Iteration",
        yaxis_title="Residual Norm ||r||",
        yaxis_type="log",
        height=400
    )
    st.plotly_chart(fig_lin, use_container_width=True)
else:
    st.info(f"No linear solver history found for Newton Iteration {selected_newton_iter}.")