import glob
import os

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Axion Analyzer 2.2")
st.title("Axion Physics Debugger 2.2 🔍")

# --- SESSION STATE INITIALIZATION ---
if "run_index" not in st.session_state:
    st.session_state.run_index = pd.DataFrame()
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None


# --- UTILITIES ---
class LogReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

    def __enter__(self):
        if self.filepath and os.path.exists(self.filepath):
            self.file = h5py.File(self.filepath, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_timesteps(self):
        if not self.file:
            return []
        return sorted([k for k in self.file.keys() if k.startswith("timestep_")])

    def get_newton_iterations(self, step_key):
        if not self.file or step_key not in self.file:
            return []
        return sorted([k for k in self.file[step_key].keys() if k.startswith("newton_iteration_")])

    def get_world_count(self, step_key):
        if not self.file or step_key not in self.file:
            return 0
        try:
            # Try to find any iteration to infer world count from data shape
            iter_keys = [k for k in self.file[step_key].keys() if k.startswith("newton_iteration_")]
            if not iter_keys:
                return 0

            # Look at dynamics/body_q for shape
            first_iter = iter_keys[0]
            dq = self.file[f"{step_key}/{first_iter}/dynamics/body_q"]
            return dq.shape[0]
        except Exception:
            return 0


# --- 1. FILE SELECTION & LOADING ---
st.sidebar.header("1. File Selection")
search_paths = ["data/logs/*.h5", "data/*.h5", "*.h5"]
files = []
for p in search_paths:
    files.extend(glob.glob(p))

if not files:
    st.error("No HDF5 logs found.")
    st.stop()

selected_file = st.sidebar.selectbox("Log File", sorted(files, reverse=True))

# LOAD BUTTON
if st.sidebar.button("Load Data", type="primary"):
    with st.spinner(f"Indexing {os.path.basename(selected_file)}..."):
        rows = []
        if os.path.exists(selected_file):
            with h5py.File(selected_file, "r") as f:
                steps = sorted([k for k in f.keys() if k.startswith("timestep_")])

                for step in steps:
                    iters = sorted([k for k in f[step].keys() if k.startswith("newton_iteration_")])
                    if not iters:
                        continue

                    last_iter = iters[-1]
                    last_iter_idx = int(last_iter.split("_")[-1])

                    # Try to get residual summary
                    try:
                        ls_grp = f[f"{step}/{last_iter}/linesearch"]
                        # batch_h_norm_sq: (Steps, Worlds)
                        norms_sq = ls_grp["batch_h_norm_sq"][()]
                        min_idxs = ls_grp["minimal_index"][()]

                        num_worlds = norms_sq.shape[1]

                        # Get the residual of the CHOSEN step for each world
                        final_residuals = []
                        for w in range(num_worlds):
                            idx = min_idxs[w]
                            res = np.sqrt(norms_sq[idx, w])
                            final_residuals.append(res)

                        max_res = np.max(final_residuals)
                        max_res_world = np.argmax(final_residuals)

                        rows.append(
                            {
                                "Timestep": int(step.split("_")[1]),
                                "Iterations": last_iter_idx + 1,
                                "Max Residual": max_res,
                                "Worst World": max_res_world,
                                "Key": step,
                            }
                        )
                    except (KeyError, IndexError):
                        rows.append(
                            {
                                "Timestep": int(step.split("_")[1]),
                                "Iterations": last_iter_idx + 1,
                                "Max Residual": 0.0,
                                "Worst World": 0,
                                "Key": step,
                            }
                        )

        st.session_state.run_index = pd.DataFrame(rows)
        st.session_state.loaded_file = selected_file
        st.rerun()

# CHECK IF DATA IS LOADED
if st.session_state.run_index.empty:
    st.info("Select a file and click **Load Data** to begin.")
    st.stop()

df = st.session_state.run_index

# Ensure we are working with the file we loaded
if st.session_state.loaded_file != selected_file:
    st.warning("Selected file matches loaded data? No. Please click 'Load Data' to refresh.")

# --- GLOBAL SIMULATION GRAPH ---
with st.expander("Simulation Overview (Timestep vs Residual)", expanded=True):
    fig_global = px.line(
        df,
        x="Timestep",
        y="Max Residual",
        log_y=True,
        markers=True,
        title="Global Convergence Health (Max Residual per Step)",
        height=300,
    )
    st.plotly_chart(fig_global, use_container_width=True)

# --- 2. TIMESTEP SELECTION ---
st.sidebar.markdown("---")
st.sidebar.header("2. Timestep")

df_sorted = df.sort_values("Max Residual", ascending=False)

selection = st.sidebar.dataframe(
    df_sorted,
    width="stretch",
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    column_config={
        "Max Residual": st.column_config.NumberColumn(format="%.2e"),
        "Worst World": st.column_config.NumberColumn(format="%d"),
    },
    height=200,
)

if selection.selection.rows:
    sel_row = df_sorted.iloc[selection.selection.rows[0]]
else:
    sel_row = df_sorted.iloc[0]

sel_step_key = sel_row["Key"]
sel_step_int = sel_row["Timestep"]
suggested_world = sel_row["Worst World"]

# --- 3. WORLD SELECTION ---
st.sidebar.markdown("---")
st.sidebar.header("3. World")

with LogReader(st.session_state.loaded_file) as log:
    num_worlds = log.get_world_count(sel_step_key)

default_world = int(suggested_world) if int(suggested_world) >= 0 else 0
max_world_idx = max(0, num_worlds - 1)
if default_world > max_world_idx:
    default_world = max_world_idx

sel_world = st.sidebar.number_input(
    "World Index", min_value=0, max_value=max_world_idx, value=default_world
)

st.sidebar.info(f"Step: {sel_step_int} | World: {sel_world}")

# --- MAIN: NEWTON HISTORY ---


def load_newton_history_extended(filepath, step_key, world_idx):
    """
    Loads L2 Norm AND Max Norm (Infinity Norm) for the Newton History.
    """
    with h5py.File(filepath, "r") as f:
        iters = sorted([k for k in f[step_key].keys() if k.startswith("newton_iteration_")])
        if not iters:
            return None

        last_iter_key = iters[-1]
        last_grp = f[f"{step_key}/{last_iter_key}"]

        if "newton_history" in last_grp:
            hist_grp = last_grp["newton_history"]

            # 1. Dynamics
            h_d = hist_grp["dynamics"]["h_d"][:, world_idx, :]  # (Iter, Dof)

            # --- L2 Norm Calculation ---
            total_sq = np.sum(h_d**2, axis=1)
            dyn_l2 = np.sqrt(total_sq)

            # --- Max Norm Calculation ---
            # Start with max of dynamics
            current_max = np.max(np.abs(h_d), axis=1)

            # 2. Constraints
            cons_sq = np.zeros_like(total_sq)

            if "constraints" in hist_grp:
                for c_key in hist_grp["constraints"]:
                    c_data = hist_grp["constraints"][c_key]
                    if "h" in c_data:
                        h_c = c_data["h"][:, world_idx, :]  # (Iter, ConstrDof)

                        # L2 Accumulation
                        c_sq_i = np.sum(h_c**2, axis=1)
                        cons_sq += c_sq_i
                        total_sq += c_sq_i

                        # Max Accumulation
                        max_c = np.max(np.abs(h_c), axis=1)
                        current_max = np.maximum(current_max, max_c)

            total_l2 = np.sqrt(total_sq)
            cons_l2 = np.sqrt(cons_sq)

            return pd.DataFrame(
                {
                    "Iteration": np.arange(len(total_l2)),
                    "Total L2": total_l2,
                    "Dynamics L2": dyn_l2,
                    "Constraint L2": cons_l2,
                    "Max Norm (Inf)": current_max,
                }
            )
    return None


hist_df = load_newton_history_extended(st.session_state.loaded_file, sel_step_key, sel_world)

st.subheader("Newton Convergence")
col1, col2 = st.columns([3, 1])

with col1:
    if hist_df is not None:
        max_iter_display = sel_row["Iterations"]
        hist_df_filtered = hist_df[hist_df["Iteration"] <= max_iter_display]

        fig = px.line(
            hist_df_filtered,
            x="Iteration",
            y=["Total L2", "Max Norm (Inf)", "Constraint L2", "Dynamics L2"],
            log_y=True,
            markers=True,
            title=f"Residual Norms (Step {sel_step_int})",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Newton history not found in log.")

with col2:
    # Use LogReader to fetch available iterations safely
    with LogReader(st.session_state.loaded_file) as log:
        avail_iters = log.get_newton_iterations(sel_step_key)

    if avail_iters:
        max_i = len(avail_iters) - 1
        sel_iter_idx = st.number_input("Detailed Iteration", 0, max_i, max_i)
        sel_iter_key = f"newton_iteration_{sel_iter_idx:02d}"
    else:
        st.warning("No iteration data found.")
        st.stop()

# --- TABS ---
tab_ls, tab_lin, tab_con = st.tabs(["Linesearch", "Linear Solver", "Constraints"])

# --- TAB: LINESEARCH ---
with tab_ls:

    def load_ls(filepath, step, iter_k, world):
        with h5py.File(filepath, "r") as f:
            path = f"{step}/{iter_k}/linesearch"
            if path in f:
                g = f[path]
                return (
                    g["steps"][()],
                    np.sqrt(g["batch_h_norm_sq"][:, world]),
                    g["minimal_index"][world],
                )
        return None

    ls_d = load_ls(st.session_state.loaded_file, sel_step_key, sel_iter_key, sel_world)
    if ls_d:
        xs, ys, idx = ls_d
        fig_ls = go.Figure()
        fig_ls.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Merit"))
        fig_ls.add_trace(
            go.Scatter(
                x=[xs[idx]],
                y=[ys[idx]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name="Selected",
            )
        )
        fig_ls.update_layout(xaxis_type="log", yaxis_type="log", title="Linesearch Merit Function")
        st.plotly_chart(fig_ls, use_container_width=True)
    else:
        st.info("No linesearch data.")

# --- TAB: LINEAR SOLVER ---
with tab_lin:

    def load_linear_robust(filepath, step, iter_k, world):
        with h5py.File(filepath, "r") as f:
            path = f"{step}/{iter_k}/linear_solver_stats"
            if path not in f:
                return None, "Group not found"

            g = f[path]

            # 1. Try "residual_sq_history" (Squared Norms)
            if "residual_squared_history" in g:
                data = g["residual_squared_history"]
                # It is squared, so take sqrt
                if data.ndim > 1:
                    return np.sqrt(data[:, world]), "residual_squared_history"
                return np.sqrt(data), "residual_squared_history"

            # 2. Fallback: look for ANY dataset
            keys = [k for k in g.keys() if isinstance(g[k], h5py.Dataset)]
            if keys:
                data = g[keys[0]]
                if data.ndim > 1:
                    return np.sqrt(data[:, world]), f"{keys[0]} (fallback)"
                return np.sqrt(data), f"{keys[0]} (fallback)"

            return None, f"Empty group. Keys found: {list(g.keys())}"

    lin_data, source_key = load_linear_robust(
        st.session_state.loaded_file, sel_step_key, sel_iter_key, sel_world
    )

    if lin_data is not None:
        st.caption(f"Loaded from: `{source_key}`")
        fig_lin = px.line(y=lin_data, log_y=True, markers=True, title="Linear Solver Convergence")
        fig_lin.update_layout(xaxis_title="Linear Iteration", yaxis_title="Residual Norm")
        st.plotly_chart(fig_lin, use_container_width=True)
    else:
        st.warning(f"Could not load linear solver data. Reason: {source_key}")

# --- TAB: CONSTRAINTS ---
with tab_con:

    @st.cache_data
    def load_constraints(filepath, step, iter_k, world):
        out = {}
        with h5py.File(filepath, "r") as f:
            path = f"{step}/{iter_k}/constraints"
            if path in f:
                for k in f[path]:
                    g = f[path][k]
                    item = {}
                    if "h" in g:
                        val = g["h"][world]
                        item["Residual"] = np.linalg.norm(val, axis=1) if val.ndim > 1 else val
                    if "body_lambda" in g:
                        val = g["body_lambda"][world]
                        item["Lambda"] = np.linalg.norm(val, axis=1) if val.ndim > 1 else val
                    if item:
                        out[k] = pd.DataFrame(item)
        return out

    c_dfs = load_constraints(st.session_state.loaded_file, sel_step_key, sel_iter_key, sel_world)
    if c_dfs:
        c_names = list(c_dfs.keys())
        subtabs = st.tabs(c_names)
        for i, name in enumerate(c_names):
            with subtabs[i]:
                d = c_dfs[name]
                st.dataframe(d, width="stretch", height=300)
                if "Residual" in d and "Lambda" in d:
                    st.plotly_chart(
                        px.scatter(d, x="Residual", y="Lambda", title=f"{name} Scatter"),
                        use_container_width=True,
                    )
    else:
        st.info("No constraints active.")

