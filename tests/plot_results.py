import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_interactive_report(csv_file="results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Please run the benchmark script first.")
        return

    # 1. Load Data
    df = pd.read_csv(csv_file)

    # Convert boolean 'Graph' to string for better legend labels
    df["Graph_Mode"] = df["Graph"].apply(lambda x: "CUDA Graph" if x else "Eager Mode")

    # Create a unique label for the legend
    df["Config"] = df["Scenario"] + " (" + df["Graph_Mode"] + ")"

    print(f"Loaded {len(df)} rows from {csv_file}. Generating plots...")

    # ==============================================================================
    # PLOT 1: Execution Time Scaling (Log-Log)
    # ==============================================================================
    # This reveals the scaling behavior (Linear vs Quadratic) and Overhead.

    fig_time = px.line(
        df,
        x="N_CONTACTS",
        y="Time_ms",
        color="Scenario",  # Different colors for scenarios (Random/Contention)
        symbol="Graph_Mode",  # Different line styles for Graph vs Eager
        log_x=True,
        log_y=True,
        error_y="Std_ms",  # Show standard deviation bars
        title="Kernel Execution Time vs. Contact Count (Log-Log)",
        markers=True,
        hover_data=["N_BODIES", "N_WORLDS", "Contacts_Per_Sec"],
    )

    fig_time.update_layout(
        xaxis_title="Number of Contacts",
        yaxis_title="Time (ms) - Lower is Better",
        template="plotly_dark",
        hovermode="x unified",
    )

    # Save to HTML
    fig_time.write_html("benchmark_time_scaling.html")
    print("-> Generated: benchmark_time_scaling.html")

    # ==============================================================================
    # PLOT 2: Throughput (Contacts per Second)
    # ==============================================================================
    # This reveals GPU Saturation.
    # The curve should rise and flatten out at the hardware's limit.

    fig_thp = px.line(
        df,
        x="N_CONTACTS",
        y="Contacts_Per_Sec",
        color="Scenario",
        symbol="Graph_Mode",
        log_x=True,
        markers=True,
        title="Throughput: Contacts Solved per Second",
        hover_data=["Time_ms"],
    )

    # Add a reference line (e.g., 100 Million ops) if you want to visually target a goal
    # fig_thp.add_hline(y=1e8, line_dash="dot", annotation_text="100M Target")

    fig_thp.update_layout(
        xaxis_title="Number of Contacts",
        yaxis_title="Contacts / Second (Higher is Better)",
        template="plotly_dark",
        hovermode="x unified",
    )

    fig_thp.write_html("benchmark_throughput.html")
    print("-> Generated: benchmark_throughput.html")

    # ==============================================================================
    # PLOT 3: The "Cost of Contention" (Bar Chart at Max Scale)
    # ==============================================================================
    # Isolates the impact of atomic collisions at the highest workload.

    max_n = df["N_CONTACTS"].max()
    df_subset = df[df["N_CONTACTS"] == max_n]

    fig_bar = px.bar(
        df_subset,
        x="Scenario",
        y="Time_ms",
        color="Graph_Mode",
        barmode="group",
        title=f"Contention Impact at N_CONTACTS = {max_n}",
        text_auto=".2f",
    )

    fig_bar.update_layout(yaxis_title="Time (ms)", template="plotly_dark")

    fig_bar.write_html("benchmark_contention_cost.html")
    print("-> Generated: benchmark_contention_cost.html")

    # ==============================================================================
    # PLOT 4: Combined Dashboard (Subplots)
    # ==============================================================================
    # A single view for convenience

    fig_dash = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Execution Time (Log scale)", "Throughput (Linear scale)"),
        horizontal_spacing=0.1,
    )

    # We iterate manually to add traces to subplots (Plotly Express specific quirk)
    # Ideally, just use the HTMLs above, but here is how to combine them:

    # (Skipping complex manual trace addition for brevity, the separate HTMLs are usually better
    # to prevent legend clutter).


if __name__ == "__main__":
    generate_interactive_report("results.csv")
