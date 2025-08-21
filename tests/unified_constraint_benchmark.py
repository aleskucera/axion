import time

import matplotlib.pyplot as plt
import pandas as pd
import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.constraints import friction_constraint_kernel
from axion.constraints import joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from contact_kernel_benchmark import setup_data as setup_contact_data
from dynamics_kernel_benchmark import setup_data as setup_dynamics_data
from friction_kernel_benchmark import setup_data as setup_friction_data
from joint_kernel_benchmark import setup_data as setup_joint_data


def setup_complete_simulation_data(
    num_bodies,
    num_contacts,
    num_joints,
    device,
    contact_kwargs={},
    friction_kwargs={},
    joint_kwargs={},
):
    """
    Setup complete simulation data using the proven working setup functions.
    """
    data = {}

    # Always setup dynamics data
    dynamics_data = setup_dynamics_data(num_bodies, device)
    data.update(dynamics_data)

    # Setup contact and friction data if contacts exist
    if num_contacts > 0:
        contact_data = setup_contact_data(
            num_bodies, num_contacts, device, **contact_kwargs
        )
        data.update(contact_data)

        # Setup friction data (reuse contact data setup with friction kwargs)
        friction_data = setup_friction_data(
            num_bodies, num_contacts, device, **friction_kwargs
        )
        # Update with friction-specific data
        data["lambda_f"] = friction_data["lambda_f"]
        data["lambda_n_prev"] = friction_data["lambda_n_prev"]
        data["h_f"] = friction_data["h_f"]
        data["J_f_values"] = friction_data["J_f_values"]
        data["C_f_values"] = friction_data["C_f_values"]

    # Setup joint data if joints exist
    if num_joints > 0:
        joint_data = setup_joint_data(num_bodies, num_joints, device, **joint_kwargs)
        # Use different keys to avoid conflicts with contact interactions
        data["joint_interactions"] = joint_data["interactions"]
        data["lambda_j"] = joint_data["lambda_j"]
        data["joint_stabilization_factor"] = joint_data["joint_stabilization_factor"]
        data["h_j"] = joint_data["h_j"]
        data["J_j_values"] = joint_data["J_j_values"]
        data["C_j_values"] = joint_data["C_j_values"]

    return data


def benchmark_complete_constraint_step(
    num_bodies,
    num_contacts,
    num_joints,
    num_iterations=200,
    device=None,
    **kwargs,
):
    """Benchmark the complete constraint solving step (all kernels together)."""
    device = device or wp.get_device()
    print(
        f"\nBenchmarking pipeline: N_b={num_bodies}, N_c={num_contacts}, N_j={num_joints}"
    )

    data = setup_complete_simulation_data(
        num_bodies,
        num_contacts,
        num_joints,
        device,
        contact_kwargs=kwargs.get("contact_kwargs", {}),
        friction_kwargs=kwargs.get("friction_kwargs", {}),
        joint_kwargs=kwargs.get("joint_kwargs", {}),
    )

    def run_complete_step():
        """Run all constraint kernels in sequence."""
        data["g"].zero_()

        # 1. Unconstrained dynamics
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=[
                data["body_qd"],
                data["body_qd_prev"],
                data["body_f"],
                data["gen_mass"],
                data["dt"],
                data["g_accel"],
                data["g"],
            ],
            device=device,
        )

        # 2. Contact constraints
        if num_contacts > 0:
            wp.launch(
                kernel=contact_constraint_kernel,
                dim=num_contacts,
                inputs=[
                    data["body_qd"],
                    data["body_qd_prev"],
                    data["interactions"],
                    data["lambda_n"],
                    data["dt"],
                    data["stabilization_factor"],
                    data["fb_alpha"],
                    data["fb_beta"],
                    data["compliance"],
                    data["g"],
                    data["h_n"],
                    data["J_n_values"],
                    data["C_n_values"],
                ],
                device=device,
            )

        # 3. Friction constraints
        if num_contacts > 0:
            wp.launch(
                kernel=friction_constraint_kernel,
                dim=num_contacts,
                inputs=[
                    data["body_qd"],
                    data["interactions"],
                    data["lambda_f"],
                    data["lambda_n_prev"],
                    data["fb_alpha"],
                    data["fb_beta"],
                    data["compliance"],
                    data["g"],
                    data["h_f"],
                    data["J_f_values"],
                    data["C_f_values"],
                ],
                device=device,
            )

        # 4. Joint constraints
        if num_joints > 0:
            wp.launch(
                kernel=joint_constraint_kernel,
                dim=(5, num_joints),
                inputs=[
                    data["body_qd"],
                    data["lambda_j"],
                    data["joint_interactions"],  # Use the renamed key
                    data["dt"],
                    data["joint_stabilization_factor"],
                    data["g"],
                    data["h_j"],
                    data["J_j_values"],
                    data["C_j_values"],
                ],
                device=device,
            )

    # Warm-up compile
    run_complete_step()
    wp.synchronize()

    # Benchmark standard launch
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        run_complete_step()
    wp.synchronize()
    standard_time = (time.perf_counter() - start_time) / num_iterations * 1000

    # Benchmark CUDA graph
    graph_time, speedup = None, None
    if device.is_cuda:
        try:
            wp.capture_begin()
            run_complete_step()
            graph = wp.capture_end()
            wp.capture_launch(graph)
            wp.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_iterations):
                wp.capture_launch(graph)
            wp.synchronize()
            graph_time = (time.perf_counter() - start_time) / num_iterations * 1000
            if graph_time > 0:
                speedup = standard_time / graph_time
        except Exception as e:
            print(f"CUDA graph capture failed: {e}")
            graph_time, speedup = None, None

    # Collect and return results
    result_data = {
        "standard_time": standard_time,
        "graph_time": graph_time,
        "speedup": speedup,
        "num_bodies": num_bodies,
        "num_contacts": num_contacts,
        "num_joints": num_joints,
    }
    result_data.update(kwargs)
    return result_data


def generate_plots(df, save_prefix="unified_benchmark"):
    """Generate comprehensive benchmark plots from a pandas DataFrame."""
    print("\nGenerating plots and summary...")

    # Create figure with subplots - similar to matplotlib documentation structure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Unified Constraint Pipeline Performance", fontsize=16)

    # Plot 1: Performance Scaling
    scaling_scenarios = [s for s in df["name"].unique() if "System" in s]
    scaling_data = df[df["name"].isin(scaling_scenarios)].sort_values("num_bodies")

    if not scaling_data.empty:
        ax1.plot(
            scaling_data["num_bodies"],
            scaling_data["standard_time"],
            "o-",
            label="Standard Launch",
            linewidth=2,
            markersize=6,
        )
        if scaling_data["graph_time"].notna().any():
            valid_graph_data = scaling_data[scaling_data["graph_time"].notna()]
            ax1.plot(
                valid_graph_data["num_bodies"],
                valid_graph_data["graph_time"],
                "s-",
                label="CUDA Graph",
                linewidth=2,
                markersize=6,
            )

    ax1.set_title("Performance Scaling vs. Body Count")
    ax1.set_xlabel("Number of Bodies")
    ax1.set_ylabel("Time per Pipeline (ms)")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Plot 2: Performance by Scenario
    scenario_names = df["name"].unique()
    scenario_times = df.groupby("name")["standard_time"].mean()

    bars = ax2.bar(range(len(scenario_names)), scenario_times.values, alpha=0.8)
    ax2.set_title("Performance by Scenario")
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Time per Pipeline (ms)")
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, rotation=45, ha="right")

    # Add value labels on bars
    for bar, time_val in zip(bars, scenario_times.values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{time_val:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: CUDA Graph Speedup
    gpu_data = df[df["speedup"].notna()].copy()
    if not gpu_data.empty:
        gpu_data = gpu_data.sort_values("speedup", ascending=False)
        bars = ax3.bar(
            range(len(gpu_data)), gpu_data["speedup"], alpha=0.8, color="green"
        )
        ax3.axhline(1.0, color="red", linestyle="--", label="No Speedup", alpha=0.7)
        ax3.set_title("CUDA Graph Speedup")
        ax3.set_xlabel("Scenario")
        ax3.set_ylabel("Speedup Factor")
        ax3.set_xticks(range(len(gpu_data)))
        ax3.set_xticklabels(gpu_data["name"], rotation=45, ha="right")
        ax3.legend()

        # Add speedup values on bars
        for bar, speedup_val in zip(bars, gpu_data["speedup"]):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{speedup_val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No GPU speedup data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=14,
        )
        ax3.set_title("CUDA Graph Speedup (No Data)")

    # Plot 4: Time vs Total Constraints
    df["total_constraints"] = df["num_contacts"] * 3 + df["num_joints"] * 5
    valid_constraint_data = df[df["total_constraints"] > 0]

    if not valid_constraint_data.empty:
        scatter = ax4.scatter(
            valid_constraint_data["total_constraints"],
            valid_constraint_data["standard_time"],
            c=valid_constraint_data["num_bodies"],
            cmap="viridis",
            s=80,
            alpha=0.7,
        )
        ax4.set_title("Performance vs. Total Constraints")
        ax4.set_xlabel("Approx. Number of Constraint Rows")
        ax4.set_ylabel("Time per Pipeline (ms)")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.grid(True, which="both", ls="--", alpha=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Number of Bodies")
    else:
        ax4.text(
            0.5,
            0.5,
            "No constraint data available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=14,
        )
        ax4.set_title("Performance vs. Total Constraints (No Data)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save results
    df.to_csv(f"{save_prefix}_results.csv", index=False)
    print(f"Detailed results saved to {save_prefix}_results.csv")


def main():
    wp.init()
    device = wp.get_device()
    print(f"Running Unified Benchmark on device: {device.name}")
    print("=" * 80)

    scenarios = [
        {
            "name": "Small System",
            "num_bodies": 100,
            "num_contacts": 200,
            "num_joints": 99,
        },
        {
            "name": "Medium System",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
        },
        {
            "name": "Large System",
            "num_bodies": 1000,
            "num_contacts": 2000,
            "num_joints": 999,
        },
        {
            "name": "X-Large System",
            "num_bodies": 2000,
            "num_contacts": 4000,
            "num_joints": 1999,
        },
        {
            "name": "Divergent Contacts",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "contact_kwargs": {"inactive_contact_ratio": 0.5},
            "friction_kwargs": {"inactive_contact_ratio": 0.5},
        },
        {
            "name": "Fixed Bodies",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "contact_kwargs": {"fixed_body_ratio": 0.2},
            "friction_kwargs": {"fixed_body_ratio": 0.2},
        },
        {
            "name": "Disabled Joints",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "joint_kwargs": {"disabled_joint_ratio": 0.5},
        },
        {
            "name": "Joints-Only System",
            "num_bodies": 400,
            "num_contacts": 0,
            "num_joints": 399,
        },
        {
            "name": "Contacts-Only System",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 0,
        },
    ]

    results = []
    for scenario in scenarios:
        try:
            result = benchmark_complete_constraint_step(
                device=device, num_iterations=200, **scenario
            )
            result["name"] = scenario["name"]  # Add scenario name to result
            results.append(result)
            print(f"  > {scenario['name']}: {result['standard_time']:.3f} ms", end="")
            if result["graph_time"]:
                print(
                    f" | Graph: {result['graph_time']:.3f} ms | Speedup: {result['speedup']:.2f}x"
                )
            else:
                print()
        except Exception as e:
            print(f"  > {scenario['name']}: FAILED - {e}")

    if results:
        results_df = pd.DataFrame(results)
        generate_plots(results_df, "unified_pipeline_benchmark")

        print("\n" + "=" * 80)
        print("UNIFIED PIPELINE BENCHMARK SUMMARY")
        print("=" * 80)

        # Display summary table
        summary_cols = [
            "name",
            "num_bodies",
            "num_contacts",
            "num_joints",
            "standard_time",
            "graph_time",
            "speedup",
        ]

        display_df = results_df[summary_cols].copy()
        for col in ["standard_time", "graph_time", "speedup"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)

        print(display_df.to_string(index=False))
    else:
        print("No successful benchmark results to display!")


if __name__ == "__main__":
    main()
