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


class ConstraintBenchmarkSuite:
    """Unified benchmark suite for all constraint kernels."""

    def __init__(self, device=None):
        wp.init()
        self.device = device or wp.get_device()
        self.results = []

    def benchmark_contact_kernel(
        self, num_bodies, num_contacts, num_iterations=200, **kwargs
    ):
        """Benchmark contact constraint kernel."""
        data = setup_contact_data(num_bodies, num_contacts, self.device, **kwargs)

        # Updated kernel arguments to use the ContactInteraction struct
        kernel_args = [
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
        ]

        return self._time_kernel(
            contact_constraint_kernel,
            num_contacts,
            kernel_args,
            num_iterations,
            "contact",
            data["g"],
        )

    def benchmark_friction_kernel(
        self, num_bodies, num_contacts, num_iterations=200, **kwargs
    ):
        """Benchmark friction constraint kernel."""
        data = setup_friction_data(num_bodies, num_contacts, self.device, **kwargs)

        # Updated kernel arguments to use the ContactInteraction struct
        kernel_args = [
            data["body_qd"],
            data["interactions"],
            data["lambda_f"],  # Renamed for clarity
            data["lambda_n_prev"],  # Renamed for clarity
            data["fb_alpha"],
            data["fb_beta"],
            data["compliance"],
            data["g"],
            data["h_f"],
            data["J_f_values"],
            data["C_f_values"],
        ]

        return self._time_kernel(
            friction_constraint_kernel,
            num_contacts,
            kernel_args,
            num_iterations,
            "friction",
            data["g"],
        )

    def benchmark_joint_kernel(
        self, num_bodies, num_joints, num_iterations=200, **kwargs
    ):
        """Benchmark joint constraint kernel."""
        data = setup_joint_data(num_bodies, num_joints, self.device, **kwargs)

        # Updated kernel arguments to use the JointInteraction struct
        kernel_args = [
            data["body_qd"],
            data["lambda_j"],
            data["interactions"],
            data["dt"],
            data["joint_stabilization_factor"],
            data["g"],
            data["h_j"],
            data["J_j_values"],
            data["C_j_values"],
        ]

        # Note: The kernel dim is now (5, num_joints) as per the new kernel.
        return self._time_kernel(
            joint_constraint_kernel,
            (5, num_joints),
            kernel_args,
            num_iterations,
            "joint",
            data["g"],
        )

    def benchmark_dynamics_kernel(self, num_bodies, num_iterations=200):
        """Benchmark unconstrained dynamics kernel."""
        data = setup_dynamics_data(num_bodies, self.device)

        # Updated kernel arguments to use the GeneralizedMass struct
        kernel_args = [
            data["body_qd"],
            data["body_qd_prev"],
            data["body_f"],
            data["gen_mass"],
            data["dt"],
            data["g_accel"],
            data["g"],
        ]

        return self._time_kernel(
            unconstrained_dynamics_kernel,
            num_bodies,
            kernel_args,
            num_iterations,
            "dynamics",
            data["g"],
        )

    def _time_kernel(self, kernel, dim, args, num_iterations, kernel_name, g_array):
        """Generic kernel timing function."""
        # Warm-up
        wp.launch(kernel=kernel, dim=dim, inputs=args, device=self.device)
        wp.synchronize()

        # Benchmark standard launch
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            g_array.zero_()
            wp.launch(kernel=kernel, dim=dim, inputs=args, device=self.device)
        wp.synchronize()
        standard_time = (time.perf_counter() - start_time) / num_iterations * 1000

        # Benchmark CUDA graph (if available)
        graph_time = None
        if self.device.is_cuda:
            wp.capture_begin()
            g_array.zero_()
            wp.launch(kernel=kernel, dim=dim, inputs=args, device=self.device)
            graph = wp.capture_end()

            wp.capture_launch(graph)
            wp.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_iterations):
                wp.capture_launch(graph)
            wp.synchronize()
            graph_time = (time.perf_counter() - start_time) / num_iterations * 1000

        return {
            "kernel": kernel_name,
            "standard_time": standard_time,
            "graph_time": graph_time,
            "speedup": standard_time / graph_time if graph_time else None,
        }

    def run_scaling_benchmark(self, scenarios):
        """Run benchmarks across different scaling scenarios."""
        print(f"Running benchmarks on device: {self.device.name}")
        print("=" * 80)

        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            print("-" * 40)

            # Dynamics kernel (always runs)
            dynamics_result = self.benchmark_dynamics_kernel(
                scenario["num_bodies"], scenario.get("num_iterations", 200)
            )
            dynamics_result.update(
                {
                    "scenario": scenario["name"],
                    "num_bodies": scenario["num_bodies"],
                    "num_contacts": scenario.get("num_contacts", 0),
                    "num_joints": scenario.get("num_joints", 0),
                }
            )
            self.results.append(dynamics_result)

            # Contact kernel (if contacts exist)
            if scenario.get("num_contacts", 0) > 0:
                contact_result = self.benchmark_contact_kernel(
                    scenario["num_bodies"],
                    scenario["num_contacts"],
                    scenario.get("num_iterations", 200),
                    **scenario.get("contact_kwargs", {}),
                )
                contact_result.update(
                    {
                        "scenario": scenario["name"],
                        "num_bodies": scenario["num_bodies"],
                        "num_contacts": scenario["num_contacts"],
                        "num_joints": scenario.get("num_joints", 0),
                    }
                )
                self.results.append(contact_result)

                # Friction kernel
                friction_result = self.benchmark_friction_kernel(
                    scenario["num_bodies"],
                    scenario["num_contacts"],
                    scenario.get("num_iterations", 200),
                    **scenario.get("friction_kwargs", {}),
                )
                friction_result.update(
                    {
                        "scenario": scenario["name"],
                        "num_bodies": scenario["num_bodies"],
                        "num_contacts": scenario["num_contacts"],
                        "num_joints": scenario.get("num_joints", 0),
                    }
                )
                self.results.append(friction_result)

            # Joint kernel (if joints exist)
            if scenario.get("num_joints", 0) > 0:
                joint_result = self.benchmark_joint_kernel(
                    scenario["num_bodies"],
                    scenario["num_joints"],
                    scenario.get("num_iterations", 200),
                    **scenario.get("joint_kwargs", {}),
                )
                joint_result.update(
                    {
                        "scenario": scenario["name"],
                        "num_bodies": scenario["num_bodies"],
                        "num_contacts": scenario.get("num_contacts", 0),
                        "num_joints": scenario["num_joints"],
                    }
                )
                self.results.append(joint_result)

            # Print results for this scenario
            for result in self.results[-4:]:  # Last few results
                if result["scenario"] == scenario["name"]:
                    print(
                        f"{result['kernel']:>12}: {result['standard_time']:>8.3f} ms",
                        end="",
                    )
                    if result["graph_time"]:
                        print(
                            f" | Graph: {result['graph_time']:>8.3f} ms | Speedup: {result['speedup']:>5.2f}x"
                        )
                    else:
                        print()

    def generate_plots(self, save_prefix="constraint_benchmark"):
        """Generate comprehensive benchmark plots."""
        if not self.results:
            print("No results to plot!")
            return

        df = pd.DataFrame(self.results)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Constraint Kernel Performance Benchmarks", fontsize=16)

        # Plot 1: Performance by kernel type
        kernel_means = df.groupby("kernel")["standard_time"].mean().sort_values()
        ax1.bar(kernel_means.index, kernel_means.values, alpha=0.7)
        ax1.set_ylabel("Average Time (ms)")
        ax1.set_title("Average Performance by Kernel Type")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Scaling with problem size
        for kernel in df["kernel"].unique():
            kernel_data = df[df["kernel"] == kernel]
            if len(kernel_data) > 1:
                # Use total constraint count as x-axis
                x_vals = (
                    kernel_data["num_bodies"]
                    + kernel_data["num_contacts"]
                    + kernel_data["num_joints"]
                )
                ax2.plot(
                    x_vals, kernel_data["standard_time"], "o-", label=kernel, alpha=0.7
                )

        ax2.set_xlabel("Total Problem Size (Bodies + Contacts + Joints)")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Performance Scaling")
        ax2.legend()
        ax2.set_yscale("log")
        ax2.set_xscale("log")

        # Plot 3: CUDA Graph speedup (if available)
        gpu_results = df[df["graph_time"].notna()]
        if not gpu_results.empty:
            speedup_data = gpu_results.groupby("kernel")["speedup"].mean().sort_values()
            bars = ax3.bar(speedup_data.index, speedup_data.values, alpha=0.7)
            ax3.axhline(
                y=1.0, color="red", linestyle="--", alpha=0.5, label="No speedup"
            )
            ax3.set_ylabel("Speedup Factor")
            ax3.set_title("CUDA Graph Speedup by Kernel")
            ax3.tick_params(axis="x", rotation=45)
            ax3.legend()

            # Add speedup values on bars
            for bar, speedup in zip(bars, speedup_data.values):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "No GPU data available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
            )
            ax3.set_title("CUDA Graph Speedup (No Data)")

        # Plot 4: Performance comparison by scenario
        scenario_pivot = df.pivot_table(
            index="scenario", columns="kernel", values="standard_time", aggfunc="mean"
        ).fillna(0)

        scenario_pivot.plot(kind="bar", ax=ax4, alpha=0.7)
        ax4.set_ylabel("Time (ms)")
        ax4.set_title("Performance by Scenario")
        ax4.tick_params(axis="x", rotation=45)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Generate summary table
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        summary = (
            df.groupby("kernel")
            .agg(
                {
                    "standard_time": ["mean", "std", "min", "max"],
                    "speedup": ["mean", "std"],
                }
            )
            .round(3)
        )

        print(summary)

        # Save detailed results to CSV
        df.to_csv(f"{save_prefix}_results.csv", index=False)
        print(f"\nDetailed results saved to {save_prefix}_results.csv")


def main():
    """Main benchmark execution."""
    benchmark_suite = ConstraintBenchmarkSuite()

    # Define benchmark scenarios
    scenarios = [
        {
            "name": "Small System",
            "num_bodies": 100,
            "num_contacts": 200,
            "num_joints": 99,
            "num_iterations": 200,
        },
        {
            "name": "Medium System",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "num_iterations": 200,
        },
        {
            "name": "Large System",
            "num_bodies": 1000,
            "num_contacts": 2000,
            "num_joints": 999,
            "num_iterations": 200,
        },
        {
            "name": "Divergent Contacts",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "contact_kwargs": {"inactive_contact_ratio": 0.5},
            "friction_kwargs": {"inactive_contact_ratio": 0.5},
            "num_iterations": 200,
        },
        {
            "name": "Fixed Bodies",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "contact_kwargs": {"fixed_body_ratio": 0.2},
            "friction_kwargs": {"fixed_body_ratio": 0.2},
            "num_iterations": 200,
        },
        {
            "name": "Disabled Joints",
            "num_bodies": 400,
            "num_contacts": 800,
            "num_joints": 399,
            "joint_kwargs": {"disabled_joint_ratio": 0.5},
            "num_iterations": 200,
        },
    ]

    # Run all benchmarks
    benchmark_suite.run_scaling_benchmark(scenarios)

    # Generate plots and analysis
    benchmark_suite.generate_plots("combined_constraint_benchmark")


if __name__ == "__main__":
    main()
