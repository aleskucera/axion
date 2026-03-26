import argparse
import pathlib
from collections import defaultdict
from typing import override
from copy import deepcopy

import h5py
import numpy as np
import yaml
import torch
import warp as wp
from matplotlib import pyplot as plt
from tqdm import tqdm

import newton
from axion import (
    AxionEngineConfig,
    GNNEngineConfig,
    RepeatedAxionEngineConfig,
    ExecutionConfig,
    InteractiveSimulator,
    LoggingConfig,
    RenderingConfig,
    SimulationConfig,
)
from axion.core.base_engine import AxionEngineBase
from axion.generation.scene_generator_new import SceneGenerator


class BenchmarkSimulator(InteractiveSimulator):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        execution_config: ExecutionConfig,
        engine_config,
        logging_config: LoggingConfig,
        seed: int,
        captured_data: dict,
    ):
        self.seed = seed
        self.captured_data = captured_data
        super().__init__(
            simulation_config,
            rendering_config,
            execution_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2
        self.builder.add_ground_plane()
        np.random.seed(self.seed)
        gen = SceneGenerator(self.builder, seed=self.seed)
        num_objects = np.random.randint(3, 15)
        for _ in range(num_objects):
            gen.generate_random_object(
                pos_bounds=((-1, -1, 0), (1, 1, 3)),
                density_bounds=(10.0, 100.0),
                size_bounds=(0.1, 0.3),
            )
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    @override
    def _single_physics_step(self, step_num: int):
        super()._single_physics_step(step_num)
        wp.synchronize()
        iter_count = self.solver.data.iter_count.numpy().copy()
        self.captured_data["iter_count"].append(int(iter_count[0]))
        residuals = self.solver.data.candidates_res_norm_sq.numpy().copy()
        self.captured_data["residuals"].append(np.sqrt(residuals))

    @override
    def run(self):
        try:
            segment_num = 0
            while self.viewer.is_running():
                self._run_simulation_segment(segment_num)
                segment_num += 1
                self._render(segment_num)
        finally:
            self.solver.save_logs()


def plot_convergence_probability(ax, residuals, label, color, atol=1e-3):
    num_trajectories = residuals.shape[0]
    converged = residuals < atol
    cnt = np.sum(converged, axis=0)
    prob = (cnt / num_trajectories) * 100.0
    x = np.arange(0, len(prob))
    ax.plot(x, prob, color=color, linewidth=2, label=label)
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Convergence Probability (%)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_ylim([0, 105])


def plot_median_residual(ax, residuals, label, color):
    median = np.median(residuals, axis=0)
    p25 = np.percentile(residuals, 25, axis=0)
    p75 = np.percentile(residuals, 75, axis=0)
    x = np.arange(0, len(median))
    ax.plot(x, median, color=color, linewidth=2, label=label)
    ax.fill_between(x, p25, p75, color=color, alpha=0.2, label=f"{label} IQR")
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Residual Norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def plot_spaghetti(ax, residuals, label, color):
    for run in residuals:
        ax.plot(run, color=color, alpha=0.1, linewidth=1)
    ax.plot([], [], color=color, label=label)
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Residual Norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 20)


def plot_results(input_h5: pathlib.Path, engines: dict, output_png: pathlib.Path = None):
    print(f"Loading results from {input_h5}...")
    residuals_by_engine = {}
    engine_names = list(engines.keys())
    with h5py.File(input_h5, "r") as f:
        for engine_name in engine_names:
            res = f[engine_name]["residuals"][:]
            res = res.transpose(0, 1, 3, 2).reshape(-1, res.shape[2])
            residuals_by_engine[engine_name] = res

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), constrained_layout=True)

    # Plot 1: Convergence Probability
    for engine_name in engine_names:
        plot_convergence_probability(
            axes[0],
            residuals_by_engine[engine_name],
            engines[engine_name]["label"],
            engines[engine_name]["color"],
        )
    axes[0].set_title("Convergence Probability")
    axes[0].legend(loc="lower right", fontsize=10)

    # Plot 2: Average Residual (log scale)
    for engine_name in engine_names:
        plot_median_residual(
            axes[1],
            residuals_by_engine[engine_name],
            engines[engine_name]["label"],
            engines[engine_name]["color"],
        )
    axes[1].set_title("Median Residual per Iteration")
    axes[1].legend(loc="upper right", fontsize=10)

    # Plot 3: Convergence Rate
    for engine_name in engine_names:
        plot_spaghetti(
            axes[2],
            residuals_by_engine[engine_name],
            engines[engine_name]["label"],
            engines[engine_name]["color"],
        )
    axes[2].set_title("Raw data")
    axes[2].legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "Newton Convergence Comparison: GNN Engine vs Axion Engine",
        fontsize=14,
        fontweight="bold",
    )

    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_png}")
    else:
        plt.show()

    plt.close()


def run_benchmark(
    num_scenes: int, model_path: pathlib.Path, output_h5: pathlib.Path, engines: dict
):
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path.parent.parent / "metadata.yaml", "r") as f:
        model_metadata = yaml.safe_load(f)
        duration_seconds = model_metadata["simulation_duration"]
        dt = model_metadata["timestep"]
        num_worlds = 1

    sim_cfg = SimulationConfig(
        duration_seconds=duration_seconds, target_timestep_seconds=dt, num_worlds=num_worlds
    )
    render_cfg = RenderingConfig(vis_type="null")
    exec_cfg = ExecutionConfig(use_cuda_graph=False)
    log_cfg = LoggingConfig()

    results = defaultdict(dict)
    for engine_name, engine_cfg in engines.items():
        print(f"Running {engine_name}")
        all_iter_counts = []
        all_residuals = []
        for seed in tqdm(range(num_scenes), desc=engine_name):
            captured = defaultdict(list)
            sim = BenchmarkSimulator(
                simulation_config=sim_cfg,
                rendering_config=render_cfg,
                execution_config=exec_cfg,
                engine_config=engine_cfg["config"],
                logging_config=log_cfg,
                seed=seed,
                captured_data=captured,
            )
            sim.run()
            all_iter_counts.append(np.array(captured["iter_count"]))
            all_residuals.append(np.array(captured["residuals"]))
        results[engine_name]["iters"] = all_iter_counts
        results[engine_name]["residuals"] = all_residuals

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("num_scenes", data=num_scenes)
        f.create_dataset("duration_seconds", data=duration_seconds)
        f.create_dataset("dt", data=dt)
        f.create_dataset("model_path", data=str(model_path))
        for engine_name in results.keys():
            grp = f.create_group(engine_name, track_order=True)
            grp.create_dataset(
                "iters", data=np.array(results[engine_name]["iters"])
            )  # (num_scenes, num_steps)
            grp.create_dataset(
                "residuals", data=np.array(results[engine_name]["residuals"])
            )  # (num_scenes, num_steps, newton_iters, num_worlds)
    print(f"Done. Results saved to {output_h5}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=5)
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default=pathlib.Path("data/gnn_data/dataset/models/first_model.pt"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("tmp_benchmark/gnn_vs_axion_benchmark_1.h5"),
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
    )
    parser.add_argument(
        "--save_png_path",
        type=pathlib.Path,
        default=None,
    )
    args = parser.parse_args()

    engines = {
        "axion_engine": {
            "config": AxionEngineConfig(max_newton_iters=100),
            "color": "tab:blue",
            "label": "Axion Engine",
        },
        "repeated_axion_engine": {
            "config": RepeatedAxionEngineConfig(max_newton_iters=100),
            "color": "tab:red",
            "label": "Repeated Axion Engine",
        },
        "gnn_engine": {
            "config": GNNEngineConfig(max_newton_iters=100, model_path=str(args.model_path)),
            "color": "tab:green",
            "label": "GNN Engine",
        },
    }

    if not args.plot_only:
        run_benchmark(
            num_scenes=args.num_scenes,
            output_h5=args.output,
            model_path=args.model_path,
            engines=engines,
        )

    plot_results(args.output, engines=engines, output_png=args.save_png_path)


if __name__ == "__main__":
    main()
