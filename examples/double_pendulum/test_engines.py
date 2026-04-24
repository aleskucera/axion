"""Compare AxionEngine, HybridGPTEngine (warm-start off/on), and RepeatedAxionEngine on identical ICs.

Reads N environments from a dataset HDF5 file, runs all engines headless,
and writes per-step solver diagnostics (iter_count, initial guesses,
converged values) to a new HDF5 file.

Usage:
    python test_engines.py --num-runs 5 --num-steps 300 \
        --input-hdf5 ../../src/axion/neural_solver/datasets/Pendulum/pendulumContactsValid250klen500envs250seed1.hdf5 \
        --output-hdf5 ../../data/engine_comparison_YYYYMMDD_HHMMSS.hdf5 \
        --hybrid-mode neural-warm-start-forces

    # Optional: disable tilted contact plane construction in the model
    python test_engines.py --no-contacts
"""
from __future__ import annotations
import argparse
import pathlib
from datetime import datetime
from collections import defaultdict
from typing import override

import h5py
import hydra.utils
import numpy as np
import newton
import warp as wp
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from axion import EngineConfig, ExecutionConfig, InteractiveSimulator
from axion import LoggingConfig, RenderingConfig, SimulationConfig
from axion.core.hybrid_gpt_engine import HybridGPTEngine
from axion.core.repeated_engine import RepeatedAxionEngine
from pendulum_articulation_definition import build_pendulum_model
from pendulum_utils import set_tilted_plane_from_coefficients

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "conf"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_INPUT_HDF5 = (
    REPO_ROOT
    / "src"
    / "axion"
    / "neural_solver"
    / "datasets"
    / "Pendulum"
    / "pendulumContactsValid250klen500envs250seed1.hdf5"
)
def default_output_hdf5_path() -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "data" / f"engine_comparison_{timestamp}.hdf5"


# ---------------------------------------------------------------------------
# Hydra config loading (no @hydra.main decorator)
# ---------------------------------------------------------------------------

def load_hydra_config(config_name: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(CONFIG_PATH.resolve()), version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def instantiate_configs(cfg):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    return sim_config, render_config, exec_config, engine_config, logging_config


# ---------------------------------------------------------------------------
# Generalized -> maximal coordinate conversion
# ---------------------------------------------------------------------------

def generalized_to_maximal(
    model: newton.Model,
    state: newton.State,
    q0: float,
    q1: float,
    qd0: float = 0.0,
    qd1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    device = state.joint_q.device
    state.joint_q.assign(wp.array([q0, q1], dtype=wp.float32, device=device))
    state.joint_qd.assign(wp.array([qd0, qd1], dtype=wp.float32, device=device))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q_np = state.body_q.numpy().reshape(-1, 7)
    body_qd_np = state.body_qd.numpy().reshape(-1, 6)
    return body_q_np, body_qd_np


# ---------------------------------------------------------------------------
# Simulator with per-step data capture
# ---------------------------------------------------------------------------

class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        plane_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0),
        with_contacts: bool = True,
        initial_state: tuple[float, float, float, float] | None = None,
        captured_data: dict | None = None,
    ):
        self.plane_coefficients = plane_coefficients
        self.with_contacts = with_contacts
        self.captured_data = captured_data
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        if initial_state is not None:
            q0, q1, qd0, qd1 = initial_state
            generalized_to_maximal(
                self.model, self.current_state,
                q0=q0, q1=q1, qd0=qd0, qd1=qd1,
            )

    @override
    def control_policy(self, state: newton.State):
        pass

    @override
    def _render(self, segment_num: int):
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.end_frame()

    @override
    def _single_physics_step(self, step_num: int):
        super()._single_physics_step(step_num)

        if self.captured_data is not None:
            wp.synchronize()
            d = self.solver.data
            self.captured_data["iter_count"].append(d.iter_count.numpy().copy())
            self.captured_data["init_guess_body_pose"].append(d.init_guess_body_pose.numpy().copy())
            self.captured_data["init_guess_body_vel"].append(d.init_guess_body_vel.numpy().copy())
            self.captured_data["init_guess_constr_force"].append(d._init_guess_constr_force.numpy().copy())
            # Final converged values after the Newton solve for this timestep.
            self.captured_data["converged_body_pose"].append(d.body_pose.numpy().copy())
            self.captured_data["converged_body_vel"].append(d.body_vel.numpy().copy())
            self.captured_data["converged_constr_force"].append(d._constr_force.numpy().copy())
            # Backward-compatible aliases used by existing analysis scripts.
            self.captured_data["body_pose"].append(d.body_pose.numpy().copy())
            self.captured_data["body_vel"].append(d.body_vel.numpy().copy())
            self.captured_data["constr_force"].append(d._constr_force.numpy().copy())
            self.captured_data["body_pose_prev"].append(d.body_pose_prev.numpy().copy())
            self.captured_data["body_vel_prev"].append(d.body_vel_prev.numpy().copy())
            self.captured_data["constr_force_prev_iter"].append(d._constr_force_prev_iter.numpy().copy())

            if isinstance(self.solver, HybridGPTEngine):
                pred_state = getattr(self.solver, "last_predicted_next_state", None)
                pred_lambda = getattr(self.solver, "last_predicted_next_lambdas", None)
                pred_body_pose = getattr(self.solver, "last_predicted_next_body_pose", None)
                pred_body_vel = getattr(self.solver, "last_predicted_next_body_vel", None)
                if pred_state is not None:
                    self.captured_data["hybrid_predicted_next_state"].append(pred_state.copy())
                if pred_lambda is not None:
                    self.captured_data["hybrid_predicted_next_lambdas"].append(pred_lambda.copy())
                if pred_body_pose is not None:
                    self.captured_data["hybrid_predicted_next_body_pose"].append(
                        pred_body_pose.copy()
                    )
                if pred_body_vel is not None:
                    self.captured_data["hybrid_predicted_next_body_vel"].append(
                        pred_body_vel.copy()
                    )
            # RepeatedAxionEngine: warm-start guess for NR#2
            if isinstance(self.solver, RepeatedAxionEngine):
                rlog = getattr(self.solver, "_repeated_step_log", None) or {}
                for key, arr in rlog.items():
                    self.captured_data[key].append(arr)

    def build_model(self) -> newton.Model:
        model = build_pendulum_model(
            num_worlds=1,
            device="cuda:0",
            with_contacts=self.with_contacts,
        )
        if self.with_contacts:
            a, b, c, d = self.plane_coefficients
            set_tilted_plane_from_coefficients(model, a, b, c, d, world_idx=0)
        return model

# ---------------------------------------------------------------------------
# Read dataset
# ---------------------------------------------------------------------------

def read_dataset(path: pathlib.Path, num_runs: int):
    # HDF5 layout: (timesteps, environments, 4)
    # Read timestep 0 (initial state) for the first num_runs environments.
    with h5py.File(path, "r") as f:
        states = f["data/states"][0, :num_runs, :]
        plane_coeffs = f["data/plane_coefficients"][0, :num_runs, :]
    return states.astype(np.float64), plane_coeffs.astype(np.float64)


# ---------------------------------------------------------------------------
# Run one engine over all N environments
# ---------------------------------------------------------------------------

HYDRA_OVERRIDES = [
    "rendering=headless",
    "execution=no_graph",
    "logging=disabled",
]


def run_engine(
    config_name: str,
    engine_label: str,
    initial_states: np.ndarray,
    plane_coefficients: np.ndarray,
    num_steps: int,
    dt: float,
    with_contacts: bool,
    engine_overrides: list[str] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Return a list (one per run) of dicts mapping field name -> (num_steps, ...) array."""
    duration = num_steps * dt
    overrides = HYDRA_OVERRIDES + [
        f"simulation.duration_seconds={duration}",
        f"simulation.target_timestep_seconds={dt}",
    ]
    if engine_overrides:
        overrides.extend(engine_overrides)
    cfg = load_hydra_config(config_name, overrides)
    sim_cfg, render_cfg, exec_cfg, engine_cfg, log_cfg = instantiate_configs(cfg)

    num_runs = len(initial_states)
    all_runs: list[dict[str, np.ndarray]] = []

    for i in tqdm(range(num_runs), desc=f"{engine_label}"):
        captured: dict[str, list] = defaultdict(list)
        ic = tuple(initial_states[i].tolist())
        pc = tuple(plane_coefficients[i].tolist())
        print(
            f"[{engine_label}] run {i:03d}: "
            f"initial_state(q0,q1,qd0,qd1)={ic}, "
            f"plane_coefficients(a,b,c,d)={pc}"
        )

        sim = Simulator(
            sim_config=sim_cfg,
            render_config=render_cfg,
            exec_config=exec_cfg,
            engine_config=engine_cfg,
            logging_config=log_cfg,
            plane_coefficients=pc,
            with_contacts=with_contacts,
            initial_state=ic,
            captured_data=captured,
        )
        sim.run()

        run_data = {k: np.stack(v, axis=0) for k, v in captured.items()}
        all_runs.append(run_data)

    return all_runs


def hybrid_engine_entry(hybrid_mode: str) -> tuple[str, str, list[str]]:
    if hybrid_mode == "no-warm-start-forces":
        return (
            "hybrid_gpt_pendulum",
            "hybrid_gpt_engine_no_warm_start_forces",
            [
                "+engine.use_warm_start_forces=false",
                "+engine.use_neural_lambda_init=false",
            ],
        )
    if hybrid_mode == "calculated-warm-start-forces":
        return (
            "hybrid_gpt_pendulum",
            "hybrid_gpt_engine_calculated_warm_start_forces",
            [
                "+engine.use_warm_start_forces=true",
                "+engine.use_neural_lambda_init=false",
            ],
        )
    if hybrid_mode == "neural-warm-start-forces":
        return (
            "hybrid_gpt_pendulum",
            "hybrid_gpt_engine_neural_warm_start_forces",
            [
                "+engine.use_warm_start_forces=false",
                "+engine.use_neural_lambda_init=true",
            ],
        )
    raise ValueError(f"Unsupported hybrid mode: {hybrid_mode}")


# ---------------------------------------------------------------------------
# Write output HDF5
# ---------------------------------------------------------------------------

def write_output_hdf5(
    path: pathlib.Path,
    engines: dict[str, list[dict[str, np.ndarray]]],
    initial_states: np.ndarray,
    plane_coefficients: np.ndarray,
    num_steps: int,
    dt: float,
    source_dataset: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to {path} ...")

    with h5py.File(path, "w") as f:
        f.attrs["num_runs"] = len(initial_states)
        f.attrs["num_steps"] = num_steps
        f.attrs["dt"] = dt
        f.attrs["source_dataset"] = source_dataset

        for engine_name, runs in engines.items():
            eng_grp = f.create_group(engine_name)
            for i, run_data in enumerate(runs):
                run_grp = eng_grp.create_group(f"run_{i:03d}")
                run_grp.attrs["initial_state"] = initial_states[i]
                run_grp.attrs["plane_coefficients"] = plane_coefficients[i]

                for key, arr in run_data.items():
                    run_grp.create_dataset(
                        key, data=arr, compression="gzip", compression_opts=4
                    )

    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare AxionEngine, HybridGPTEngine (warm-start on/off), "
            "and RepeatedAxionEngine"
        )
    )
    p.add_argument(
        "--num-runs", type=int, default=5,
        help="Number of environments (initial conditions) to test",
    )
    p.add_argument(
        "--num-steps", type=int, default=300,
        help="Number of simulation steps per run",
    )
    p.add_argument(
        "--dt", type=float, default=0.01,
        help="Simulation timestep (seconds)",
    )
    p.add_argument(
        "--input-hdf5", type=str, default=str(DEFAULT_INPUT_HDF5),
        help="Path to the source dataset HDF5 file",
    )
    p.add_argument(
        "--output-hdf5", type=str, default=str(default_output_hdf5_path()),
        help="Path for the output comparison HDF5 file",
    )
    p.add_argument(
        "--no-contacts",
        action="store_true",
        help="Disable tilted contact-plane construction in the pendulum model.",
    )
    p.add_argument(
        "--hybrid-mode",
        type=str,
        default="neural-warm-start-forces",
        choices=[
            "no-warm-start-forces",
            "calculated-warm-start-forces",
            "neural-warm-start-forces",
        ],
        help=(
            "Hybrid engine force initialization mode: "
            "'no-warm-start-forces' (zero init), "
            "'calculated-warm-start-forces' (analytical warm start), or "
            "'neural-warm-start-forces' (neural lambda init)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path = pathlib.Path(args.input_hdf5)
    output_path = pathlib.Path(args.output_hdf5)

    print(f"Reading {args.num_runs} environments from {input_path} ...")
    initial_states, plane_coefficients = read_dataset(input_path, args.num_runs)

    hybrid_config_name, hybrid_label, hybrid_overrides = hybrid_engine_entry(args.hybrid_mode)
    engines_config = [
        ("pendulum", "axion_engine", None),
        (hybrid_config_name, hybrid_label, hybrid_overrides),
        ("repeated_pendulum", "repeated_axion_engine", None),
    ]

    results: dict[str, list[dict[str, np.ndarray]]] = {}
    for config_name, engine_label, engine_overrides in engines_config:
        print(f"\n{'='*60}")
        print(f"Running {engine_label} ({config_name}) ...")
        print(f"{'='*60}")
        results[engine_label] = run_engine(
            config_name=config_name,
            engine_label=engine_label,
            initial_states=initial_states,
            plane_coefficients=plane_coefficients,
            num_steps=args.num_steps,
            dt=args.dt,
            with_contacts=not args.no_contacts,
            engine_overrides=engine_overrides,
        )

    write_output_hdf5(
        path=output_path,
        engines=results,
        initial_states=initial_states,
        plane_coefficients=plane_coefficients,
        num_steps=args.num_steps,
        dt=args.dt,
        source_dataset=str(input_path),
    )


if __name__ == "__main__":
    main()
