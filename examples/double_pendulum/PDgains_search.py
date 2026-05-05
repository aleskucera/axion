"""Parallel PD gain grid search on the double pendulum (multi-world Axion simulation).

Runs 625 worlds (5^4 combinations of per-joint ke/kd), 10 episodes with shared random
initial conditions and episode-specific joint targets, records trajectories, and ranks
gains by mean settling time to the targets.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import override

import numpy as np
import newton
import warp as wp
from tqdm import tqdm
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import JointMode
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.base_simulator import BaseSimulator
from pendulum_articulation_definition import build_pendulum_model

# Match generate_dataset_pendulum joint target box
JOINT_TARGET_MIN = np.array([0.0, -np.pi / 3.0], dtype=np.float64)
JOINT_TARGET_MAX = np.array([np.pi, np.pi / 3.0], dtype=np.float64)

NUM_EPISODES = 5
NUM_STEPS = 1000
DT = 0.01
SETTLE_THRESH = 0.05
SEED = 42

# Same stiffness/damping spans as plan; 5 samples per axis → 5^4 worlds
KE_VALUES = np.linspace(350.0, 3500, 3)
KD_VALUES = np.linspace(35.0, 350, 3)
REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "PD_gains_search"


def make_gain_combinations() -> list[tuple[float, float, float, float]]:
    """Cartesian product (ke_q0, kd_q0, ke_q1, kd_q1)."""
    tuples: list[tuple[float, float, float, float]] = []
    for ke0, kd0, ke1, kd1 in itertools.product(
        KE_VALUES, KD_VALUES, KE_VALUES, KD_VALUES
    ):
        tuples.append((float(ke0), float(kd0), float(ke1), float(kd1)))
    return tuples


def compute_l1_tracking_error(
    states: np.ndarray,
    episode_targets: np.ndarray,
) -> np.ndarray:
    """Mean time-averaged L1 tracking error per world, averaged over episodes.

    states: (n_eps, n_steps, n_worlds, 4) with [q0, q1, qd0, qd1]
    episode_targets: (n_eps, 2)
    """
    trajs = states[..., :2]
    abs_err = np.abs(trajs - episode_targets[:, np.newaxis, np.newaxis, :])
    per_step_l1 = abs_err.sum(axis=-1)  # (n_eps, n_steps, n_worlds)
    return per_step_l1.mean(axis=(0, 1))


def compute_settling_times(
    states: np.ndarray,
    episode_targets: np.ndarray,
    dt: float,
    threshold: float,
) -> np.ndarray:
    """Mean settling time (seconds) per world, averaged over episodes."""
    trajs = states[..., :2]
    _, num_steps, num_worlds, _ = trajs.shape
    err = np.abs(trajs - episode_targets[:, np.newaxis, np.newaxis, :]).max(axis=-1)
    settled = err <= threshold
    steps_to_settle = np.full((trajs.shape[0], num_worlds), num_steps, dtype=np.float64)

    for e in range(trajs.shape[0]):
        for w in range(num_worlds):
            s = settled[e, :, w]
            suffix_ok = np.logical_and.accumulate(s[::-1])[::-1]
            if suffix_ok.any():
                steps_to_settle[e, w] = float(np.argmax(suffix_ok))

    return steps_to_settle.mean(axis=0) * dt


def compute_state_step_difference_stats(
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world max one-step differences for q and qd separately.

    Returns:
        max_step_diff_q: max |dq| across episodes, timesteps, and q dims (rad)
        median_step_diff_q: median |dq| across episodes, timesteps, and q dims (rad)
        max_step_diff_qd: max |dqd| across episodes, timesteps, and qd dims (rad/s)
    """
    if states.shape[1] < 2:
        zeros = np.zeros(states.shape[2], dtype=np.float64)
        return zeros, zeros, zeros
    deltas = np.abs(states[:, 1:, :, :] - states[:, :-1, :, :])
    max_step_diff_q = deltas[..., :2].max(axis=(0, 1, 3))
    median_step_diff_q = np.median(deltas[..., :2], axis=(0, 1, 3))
    max_step_diff_qd = deltas[..., 2:].max(axis=(0, 1, 3))
    return max_step_diff_q, median_step_diff_q, max_step_diff_qd


def save_search_report(
    gain_combos: list[tuple[float, float, float, float]],
    l1_error: np.ndarray,
    settling_time: np.ndarray,
    max_step_diff_q: np.ndarray,
    median_step_diff_q: np.ndarray,
    max_step_diff_qd: np.ndarray,
    sorted_idx: np.ndarray,
    top_k: int,
) -> Path:
    """Write a text report with linspace setup and top search results."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"pd_gains_search_report_{ts}.txt"

    lines: list[str] = []
    lines.append("PD Gains Search Report")
    lines.append(f"timestamp: {ts}")
    lines.append("")
    lines.append("PD gains linspace definition:")
    lines.append(
        f"  KE_VALUES = np.linspace({float(KE_VALUES[0]):.1f}, {float(KE_VALUES[-1]):.1f}, {len(KE_VALUES)})"
    )
    lines.append(
        f"  KD_VALUES = np.linspace({float(KD_VALUES[0]):.1f}, {float(KD_VALUES[-1]):.1f}, {len(KD_VALUES)})"
    )
    lines.append(f"  KE values: {np.array2string(KE_VALUES, precision=4)}")
    lines.append(f"  KD values: {np.array2string(KD_VALUES, precision=4)}")
    lines.append("")
    lines.append(
        f"Top {top_k} gain tuples (ke0, kd0, ke1, kd1) by mean L1 tracking error [rad]:"
    )

    for rank, idx in enumerate(sorted_idx[:top_k], start=1):
        gains = gain_combos[int(idx)]
        lines.append(
            f"  {rank:2d}. idx={int(idx):4d} l1_error={float(l1_error[idx]):.4f} "
            f"settle={float(settling_time[idx]):.4f}s "
            f"max_step_diff_q={float(max_step_diff_q[idx]):.4f}rad "
            f"median_step_diff_q={float(median_step_diff_q[idx]):.4f}rad "
            f"max_step_diff_qd={float(max_step_diff_qd[idx]):.4f}rad/s "
            f"ke0={gains[0]:.1f} kd0={gains[1]:.1f} "
            f"ke1={gains[2]:.1f} kd1={gains[3]:.1f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


class PDGainsSearchSim(BaseSimulator):
    def __init__(
        self,
        gain_combos: list[tuple[float, float, float, float]],
        *,
        device: str = "cuda:0",
    ):
        self.gain_combos = gain_combos
        self.num_worlds = len(gain_combos)
        self._device = device

        sim_config = SimulationConfig(
            duration_seconds=NUM_STEPS * DT,
            target_timestep_seconds=DT,
        )
        render_cfg = RenderingConfig(vis_type="null")
        exec_cfg = ExecutionConfig(use_cuda_graph=False)
        log_cfg = LoggingConfig()
        engine_cfg = AxionEngineConfig()

        super().__init__(
            sim_config,
            render_cfg,
            exec_cfg,
            engine_cfg,
            log_cfg,
        )

        self._q_target_flat = wp.empty(
            2 * self.num_worlds, dtype=wp.float32, device=self.model.device
        )

    @override
    def build_model(self) -> newton.Model:
        model = build_pendulum_model(
            num_worlds=len(self.gain_combos),
            device=self._device,
            with_contacts=False,
            joint_dof_mode=JointMode.TARGET_POSITION,
        )

        ke = model.joint_target_ke.numpy().copy()
        kd = model.joint_target_kd.numpy().copy()
        for w, (ke0, kd0, ke1, kd1) in enumerate(self.gain_combos):
            ke[w * 2], ke[w * 2 + 1] = ke0, ke1
            kd[w * 2], kd[w * 2 + 1] = kd0, kd1

        model.joint_target_ke.assign(
            wp.array(ke, dtype=wp.float32, device=model.device)
        )
        model.joint_target_kd.assign(
            wp.array(kd, dtype=wp.float32, device=model.device)
        )
        return model

    @override
    def control_policy(self, current_state: newton.State) -> None:
        wp.copy(dest=self.control.joint_target_pos, src=self._q_target_flat)

    def _reset_episode(
        self,
        ic: tuple[float, float, float, float],
        joint_target: tuple[float, float],
    ) -> None:
        t_host = np.tile(
            [float(joint_target[0]), float(joint_target[1])],
            self.num_worlds,
        ).astype(np.float32)
        self._q_target_flat.assign(
            wp.array(t_host, dtype=wp.float32, device=self.model.device)
        )

        joint_q = np.tile([ic[0], ic[1]], self.num_worlds).astype(np.float32)
        joint_qd = np.tile([ic[2], ic[3]], self.num_worlds).astype(np.float32)
        self.current_state.joint_q.assign(
            wp.array(joint_q, dtype=wp.float32, device=self.model.device)
        )
        self.current_state.joint_qd.assign(
            wp.array(joint_qd, dtype=wp.float32, device=self.model.device)
        )
        newton.eval_fk(
            self.model,
            self.current_state.joint_q,
            self.current_state.joint_qd,
            self.current_state,
        )
        self._copy_state(self.next_state, self.current_state)

        self.clock._current_step = 0
        self.clock._current_time = 0.0

    def run_episodes(
        self,
        initial_conditions: np.ndarray,
        episode_joint_targets: np.ndarray,
    ) -> np.ndarray:
        """Record [q0,q1,qd0,qd1] trajectories; shape (n_eps, n_steps, n_worlds, 4)."""
        n_eps = initial_conditions.shape[0]
        states = np.zeros(
            (n_eps, NUM_STEPS, self.num_worlds, 4), dtype=np.float32
        )

        for ep in tqdm(range(n_eps), desc="Episodes", unit="episode"):
            ic = tuple(float(x) for x in initial_conditions[ep])
            jt = tuple(float(x) for x in episode_joint_targets[ep])
            self._reset_episode(ic, jt)
            for step in tqdm(
                range(NUM_STEPS),
                desc=f"Episode {ep + 1}/{n_eps}",
                unit="step",
                leave=False,
            ):
                self._single_physics_step(step)
                wp.synchronize()
                # Convert stepped maximal state back to generalized coordinates so
                # recorded joint trajectories reflect actual simulated motion.
                newton.eval_ik(
                    self.model,
                    self.current_state,
                    self.current_state.joint_q,
                    self.current_state.joint_qd,
                )
                joint_q_flat = self.current_state.joint_q.numpy()
                joint_qd_flat = self.current_state.joint_qd.numpy()
                states[ep, step, :, :2] = joint_q_flat.reshape(self.num_worlds, 2)
                states[ep, step, :, 2:] = joint_qd_flat.reshape(self.num_worlds, 2)

        return states


def main() -> None:
    gain_combos = make_gain_combinations()
    expected_num_worlds = (len(KE_VALUES) * len(KD_VALUES)) ** 2
    if len(gain_combos) != expected_num_worlds:
        raise RuntimeError(
            "Unexpected gain grid size: "
            f"{len(gain_combos)} != {expected_num_worlds}"
        )
    print(
        f"Gain grid: {len(KE_VALUES)} KE values x {len(KD_VALUES)} KD values "
        f"-> {len(gain_combos)} worlds"
    )

    rng = np.random.default_rng(SEED)
    ic_low = np.array([-np.pi, -np.pi, -2.0 * np.pi, -2.0 * np.pi], dtype=np.float64)
    ic_high = np.array([np.pi, np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=np.float64)
    initial_conditions = rng.uniform(low=ic_low, high=ic_high, size=(NUM_EPISODES, 4))
    episode_targets = rng.uniform(
        low=JOINT_TARGET_MIN, high=JOINT_TARGET_MAX, size=(NUM_EPISODES, 2)
    )

    sim = PDGainsSearchSim(gain_combos)
    states = sim.run_episodes(initial_conditions, episode_targets)
    l1_error = compute_l1_tracking_error(states, episode_targets)
    settling_time = compute_settling_times(
        states,
        episode_targets,
        dt=DT,
        threshold=SETTLE_THRESH,
    )
    max_step_diff_q, median_step_diff_q, max_step_diff_qd = compute_state_step_difference_stats(states)

    sorted_idx = np.argsort(l1_error)
    top_k = min(10, len(gain_combos))
    print(
        f"Top {top_k} gain tuples (ke0, kd0, ke1, kd1) "
        "by mean L1 tracking error [rad]:"
    )
    for rank, idx in enumerate(sorted_idx[:top_k], start=1):
        gains = gain_combos[int(idx)]
        print(
            f"  {rank:2d}. idx={int(idx):4d} l1_error={float(l1_error[idx]):.4f} "
            f"settle={float(settling_time[idx]):.4f}s "
            f"max_step_diff_q={float(max_step_diff_q[idx]):.4f}rad "
            f"median_step_diff_q={float(median_step_diff_q[idx]):.4f}rad "
            f"max_step_diff_qd={float(max_step_diff_qd[idx]):.4f}rad/s "
            f"ke0={gains[0]:.1f} kd0={gains[1]:.1f} "
            f"ke1={gains[2]:.1f} kd1={gains[3]:.1f}"
        )

    report_path = save_search_report(
        gain_combos=gain_combos,
        l1_error=l1_error,
        settling_time=settling_time,
        max_step_diff_q=max_step_diff_q,
        median_step_diff_q=median_step_diff_q,
        max_step_diff_qd=max_step_diff_qd,
        sorted_idx=sorted_idx,
        top_k=top_k,
    )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
