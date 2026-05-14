from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import newton
import numpy as np

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "data" / "logs" / "multirollouts"

STATE_DIM = 4


class PendulumStateLogger:
    """Lightweight HDF5 logger for double-pendulum minimal coordinates.

    Records ``[q0, q1, qd0, qd1]`` once per render segment and writes a
    timestamped HDF5 file to ``data/logs/`` on :meth:`save`.

    HDF5 schema::

        data/
            states  (T, 4)  float32  [q0, q1, qd0, qd1] per logged step
            time    (T,)    float32  simulation time [s]
        attrs:
            script_name       str
            dt                float   time between consecutive logged samples [s]
            duration_seconds  float
            state_dim         int     always 4
    """

    STATE_DIM = STATE_DIM

    def __init__(
        self,
        script_name: str,
        dt: float,
        duration_seconds: float,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.script_name = script_name
        self.dt = dt
        self.duration_seconds = duration_seconds

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(output_dir) if output_dir is not None else _DEFAULT_LOG_DIR
        self.output_path: Path = log_dir / f"{script_name}_{timestamp}.h5"

        self._states: list[np.ndarray] = []
        self._times: list[float] = []
        self._saved = False

    def log_step(self, state: newton.State, sim_time: float) -> None:
        """Append one minimal-coordinate sample to the in-memory buffer.

        Duplicate frames (same ``sim_time`` as the previous entry) are silently
        dropped, which prevents double-logging when a GL viewer is paused.
        """
        if self._times and abs(self._times[-1] - sim_time) < 1e-9:
            return

        q = state.joint_q.numpy().ravel()[:2].astype(np.float32)
        qd = state.joint_qd.numpy().ravel()[:2].astype(np.float32)
        self._states.append(np.concatenate([q, qd]))
        self._times.append(float(sim_time))

    def save(self) -> None:
        """Flush the buffer to disk as a compressed HDF5 file."""
        if self._saved:
            return
        if not self._states:
            print("[PendulumStateLogger] Nothing to save — buffer is empty.")
            return

        states = np.stack(self._states, axis=0)       # (T, 4)
        times = np.array(self._times, dtype=np.float32)  # (T,)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_path, "w") as f:
            grp = f.create_group("data")
            grp.attrs["script_name"] = self.script_name
            grp.attrs["dt"] = float(self.dt)
            grp.attrs["duration_seconds"] = float(self.duration_seconds)
            grp.attrs["state_dim"] = self.STATE_DIM
            grp.create_dataset(
                "states", data=states, compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "time", data=times, compression="gzip", compression_opts=4
            )

        self._saved = True
        print(
            f"[PendulumStateLogger] Saved {len(self._states)} steps → {self.output_path}"
        )


class MultiRolloutStateLogger:
    """HDF5 logger that collects state trajectories from N independent rollouts.

    Each rollout is started with :meth:`start_rollout`, logged step-by-step
    with :meth:`log_step` (same signature as :class:`PendulumStateLogger`),
    and committed with :meth:`finish_rollout`.  All trajectories are written
    to a single compressed HDF5 file on :meth:`save`.

    The output file is ``{filename_stem}.h5`` under ``data/logs/`` (or
    ``output_dir``) when ``filename_stem`` is set; otherwise a timestamped
    ``multi_rollout_{engine}_{timestamp}.h5`` name is used.

    HDF5 schema::

        attrs:
            engine            str   "axion" | "gpt"
            n_rollouts        int
            seed              int
            dt                float  time between consecutive logged samples [s]
            duration_seconds  float
            state_dim         int    always 4
        rollouts/
            rollout_000/
                states         (T, 4)  float32  [q0, q1, qd0, qd1]
                time           (T,)    float32  simulation time [s]
                initial_state  (4,)    float32  [q0, q1, qd0, qd1] at t=0
            rollout_001/ ...
    """

    STATE_DIM = STATE_DIM

    def __init__(
        self,
        engine: str,
        n_rollouts: int,
        seed: int,
        dt: float,
        duration_seconds: float,
        output_dir: Optional[Path] = None,
        filename_stem: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.n_rollouts = n_rollouts
        self.seed = seed
        self.dt = dt
        self.duration_seconds = duration_seconds

        log_dir = Path(output_dir) if output_dir is not None else _DEFAULT_LOG_DIR
        if filename_stem is not None:
            safe_stem = filename_stem.replace("/", "_").replace("\\", "_")
            self.output_path: Path = log_dir / f"{safe_stem}.h5"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = log_dir / f"multi_rollout_{engine}_{timestamp}.h5"

        # Completed rollouts: list of (initial_state, states_array, times_array)
        self._rollouts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # Buffers for the rollout currently in progress
        self._current_ic: Optional[np.ndarray] = None
        self._current_states: list[np.ndarray] = []
        self._current_times: list[float] = []

        self._saved = False

    def start_rollout(self, initial_state: tuple[float, float, float, float]) -> None:
        """Begin a new rollout.  Clears the per-rollout buffers and records the IC."""
        self._current_ic = np.array(initial_state, dtype=np.float32)
        self._current_states = []
        self._current_times = []

    def log_step(self, state: newton.State, sim_time: float) -> None:
        """Append one minimal-coordinate sample to the current rollout buffer.

        Duplicate frames (same ``sim_time`` as the previous entry) are silently
        dropped, which prevents double-logging when a viewer is paused.
        """
        if self._current_times and abs(self._current_times[-1] - sim_time) < 1e-9:
            return

        q = state.joint_q.numpy().ravel()[:2].astype(np.float32)
        qd = state.joint_qd.numpy().ravel()[:2].astype(np.float32)
        self._current_states.append(np.concatenate([q, qd]))
        self._current_times.append(float(sim_time))

    def log_step_from_array(self, state_array: np.ndarray, sim_time: float) -> None:
        """Append a pre-extracted (STATE_DIM,) float32 state array instead of a newton.State.

        Useful when the state to log is not yet materialised in a newton.State —
        e.g. the NN-predicted next state from AxionEngineWithNeuralLambdas.
        Duplicate frames (same sim_time as the previous entry) are silently dropped.
        """
        if self._current_times and abs(self._current_times[-1] - sim_time) < 1e-9:
            return
        self._current_states.append(state_array.astype(np.float32).ravel()[:self.STATE_DIM])
        self._current_times.append(float(sim_time))

    def finish_rollout(self) -> None:
        """Commit the current rollout buffer to the completed rollouts list."""
        if not self._current_states:
            print(f"[MultiRolloutStateLogger] Rollout {len(self._rollouts)} is empty — skipping.")
            return
        states = np.stack(self._current_states, axis=0)        # (T, 4)
        times = np.array(self._current_times, dtype=np.float32)  # (T,)
        ic = self._current_ic if self._current_ic is not None else np.zeros(self.STATE_DIM, dtype=np.float32)
        self._rollouts.append((ic, states, times))

    def save(self) -> None:
        """Write all completed rollouts to a single compressed HDF5 file."""
        if self._saved:
            return
        if not self._rollouts:
            print("[MultiRolloutStateLogger] Nothing to save — no rollouts completed.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_path, "w") as f:
            f.attrs["engine"] = self.engine
            f.attrs["n_rollouts"] = len(self._rollouts)
            f.attrs["seed"] = self.seed
            f.attrs["dt"] = float(self.dt)
            f.attrs["duration_seconds"] = float(self.duration_seconds)
            f.attrs["state_dim"] = self.STATE_DIM

            grp = f.create_group("rollouts")
            for idx, (ic, states, times) in enumerate(self._rollouts):
                rgrp = grp.create_group(f"rollout_{idx:03d}")
                rgrp.create_dataset("states", data=states, compression="gzip", compression_opts=4)
                rgrp.create_dataset("time", data=times, compression="gzip", compression_opts=4)
                rgrp.create_dataset("initial_state", data=ic)

        self._saved = True
        print(
            f"[MultiRolloutStateLogger] Saved {len(self._rollouts)} rollouts "
            f"({len(self._rollouts[0][1])} steps each) → {self.output_path}"
        )
