from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import newton
import numpy as np

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "data" / "logs"


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

    STATE_DIM = 4

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
