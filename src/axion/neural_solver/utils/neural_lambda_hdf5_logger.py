from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np


class NeuralLambdaHDF5Logger:
    """
    One-off logger for neural-lambda engine experiments.

    The output schema is intentionally close to neural_solver/generate datasets:
    - top-level group: data/
    - per-step stacked arrays (T, B, ...)
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self._records: Dict[str, list[np.ndarray]] = {}
        self._saved = False

    def append_step(
        self,
        *,
        states: np.ndarray,
        next_states: np.ndarray,
        contact_normals: np.ndarray,
        contact_depths: np.ndarray,
        contact_points_0: np.ndarray,
        contact_points_1: np.ndarray,
        contact_thicknesses: np.ndarray,
        lambdas: Optional[np.ndarray],
        next_lambdas: np.ndarray,
        gravity_dir: np.ndarray,
        root_body_q: np.ndarray,
        predicted_next_lambdas: Optional[np.ndarray] = None,
        lambda_activity: Optional[np.ndarray] = None,
    ) -> None:
        step_payload = {
            "states": states,
            "next_states": next_states,
            "contact_normals": contact_normals,
            "contact_depths": contact_depths,
            "contact_points_0": contact_points_0,
            "contact_points_1": contact_points_1,
            "contact_thicknesses": contact_thicknesses,
            "next_lambdas": next_lambdas,
            "gravity_dir": gravity_dir,
            "root_body_q": root_body_q,
        }
        if lambdas is not None:
            step_payload["lambdas"] = lambdas
        if predicted_next_lambdas is not None:
            step_payload["predicted_next_lambdas"] = predicted_next_lambdas
        if lambda_activity is not None:
            step_payload["lambda_activity"] = lambda_activity

        for key, value in step_payload.items():
            arr = np.asarray(value)
            self._records.setdefault(key, []).append(arr)

    def _stack_records(self) -> Dict[str, np.ndarray]:
        stacked: Dict[str, np.ndarray] = {}
        for key, values in self._records.items():
            if len(values) == 0:
                continue
            stacked[key] = np.stack(values, axis=0)
        return stacked

    def save(self) -> None:
        if self._saved:
            return

        datasets = self._stack_records()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.output_path, "w") as f:
            data_grp = f.create_group("data")
            data_grp.attrs["mode"] = "trajectory"

            for key, arr in datasets.items():
                data_grp.create_dataset(
                    name=key,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                )

            if "states" in datasets:
                data_grp.attrs["state_dim"] = int(datasets["states"].shape[-1])
            if "next_states" in datasets:
                data_grp.attrs["next_state_dim"] = int(datasets["next_states"].shape[-1])
            if "contact_depths" in datasets:
                data_grp.attrs["num_contacts_per_env"] = int(datasets["contact_depths"].shape[-1])
            if "states" in datasets:
                total_trajectories = int(datasets["states"].shape[1])
                data_grp.attrs["total_trajectories"] = total_trajectories
                data_grp.attrs["total_transitions"] = int(
                    datasets["states"].shape[0] * total_trajectories
                )

        self._saved = True

