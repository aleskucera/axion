from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_LOG_DIR = Path(__file__).resolve().parents[4] / "data/logs/multirollouts"
TRAJ_ID = 8

# Edit when running via F5: same stem suffix as log_multiple_rollouts_autoregressive._hdf5_filename_stem
# (only the engine prefix differs between the files: Axion_, GPT_, TeacherForcedGPT_, AxionNeuralLambdas_).

# _SHARED_STEM = ("100roll_seed0_200steps_dt0p01_qm3p14159_3p14159_qdm5_5_pl0_0_1_0")
_SHARED_STEM = ("10roll_seed0_200steps_dt0p01_qm3p14159_3p14159_qdm1_1_pl0_0_1_0")

AXION_H5 = _LOG_DIR / f"Axion_{_SHARED_STEM}.h5"
GPT_H5 = _LOG_DIR / f"GPT_{_SHARED_STEM}.h5"
TEACHER_FORCED_GPT_H5 = _LOG_DIR / f"TeacherForcedGPT_{_SHARED_STEM}.h5"
AXION_NEURAL_LAMBDAS_H5 = _LOG_DIR / f"AxionNeuralLambdas_{_SHARED_STEM}.h5"

SIM_DT: float | None = None  # None → x is step index; positive → x is time in seconds

BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LINEWIDTH = 2.5
GRID_ALPHA = 0.3

STATE_DIM = 4
_STATE_Y_LABELS = (r"$q_0$", r"$q_1$", r"$\dot{q}_0$", r"$\dot{q}_1$")

_ROLLOUT_NAME_RE = re.compile(r"^rollout_(\d+)$")


def _apply_plot_style() -> None:
    """Same rcParams mapping as plot_error_from_multirollouts._apply_plot_style."""
    plt.rcParams.update(
        {
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": AXES_LABELS_FONTSIZE,
            "axes.titlesize": AXES_LABELS_FONTSIZE + 1,
            "xtick.labelsize": AXES_TICKS_FONTSIZE,
            "ytick.labelsize": AXES_TICKS_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
        }
    )


def _sorted_rollout_group_names(h5f: h5py.File) -> list[str]:
    names: list[tuple[int, str]] = []
    for key in h5f["rollouts"].keys():
        m = _ROLLOUT_NAME_RE.match(str(key))
        if m is not None:
            names.append((int(m.group(1)), str(key)))
    names.sort(key=lambda x: x[0])
    return [n for _, n in names]


def _load_rollout_states(
    hdf5_path: Path, *, traj_id: int, expected_engine: str
) -> np.ndarray:
    """Load one rollout's states from MultiRolloutStateLogger HDF5 into shape (T, 4).

    ``traj_id`` selects ``rollout_names[traj_id]`` where ``rollout_names`` are sorted by
    numeric suffix (see :func:`_sorted_rollout_group_names`).
    """
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    expected_engine = expected_engine.lower()
    with h5py.File(hdf5_path, "r") as h5f:
        eng = str(h5f.attrs.get("engine", "")).lower()
        if eng and eng != expected_engine:
            raise ValueError(
                f"{hdf5_path} has attrs['engine']={h5f.attrs.get('engine')!r}, "
                f"expected {expected_engine!r}"
            )
        if "rollouts" not in h5f:
            raise KeyError(f"Missing top-level group 'rollouts' in {hdf5_path}")

        rollout_names = _sorted_rollout_group_names(h5f)
        if not rollout_names:
            raise KeyError(f"No rollout_* groups under rollouts in {hdf5_path}")

        n_roll = len(rollout_names)
        if traj_id < 0 or traj_id >= n_roll:
            raise IndexError(
                f"TRAJ_ID={traj_id} out of range for {hdf5_path}: "
                f"have {n_roll} rollout(s), indices 0..{n_roll - 1}"
            )

        rollout_name = rollout_names[traj_id]
        rgrp = h5f["rollouts"][rollout_name]
        if "states" not in rgrp:
            raise KeyError(
                f"Missing dataset rollouts/{rollout_name}/states in {hdf5_path}"
            )
        arr = np.asarray(rgrp["states"][:], dtype=np.float64).squeeze()
        if arr.ndim != 2 or arr.shape[1] != STATE_DIM:
            raise ValueError(
                f"rollouts/{rollout_name}/states must be (T, {STATE_DIM}); "
                f"got {arr.shape} in {hdf5_path}"
            )

    return arr


def _build_x_axis(t_len: int) -> tuple[np.ndarray, str]:
    if SIM_DT is None:
        return np.arange(t_len, dtype=float), r"Time step $t$ [-]"
    if SIM_DT <= 0:
        raise ValueError(f"SIM_DT must be positive when set, got {SIM_DT}")
    return np.arange(t_len, dtype=float) * SIM_DT, "Time [s]"


def main() -> None:
    _apply_plot_style()
    tid = TRAJ_ID
    axion = _load_rollout_states(AXION_H5, traj_id=tid, expected_engine="axion")
    gpt = _load_rollout_states(GPT_H5, traj_id=tid, expected_engine="gpt")
    teacher_forced = _load_rollout_states(
        TEACHER_FORCED_GPT_H5, traj_id=tid, expected_engine="teacher_forced_gpt"
    )
    axion_neural_lambdas = _load_rollout_states(
        AXION_NEURAL_LAMBDAS_H5, traj_id=tid, expected_engine="axion_neural_lambdas"
    )
    for name, arr in (
        ("GPT", gpt),
        ("TeacherForcedGPT", teacher_forced),
        ("AxionNeuralLambdas", axion_neural_lambdas),
    ):
        if axion.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch: Axion T={axion.shape[0]}, {name} T={arr.shape[0]}"
            )

    t_len = int(axion.shape[0])
    x_axis, x_label = _build_x_axis(t_len)

    engines: tuple[tuple[np.ndarray, str, str], ...] = (
        (axion, "Axion", "tab:green"),
        (gpt, "Neural prediction (autoregressive)", "black"),
        (teacher_forced, "Neural prediction (teacher-forced)", "red"),
        (
            axion_neural_lambdas,
            "NN state (Axion+neural lambdas)",
            "tab:blue",
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()

    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for k in range(STATE_DIM):
        ax_k = axes_flat[k]
        for states_arr, label, color in engines:
            (line,) = ax_k.plot(
                x_axis,
                states_arr[:, k],
                linewidth=LINEWIDTH,
                color=color,
                label=label if k == 0 else "_nolegend_",
            )
            if k == 0:
                handles.append(line)
                labels.append(label)
        ax_k.set_ylabel(_STATE_Y_LABELS[k])
        ax_k.grid(True, alpha=GRID_ALPHA)
        if k >= 2:
            ax_k.set_xlabel(x_label)

    fig.suptitle(
        f"Minimal coordinates (TRAJ_ID={tid}): Axion vs neural engines",
        y=1.02,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    plt.show()


if __name__ == "__main__":
    main()
