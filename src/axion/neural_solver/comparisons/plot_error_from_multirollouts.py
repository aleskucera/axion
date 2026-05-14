from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_LOG_DIR = Path(__file__).resolve().parents[4] / "data/logs/multirollouts"

# Edit when running via F5: same stem suffix as log_multiple_rollouts_autoregressive._hdf5_filename_stem
# (only the engine prefix differs between the files: Axion_, GPT_, TeacherForcedGPT_, AxionNeuralLambdas_).

#_SHARED_STEM = ("100roll_seed0_200steps_dt0p01_qm3p14159_3p14159_qdm5_5_pl0_0_1_0")
_SHARED_STEM = ("50roll_seed0_100steps_dt0p01_qm3p14159_3p14159_qdm3_3_pl0_0_1_0")

AXION_H5 = _LOG_DIR / f"Axion_{_SHARED_STEM}.h5"
GPT_H5 = _LOG_DIR / f"GPT_{_SHARED_STEM}.h5"
TEACHER_FORCED_GPT_H5 = _LOG_DIR / f"TeacherForcedGPT_{_SHARED_STEM}.h5"
# e.g. data/logs/AxionNeuralLambdas_100roll_seed0_200steps_dt0p01_qm3p14159_3p14159_qdm5_5_pl0_0_1_0.h5
INCLUDE_AXION_NEURAL_LAMBDAS = False
AXION_NEURAL_LAMBDAS_H5 = _LOG_DIR / f"AxionNeuralLambdas_{_SHARED_STEM}.h5"

SIM_DT: float | None = None  # None → x is step index; positive → x is time in seconds

# Match plot_hdf5log_from_example.py lines 80–86 (plotting typography / line / grid).
BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LINEWIDTH = 2.5  # Used for every ax.plot linewidth in this script
GRID_ALPHA = 0.3

_ROLLOUT_NAME_RE = re.compile(r"^rollout_(\d+)$")


def _apply_plot_style() -> None:
    """Same rcParams mapping as plot_hdf5log_from_example._apply_academic_matplotlib_style."""
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


def _load_multi_rollout_states(hdf5_path: Path, *, expected_engine: str) -> np.ndarray:
    """Load MultiRolloutStateLogger HDF5 into shape (R, T, 4)."""
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

        states_list: list[np.ndarray] = []
        t_lens: list[int] = []
        for name in rollout_names:
            rgrp = h5f["rollouts"][name]
            if "states" not in rgrp:
                raise KeyError(f"Missing dataset rollouts/{name}/states in {hdf5_path}")
            arr = np.asarray(rgrp["states"][:], dtype=np.float64).squeeze()
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(
                    f"rollouts/{name}/states must be (T, 4); got {arr.shape} in {hdf5_path}"
                )
            states_list.append(arr)
            t_lens.append(arr.shape[0])

        if len(set(t_lens)) != 1:
            raise ValueError(
                f"Inconsistent trajectory lengths across rollouts in {hdf5_path}: "
                f"unique lengths {sorted(set(t_lens))!r}"
            )

    return np.stack(states_list, axis=0)


def _wrap_to_pi_inplace_np(angles: np.ndarray) -> np.ndarray:
    """In-place wrap of angles to ``[-pi, pi)``.

    Same steps as ``_wrap_to_pi_`` in ``transformer_neural_utils_provider_new``
    (add ``pi``, ``remainder`` by ``2*pi``, subtract ``pi``).
    """
    two_pi = 2.0 * np.pi
    np.add(angles, np.pi, out=angles)
    np.remainder(angles, two_pi, out=angles)
    np.subtract(angles, np.pi, out=angles)
    return angles


def _wrap_axion_joint_positions_to_pi_inplace(axion_rt4: np.ndarray) -> None:
    """Wrap minimal-coordinate joint angles ``q0, q1`` (columns 0–1) to ``[-pi, pi)`` in place."""
    if axion_rt4.ndim != 3 or axion_rt4.shape[2] < 2:
        raise ValueError(f"Expected (R, T, 4+) states; got {axion_rt4.shape}")
    _wrap_to_pi_inplace_np(axion_rt4[..., :2])


def _mean_l1_separation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-time average over rollouts of the L1 state separation.

    For each time index ``t``, returns ``(1/R) * sum_r ||a[r,t,:] - b[r,t,:]||_1``
    where ``a,b`` have shape ``(R, T, 4)``.  Output shape is ``(T,)``.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"Expected (R, T, 4); got {a.shape}")
    diff = a - b
    l1_per_rollout_and_time = np.sum(np.abs(diff), axis=-1)  # (R, T)
    return np.mean(l1_per_rollout_and_time, axis=0)  # (T,) — mean over rollouts r at each t


def _build_x_axis(t_len: int) -> tuple[np.ndarray, str]:
    if SIM_DT is None:
        return np.arange(t_len, dtype=float), r"Time step $t$ [-]"
    if SIM_DT <= 0:
        raise ValueError(f"SIM_DT must be positive when set, got {SIM_DT}")
    return np.arange(t_len, dtype=float) * SIM_DT, "Time [s]"


def main() -> None:
    _apply_plot_style()
    axion = _load_multi_rollout_states(AXION_H5, expected_engine="axion")
    _wrap_axion_joint_positions_to_pi_inplace(axion)
    gpt = _load_multi_rollout_states(GPT_H5, expected_engine="gpt")
    teacher_forced = _load_multi_rollout_states(
        TEACHER_FORCED_GPT_H5, expected_engine="teacher_forced_gpt"
    )
    axion_neural_lambdas: np.ndarray | None = None
    if INCLUDE_AXION_NEURAL_LAMBDAS:
        axion_neural_lambdas = _load_multi_rollout_states(
            AXION_NEURAL_LAMBDAS_H5, expected_engine="axion_neural_lambdas"
        )

    to_validate: list[tuple[str, np.ndarray]] = [
        ("GPT", gpt),
        ("TeacherForcedGPT", teacher_forced),
    ]
    if axion_neural_lambdas is not None:
        to_validate.append(("AxionNeuralLambdas", axion_neural_lambdas))

    for name, arr in to_validate:
        if axion.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Rollout count mismatch: Axion R={axion.shape[0]}, {name} R={arr.shape[0]}"
            )
        if axion.shape[1] != arr.shape[1]:
            raise ValueError(
                f"Time length mismatch: Axion T={axion.shape[1]}, {name} T={arr.shape[1]}"
            )

    l1_axion_vs_gpt = _mean_l1_separation(axion, gpt)
    l1_teacher_minus_axion = _mean_l1_separation(teacher_forced, axion)
    t_len = int(l1_axion_vs_gpt.shape[0])
    x_axis, x_label = _build_x_axis(t_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_axis,
        l1_axion_vs_gpt,
        linewidth=LINEWIDTH,
        color="black",
        label = (
            "Neural prediction (autoregressive)"
        )
        # label=(
        #     r"$\frac{1}{R}\sum_r \|\mathbf{s}^{\mathrm{Axion}}_{r,t}"
        #     r"-\mathbf{s}^{\mathrm{GPT}}_{r,t}\|_1$"
        # ),
    )
    ax.plot(
        x_axis,
        l1_teacher_minus_axion,
        linewidth=LINEWIDTH,
        color="red",
        label = (
            "Neural prediction (teacher-forced)"
        )
        # label=(
        #     r"$\frac{1}{R}\sum_r \|\mathbf{s}^{\mathrm{TeacherForcedGPT}}_{r,t}"
        #     r"-\mathbf{s}^{\mathrm{Axion}}_{r,t}\|_1$"
        # ),
    )
    if axion_neural_lambdas is not None:
        l1_axion_neural_lambdas_minus_axion = _mean_l1_separation(
            axion_neural_lambdas, axion
        )
        ax.plot(
            x_axis,
            l1_axion_neural_lambdas_minus_axion,
            linewidth=LINEWIDTH,
            color="tab:blue",
            label="NN state (Axion+neural lambdas log) vs Axion",
        )
    ax.set_xlabel(x_label)
    n_rollouts = int(axion.shape[0])
    ax.set_title(
        rf"Mean L1 state error: $\|\hat{{\mathbf{{s}}}}_{{t+1}} - \mathbf{{s}}_{{t+1}}\|_1$ "
        rf"per {n_rollouts} trajectories"
    )
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
