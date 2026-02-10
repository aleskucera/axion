"""
Compare double-pendulum simulation data from two sources:

1. Neural-solver dataset  (generalized coordinates)
   Path: src/axion/neural_solver/datasets/PendulumWithContact/pendulum1.hdf5
   Layout: data/states  (N, 1, 4)  →  [q0, q1, qd0, qd1]
           data/next_states (N, 1, 4)

2. AxionEngine log  (maximal coordinates)
   Path: data/logs/pendulum_AxionEngine.h5
   Layout: timestep_XXXX / newton_iteration_YY / dynamics / body_q  (1, 2, 7)
           timestep_XXXX / newton_iteration_YY / dynamics / body_u  (1, 2, 6)

The AxionEngine log stores body transforms (body_q = [x,y,z,qx,qy,qz,qw])
and spatial velocities (body_u = [vx,vy,vz,wx,wy,wz]) in maximal coordinates.
We convert them to generalized joint coordinates [q0, q1, qd0, qd1] via
newton.eval_ik (inverse kinematics) so both datasets live in the same space.

For each AxionEngine timestep we pick the *last* Newton iteration
(= converged solution).
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import newton

from axion.core.model_builder import AxionModelBuilder
from axion import JointMode


# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # axion/

DATASET_PATH = (
    _REPO_ROOT
    / "src"
    / "axion"
    / "neural_solver"
    / "datasets"
    / "PendulumWithContact"
    / "pendulum1.hdf5"
)
ENGINE_LOG_PATH = _REPO_ROOT / "data" / "logs" / "pendulum_AxionEngine.h5"

# Physics constants matching AxionEnv / pendulum_AxionEngine.py
PENDULUM_HEIGHT = 5.0
FRAME_DT = 0.1  # seconds per logged step (both sources)

# ---------------------------------------------------------------------------
# 1. Build the same pendulum model used by AxionEnv
# ---------------------------------------------------------------------------

def build_pendulum_model(device: str = "cpu") -> newton.Model:
    """Construct the 2-link revolute pendulum identical to AxionEnv."""
    builder = AxionModelBuilder()

    chain_width = 1.5
    hx = chain_width * 0.5
    link_config = newton.ModelBuilder.ShapeConfig(
        density=500.0, ke=1.0e4, kd=1.0e3, kf=1.0e4
    )
    capsule_xform = wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2),
    )

    link_0 = builder.add_link(armature=0.1)
    builder.add_shape_capsule(
        link_0, xform=capsule_xform, radius=0.1,
        half_height=hx, cfg=link_config,
    )

    link_1 = builder.add_link(armature=0.1)
    builder.add_shape_capsule(
        link_1, xform=capsule_xform, radius=0.1,
        half_height=hx, cfg=link_config,
    )

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(
            p=wp.vec3(0.0, 0.0, PENDULUM_HEIGHT), q=wp.quat_identity()
        ),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        custom_attributes={"joint_dof_mode": [JointMode.NONE]},
    )

    j1 = builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=500.0,
        target_kd=5.0,
        custom_attributes={"joint_dof_mode": [JointMode.NONE]},
        armature=0.1,
    )

    builder.add_articulation([j0, j1], key="pendulum")
    builder.add_ground_plane()

    return builder.finalize_replicated(
        num_worlds=1, gravity=-9.81, device=device,
    )


# ---------------------------------------------------------------------------
# 2. Read neural-solver dataset (already in generalized coordinates)
# ---------------------------------------------------------------------------

def load_dataset_generalized(path: pathlib.Path) -> np.ndarray:
    """Return shape (T, 4) array of [q0, q1, qd0, qd1] over time.

    The HDF5 stores transitions: states[i] → next_states[i].
    Since states[i+1] == next_states[i], the full trajectory is
    states[0..N-1] followed by next_states[N-1].
    """
    with h5py.File(str(path), "r") as f:
        states = f["data/states"][:, 0, :]        # (N, 4)
        next_states = f["data/next_states"][:, 0, :]  # (N, 4)

    # Append the final state that only appears in next_states
    trajectory = np.concatenate([states, next_states[-1:]], axis=0)  # (N+1, 4)
    return trajectory


# ---------------------------------------------------------------------------
# 3. Read AxionEngine log & convert via newton.eval_ik
# ---------------------------------------------------------------------------

def _last_newton_iter_key(group: h5py.Group) -> str:
    """Return the key of the highest-numbered newton_iteration_XX in *group*."""
    ni_keys = sorted(
        k for k in group.keys() if k.startswith("newton_iteration_")
    )
    return ni_keys[-1]


def _extract_body_data(group: h5py.Group):
    """From a timestep group, get body_q and body_u from the last Newton iteration.

    Returns
    -------
    body_q : np.ndarray, shape (num_bodies, 7)
    body_u : np.ndarray, shape (num_bodies, 6)
    """
    last_ni = _last_newton_iter_key(group)
    dynamics = group[last_ni]["dynamics"]
    body_q = dynamics["body_q"][0]   # squeeze batch dim → (num_bodies, 7)
    body_u = dynamics["body_u"][0]   # squeeze batch dim → (num_bodies, 6)
    return body_q, body_u


def load_engine_log_generalized(
    path: pathlib.Path,
    model: newton.Model,
    device: str = "cpu",
) -> np.ndarray:
    """Read maximal-coordinate log and convert to generalized coords via eval_ik.

    Returns shape (T, 4) array of [q0, q1, qd0, qd1] over time.
    """
    state = model.state()

    with h5py.File(str(path), "r") as f:
        # Collect all timestep groups in order.
        # Top-level newton_iteration_XX groups belong to timestep 0.
        # timestep_XXXX groups (1-indexed) are subsequent steps.
        timestep_keys = sorted(
            k for k in f.keys() if k.startswith("timestep_")
        )

        # --- timestep 0 (top-level newton iterations) ---
        all_body_q: list[np.ndarray] = []
        all_body_u: list[np.ndarray] = []

        ni_top = sorted(
            k for k in f.keys() if k.startswith("newton_iteration_")
        )
        if ni_top:
            last_ni = ni_top[-1]
            bq = f[last_ni]["dynamics"]["body_q"][0]
            bu = f[last_ni]["dynamics"]["body_u"][0]
            all_body_q.append(bq)
            all_body_u.append(bu)

        # --- timestep_0001 … timestep_NNNN ---
        for ts_key in timestep_keys:
            bq, bu = _extract_body_data(f[ts_key])
            all_body_q.append(bq)
            all_body_u.append(bu)

    # Convert every frame to generalized coords using newton.eval_ik
    num_steps = len(all_body_q)
    trajectory = np.zeros((num_steps, 4), dtype=np.float32)

    for i in range(num_steps):
        bq_np = all_body_q[i].astype(np.float32)   # (num_bodies, 7)
        bu_np = all_body_u[i].astype(np.float32)   # (num_bodies, 6)

        # Write maximal coords into the state
        state.body_q.assign(
            wp.array(bq_np, dtype=wp.transform, device=device)
        )
        state.body_qd.assign(
            wp.array(bu_np, dtype=wp.spatial_vector, device=device)
        )

        # Inverse kinematics → populates joint_q, joint_qd
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)

        jq = state.joint_q.numpy()    # (2,) → [q0, q1]
        jqd = state.joint_qd.numpy()  # (2,) → [qd0, qd1]
        trajectory[i] = np.concatenate([jq, jqd])

    return trajectory


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    traj_dataset: np.ndarray,
    traj_engine: np.ndarray,
    dt: float = FRAME_DT,
    save_path: pathlib.Path | None = None,
) -> None:
    """Plot q0, q1, qd0, qd1 from both sources on shared time axes."""

    t_ds = np.arange(len(traj_dataset)) * dt
    t_eng = np.arange(len(traj_engine)) * dt

    labels = [
        (r"$q_0$ (joint 0 angle)", "rad"),
        (r"$q_1$ (joint 1 angle)", "rad"),
        (r"$\dot{q}_0$ (joint 0 velocity)", "rad/s"),
        (r"$\dot{q}_1$ (joint 1 velocity)", "rad/s"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Double-Pendulum: Dataset vs AxionEngine Log", fontsize=14)

    for ax, col, (label, unit) in zip(axes, range(4), labels):
        ax.plot(t_ds, traj_dataset[:, col], "o", label="Dataset gen. script using Axion Engine", markersize=2.5)
        ax.plot(t_eng, traj_engine[:, col], "s", label="Example script using Axion Engine", markersize=2.5)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cpu"

    print(f"Dataset path:    {DATASET_PATH}")
    print(f"Engine log path: {ENGINE_LOG_PATH}")

    if not DATASET_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATASET_PATH}")
    if not ENGINE_LOG_PATH.exists():
        sys.exit(f"ERROR: engine log not found at {ENGINE_LOG_PATH}")

    # Build the pendulum model (needed for eval_ik)
    print("Building pendulum model …")
    model = build_pendulum_model(device=device)

    # Load dataset (already in generalized coordinates)
    print("Loading neural-solver dataset …")
    traj_dataset = load_dataset_generalized(DATASET_PATH)
    print(f"  → {len(traj_dataset)} time steps, state range: "
          f"q0=[{traj_dataset[:,0].min():.3f}, {traj_dataset[:,0].max():.3f}], "
          f"q1=[{traj_dataset[:,1].min():.3f}, {traj_dataset[:,1].max():.3f}]")

    # Load engine log and run IK
    print("Loading AxionEngine log & running inverse kinematics …")
    traj_engine = load_engine_log_generalized(ENGINE_LOG_PATH, model, device=device)
    print(f"  → {len(traj_engine)} time steps, state range: "
          f"q0=[{traj_engine[:,0].min():.3f}, {traj_engine[:,0].max():.3f}], "
          f"q1=[{traj_engine[:,1].min():.3f}, {traj_engine[:,1].max():.3f}]")

    # Plot comparison
    save_fig = _REPO_ROOT / "src" / "axion" / "neural_solver" / "comparison_plot.png"
    plot_comparison(traj_dataset, traj_engine, dt=FRAME_DT, save_path=save_fig)


if __name__ == "__main__":
    main()
