import pathlib
import time
from typing import Optional

import h5py
import numpy as np
import newton
import warp as wp

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

from pendulum_articulation_definition import build_pendulum_model, PENDULUM_HEIGHT
from pendulum_utils import set_tilted_plane_from_coefficients

HDF5_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src" / "axion" / "neural_solver" / "datasets" / "Pendulum"
    / "pendulumControlTest2.hdf5"
)

DT = 0.01
TRAJECTORY_INDEX = 2
PLAYBACK_SPEED = 1.0

def _safe_attr(attrs, key: str):
    return attrs[key] if key in attrs else None

def _print_dataset_summary(
    states: np.ndarray,
    plane_coeffs: Optional[np.ndarray],
    next_states: Optional[np.ndarray],
    joint_target_pos: Optional[np.ndarray],
    state_dim_attr,
    joint_target_dim_attr,
    traj_idx: int,
):
    print(f"Loaded trajectory {traj_idx}: {states.shape[0]} timesteps")
    print(
        f"  states shape={states.shape}, q0 range [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}], "
        f"q1 range [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]"
    )

    if state_dim_attr is not None:
        print(f"  metadata state_dim={int(state_dim_attr)}")

    if plane_coeffs is not None:
        print(
            f"  Plane coefficients (a,b,c,d): "
            f"[{plane_coeffs[0]:.4f}, {plane_coeffs[1]:.4f}, {plane_coeffs[2]:.4f}, {plane_coeffs[3]:.4f}]"
        )
    else:
        print("  Plane coefficients not found; using model default plane.")

    if next_states is not None:
        print(f"  next_states shape={next_states.shape}")
    else:
        print("  next_states not found.")

    if joint_target_pos is not None:
        mins = joint_target_pos.min(axis=0)
        maxs = joint_target_pos.max(axis=0)
        print(f"  joint_target_pos shape={joint_target_pos.shape}")
        for i, (mn, mx) in enumerate(zip(mins, maxs, strict=False)):
            print(f"    target q{i} range [{mn:.4f}, {mx:.4f}]")
        print(f"    first-step target = {joint_target_pos[0].tolist()}")
    else:
        print("  joint_target_pos not found.")

    if joint_target_dim_attr is not None:
        print(f"  metadata joint_target_dim={int(joint_target_dim_attr)}")

    q0, q1, qd0, qd1 = states[0, 0], states[0, 1], states[0, 2], states[0, 3]
    print(f"  Initial state (q0, q1, q0_dot, q1_dot): [{q0:.4f}, {q1:.4f}, {qd0:.4f}, {qd1:.4f}]")

def load_trajectory(hdf5_path: pathlib.Path, traj_idx: int):
    """Load one trajectory from HDF5 with backward-compatible optional fields.

    Returns:
        states: (num_steps, 4) with [q0, q1, q0_dot, q1_dot]
        plane_coeffs: (4,) or None
        next_states: (num_steps, 4) or None
        joint_target_pos: (num_steps, 2) or None
    """
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Expected HDF5 group 'data' in {hdf5_path}")
        data_grp = f["data"]

        if "states" not in data_grp:
            raise KeyError(f"Expected dataset 'data/states' in {hdf5_path}")
        num_trajectories = data_grp["states"].shape[1]
        if not (0 <= traj_idx < num_trajectories):
            raise IndexError(
                f"Trajectory index {traj_idx} out of range [0, {num_trajectories - 1}]"
            )
        states = np.array(data_grp["states"][:, traj_idx, :])

        state_dim_attr = _safe_attr(data_grp.attrs, "state_dim")
        joint_target_dim_attr = _safe_attr(data_grp.attrs, "joint_target_dim")

        if "plane_coefficients" in data_grp:
            # (num_steps, num_trajectories, 4) -> first step defines plane for trajectory
            plane_coeffs = np.array(data_grp["plane_coefficients"][0, traj_idx, :])
        else:
            plane_coeffs = None

        if "next_states" in data_grp:
            next_states = np.array(data_grp["next_states"][:, traj_idx, :])
        else:
            next_states = None

        if "joint_target_pos" in data_grp:
            joint_target_pos = np.array(data_grp["joint_target_pos"][:, traj_idx, :])
        else:
            joint_target_pos = None

    _print_dataset_summary(
        states=states,
        plane_coeffs=plane_coeffs,
        next_states=next_states,
        joint_target_pos=joint_target_pos,
        state_dim_attr=state_dim_attr,
        joint_target_dim_attr=joint_target_dim_attr,
        traj_idx=traj_idx,
    )
    return states, plane_coeffs, next_states, joint_target_pos


def set_state_from_generalized(
    model: newton.Model,
    state: newton.State,
    q0: float,
    q1: float,
    qd0: float = 0.0,
    qd1: float = 0.0,
):
    """Write generalized coordinates into state and run forward kinematics."""
    device = state.joint_q.device
    state.joint_q.assign(wp.array([q0, q1], dtype=wp.float32, device=device))
    state.joint_qd.assign(wp.array([qd0, qd1], dtype=wp.float32, device=device))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def draw_axes(viewer, device):
    """Draw world-origin and pendulum-anchor reference frames."""
    axis_len = 1.0
    origin = wp.vec3(0.0, 0.0, 0.0)

    starts = wp.array([origin, origin, origin], dtype=wp.vec3, device=device)
    ends = wp.array(
        [wp.vec3(axis_len, 0.0, 0.0),
         wp.vec3(0.0, axis_len, 0.0),
         wp.vec3(0.0, 0.0, axis_len)],
        dtype=wp.vec3, device=device,
    )
    colors = wp.array(
        [wp.vec3(1.0, 0.0, 0.0),
         wp.vec3(0.0, 1.0, 0.0),
         wp.vec3(0.0, 0.0, 1.0)],
        dtype=wp.vec3, device=device,
    )
    viewer.log_lines("world_axes", starts, ends, colors, width=0.08)

    anchor = wp.vec3(0.0, 0.0, PENDULUM_HEIGHT)
    a_len = 0.5
    a_starts = wp.array([anchor, anchor, anchor], dtype=wp.vec3, device=device)
    a_ends = wp.array(
        [wp.vec3(a_len, 0.0, PENDULUM_HEIGHT),
         wp.vec3(0.0, a_len, PENDULUM_HEIGHT),
         wp.vec3(0.0, 0.0, PENDULUM_HEIGHT + a_len)],
        dtype=wp.vec3, device=device,
    )
    viewer.log_lines("anchor_frame", a_starts, a_ends, colors, width=0.08)


def main():
    wp.init()

    trajectory, plane_coeffs, _next_states, _joint_target_pos = load_trajectory(HDF5_PATH, TRAJECTORY_INDEX)
    num_steps = trajectory.shape[0]
    total_sim_time = num_steps * DT

    model = build_pendulum_model(num_worlds=1, device="cuda:0")
    state = model.state()

    if plane_coeffs is not None:
        set_tilted_plane_from_coefficients(
            model,
            plane_coeffs[0], plane_coeffs[1], plane_coeffs[2], plane_coeffs[3],
            world_idx=0,
        )

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_world_offsets((20.0, 20.0, 0.0))
    viewer._paused = True

    device = wp.get_device()
    step_idx = 0

    set_state_from_generalized(
        model, state,
        q0=float(trajectory[0, 0]),
        q1=float(trajectory[0, 1]),
        qd0=float(trajectory[0, 2]),
        qd1=float(trajectory[0, 3]),
    )

    print("Playback mode: recorded states only (no online control/reconstruction).")
    print(f"Replaying {num_steps} steps ({total_sim_time:.2f}s) at {PLAYBACK_SPEED}x speed.")
    print("Press SPACE to unpause / pause playback.")

    start_wall_time = None

    while viewer.is_running():
        if not viewer.is_paused():
            if start_wall_time is None:
                start_wall_time = time.time()

            elapsed = (time.time() - start_wall_time) * PLAYBACK_SPEED
            step_idx = min(int(elapsed / DT), num_steps - 1)

            if elapsed >= total_sim_time:
                start_wall_time = time.time()
                step_idx = 0

            row = trajectory[step_idx]
            set_state_from_generalized(
                model, state,
                q0=float(row[0]), q1=float(row[1]),
                qd0=float(row[2]), qd1=float(row[3]),
            )
        else:
            start_wall_time = None

        sim_time = step_idx * DT
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        draw_axes(viewer, device)
        viewer.end_frame()

        wp.synchronize()


if __name__ == "__main__":
    main()
