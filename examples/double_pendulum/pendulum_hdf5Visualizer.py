import pathlib
import time

import h5py
import numpy as np
import newton
import warp as wp

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

from pendulum_articulation_definition import build_pendulum_model, PENDULUM_HEIGHT

HDF5_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src" / "axion" / "neural_solver" / "datasets" / "Pendulum"
    / "pendulumStatesOnlyTrain1Mlen2000envs250seed0.hdf5"
)

DT = 0.01
TRAJECTORY_INDEX = 100
PLAYBACK_SPEED = 1.0


def load_trajectory(hdf5_path: pathlib.Path, traj_idx: int) -> np.ndarray:
    """Load a single trajectory from the HDF5 dataset.

    Returns:
        Array of shape (num_timesteps, 4) with columns [q0, q1, q0_dot, q1_dot].
    """
    with h5py.File(hdf5_path, "r") as f:
        states = f["data"]["states"][:, traj_idx, :]
    print(f"Loaded trajectory {traj_idx}: {states.shape[0]} timesteps, "
          f"q0 range [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}], "
          f"q1 range [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
    return states


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

    trajectory = load_trajectory(HDF5_PATH, TRAJECTORY_INDEX)
    num_steps = trajectory.shape[0]
    total_sim_time = num_steps * DT

    model = build_pendulum_model(num_worlds=1, device="cuda:0")
    state = model.state()

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
