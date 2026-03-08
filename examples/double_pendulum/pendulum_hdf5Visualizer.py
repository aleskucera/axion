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
    / "pendulumContacts.hdf5"
)

DT = 0.01
TRAJECTORY_INDEX = 7
PLAYBACK_SPEED = 1.0


def load_trajectory(hdf5_path: pathlib.Path, traj_idx: int):
    """Load a single trajectory and its plane coefficients from the HDF5 dataset.

    Returns:
        states: Array of shape (num_timesteps, 4) with columns [q0, q1, q0_dot, q1_dot].
        plane_coeffs: Array of shape (4,) with (a, b, c, d) or None if not present.
    """
    with h5py.File(hdf5_path, "r") as f:
        states = np.array(f["data"]["states"][:, traj_idx, :])
        if "plane_coefficients" in f["data"]:
            # shape (num_timesteps, num_trajectories, 4) -> use first step for this trajectory
            plane_coeffs = np.array(f["data"]["plane_coefficients"][0, traj_idx, :])
        else:
            plane_coeffs = None
    print(f"Loaded trajectory {traj_idx}: {states.shape[0]} timesteps, "
          f"q0 range [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}], "
          f"q1 range [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
    if plane_coeffs is not None:
        print(f"  Plane coefficients (a,b,c,d): [{plane_coeffs[0]:.4f}, {plane_coeffs[1]:.4f}, "
              f"{plane_coeffs[2]:.4f}, {plane_coeffs[3]:.4f}]")
    return states, plane_coeffs


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


def _tilted_plane_shape_index(model: newton.Model, world_idx: int = 0) -> int:
    """Return the shape index of the tilted plane in the given world.
    Same logic as AxionEngineWrapper: last static plane in the world (added after ground)."""
    shape_type = model.shape_type.numpy()
    shape_body = model.shape_body.numpy()
    shape_world = model.shape_world.numpy()
    is_static_plane = (
        (shape_type == int(newton.GeoType.PLANE)) & (shape_body == -1)
    )
    plane_indices = np.where(is_static_plane)[0]
    world_planes = plane_indices[shape_world[plane_indices] == world_idx]
    assert world_planes.size >= 2, (
        f"Expected at least 2 plane shapes in world {world_idx}, found {world_planes.size}"
    )
    return int(world_planes[-1])


def set_tilted_plane_from_coefficients(
    model: newton.Model,
    a: float,
    b: float,
    c: float,
    d: float = 0.0,
    world_idx: int = 0,
) -> None:
    """Set the tilted plane orientation from plane equation ax + by + cz + d = 0.
    (a,b,c) is the normal; it is normalized. The plane in the scene passes through
    the origin (d=0 in dataset), so we only set rotation. If d != 0, the plane
    position could be set from -d*n for unit normal n; currently we keep position at origin."""
    n = np.array([a, b, c], dtype=np.float64)
    nnorm = np.linalg.norm(n)
    if nnorm < 1e-8:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / nnorm
    rot = wp.quat_between_vectors(
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(float(n[0]), float(n[1]), float(n[2])),
    )
    transforms = model.shape_transform.numpy()
    shape_idx = _tilted_plane_shape_index(model, world_idx)
    # transform layout: p (3), q (4)
    transforms[shape_idx, 3:7] = np.array([rot[0], rot[1], rot[2], rot[3]])
    if abs(d) > 1e-8:
        # offset plane so it lies on ax+by+cz+d=0: a point on the plane is -d*n
        transforms[shape_idx, 0:3] = -d * n
    model.shape_transform.assign(
        wp.array(transforms, dtype=wp.transform, device=model.device)
    )


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

    trajectory, plane_coeffs = load_trajectory(HDF5_PATH, TRAJECTORY_INDEX)
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
