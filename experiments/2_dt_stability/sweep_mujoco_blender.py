"""Run MuJoCo over a fixed list of dt values and dump trajectories for Blender.

Each dt value produces one "iteration" of the resulting npz. The Blender
importer for this experiment (experiments/2_dt_stability/import_to_blender.py)
stacks them on the timeline so you can watch the same obstacle traversal at
progressively coarser timesteps and see when MuJoCo loses stability.

Usage:
    python experiments/2_dt_stability/sweep_mujoco_blender.py \
        --export experiments/2_dt_stability/results/mujoco_dt.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_mujoco import (
    DURATION,
    HELHEST_OBSTACLE_XML,
    KV,
    MU,
    OBSTACLE_HEIGHT,
    OBSTACLE_MU,
    OBSTACLE_X,
    RAMP_TIME,
    WHEEL_VEL,
)

# Bodies that get per-step pose capture. Index 0 must be chassis.
RENDER_BODIES = ["chassis", "left_wheel", "right_wheel", "rear_wheel"]

# dt values rendered as "iterations" (chosen to span the stability story).
# Nominal-config max stable dt is ~1.5 ms; we pick a spread that shows:
#   0.001 — stable, robot smoothly traverses the obstacle
#   0.005 — stalls at the obstacle (~x=1, can't climb)
#   0.05  — chassis bounces / jumps high (z reaches ~2 m)
#   0.1   — chassis flies away (large unbounded x)
#   0.2   — catastrophic explosion
DT_LIST = [0.001, 0.005, 0.05, 0.1, 0.2]

DEFAULT_FPS = 30
PLANE_HALF_EXTENT = 5.0  # cap the ground plane render size; mujoco default is huge


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qxyz = q[:3]
    qw = q[3]
    return v + 2.0 * np.cross(qxyz, np.cross(qxyz, v) + qw * v)


def _body_relative_to_ancestor(model, body_id: int, ancestor_id: int):
    """(pos, quat_xyzw) of body_id expressed in ancestor_id's frame, or None."""
    chain: list[int] = []
    cur = body_id
    while cur != ancestor_id and cur != 0:
        chain.append(cur)
        cur = int(model.body_parentid[cur])
    if cur != ancestor_id:
        return None
    pos = np.zeros(3, dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for bid in reversed(chain):
        body_pos = np.array(model.body_pos[bid], dtype=np.float32)
        body_quat = _quat_wxyz_to_xyzw(np.array(model.body_quat[bid], dtype=np.float32))
        new_pos = pos + _quat_rotate(quat, body_pos)
        new_quat = _quat_mul(quat, body_quat)
        pos, quat = new_pos, new_quat
    return pos, quat


def extract_shapes(model) -> list[dict]:
    """Per-geom shape descriptors compatible with the Blender importer."""
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    body_to_render_idx: dict[int, int] = {}
    for i, name in enumerate(RENDER_BODIES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise RuntimeError(f"body '{name}' not found in model")
        body_to_render_idx[bid] = i

    shapes: list[dict] = []
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        gtype = int(model.geom_type[g])
        geom_pos = np.array(model.geom_pos[g], dtype=np.float32)
        geom_quat = _quat_wxyz_to_xyzw(np.array(model.geom_quat[g], dtype=np.float32))
        geom_size = np.array(model.geom_size[g], dtype=np.float32).copy()

        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            # mujoco default plane is 100×100; clip so the Blender scene isn't huge
            geom_size[0] = min(geom_size[0], PLANE_HALF_EXTENT)
            geom_size[1] = min(geom_size[1], PLANE_HALF_EXTENT)

        if body_id == 0:
            body_idx = -1
            local_xform = np.array([*geom_pos, *geom_quat], dtype=np.float32)
        elif body_id in body_to_render_idx:
            body_idx = body_to_render_idx[body_id]
            local_xform = np.array([*geom_pos, *geom_quat], dtype=np.float32)
        else:
            res = _body_relative_to_ancestor(model, body_id, chassis_id)
            if res is None:
                continue  # not chassis-attached; skip
            body_pos_in_chassis, body_quat_in_chassis = res
            geom_pos_in_chassis = body_pos_in_chassis + _quat_rotate(
                body_quat_in_chassis, geom_pos
            )
            geom_quat_in_chassis = _quat_mul(body_quat_in_chassis, geom_quat)
            local_xform = np.array(
                [*geom_pos_in_chassis, *geom_quat_in_chassis], dtype=np.float32
            )
            body_idx = body_to_render_idx[chassis_id]

        shapes.append(
            {
                "body_idx": int(body_idx),
                "geo_type": gtype,
                "geo_scale": geom_size.astype(np.float32),
                "geo_thickness": 0.0,
                "geo_is_solid": True,
                "local_xform": local_xform.astype(np.float32),
            }
        )
    return shapes


def _snapshot_poses(data, body_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(body_ids), 7), dtype=np.float32)
    for i, bid in enumerate(body_ids):
        out[i, 0:3] = data.xpos[bid]
        wxyz = np.asarray(data.xquat[bid], dtype=np.float32)
        out[i, 3:7] = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    return out


def _resample_poses(raw_poses: np.ndarray, raw_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Resample [N_raw, B, 7] body poses onto target_times. Linear xyz, nlerp quat."""
    target_T = len(target_times)
    num_bodies = raw_poses.shape[1]
    out = np.zeros((target_T, num_bodies, 7), dtype=np.float32)
    for b in range(num_bodies):
        for d in range(3):
            out[:, b, d] = np.interp(target_times, raw_times, raw_poses[:, b, d])
        for k, t_target in enumerate(target_times):
            idx = int(np.searchsorted(raw_times, t_target, side="right")) - 1
            idx = max(0, min(idx, len(raw_times) - 1))
            if idx + 1 >= len(raw_times):
                out[k, b, 3:7] = raw_poses[idx, b, 3:7]
                continue
            t0 = float(raw_times[idx])
            t1 = float(raw_times[idx + 1])
            alpha = 0.0 if t1 == t0 else (float(t_target) - t0) / (t1 - t0)
            q0 = raw_poses[idx, b, 3:7].astype(np.float32)
            q1 = raw_poses[idx + 1, b, 3:7].astype(np.float32)
            if float(np.dot(q0, q1)) < 0.0:
                q1 = -q1
            qm = (1.0 - alpha) * q0 + alpha * q1
            n = float(np.linalg.norm(qm))
            out[k, b, 3:7] = qm / max(n, 1e-9)
    return out


def simulate_dt(dt: float, target_times: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run mujoco at given dt, capture per-step poses, resample to target_times.

    Returns (resampled_poses [target_T, B, 7], is_stable). "Stable" matches the
    sweep_mujoco predicate: no NaN, chassis z stays in [0.05, 2.0], and the
    robot actually drives past the obstacle (chassis x_final > obstacle_x + 1).
    """
    xml = HELHEST_OBSTACLE_XML.format(
        dt=dt,
        kv=KV,
        mu=MU,
        obstacle_mu=OBSTACLE_MU,
        obstacle_x=OBSTACLE_X,
        obstacle_height=OBSTACLE_HEIGHT,
        chassis_qw=1.0,
        chassis_qz=0.0,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in RENDER_BODIES
    ]
    T = int(DURATION / dt)
    raw_poses: list[np.ndarray] = [_snapshot_poses(data, body_ids)]
    raw_times: list[float] = [0.0]

    has_nan = False
    z_min = float(raw_poses[0][0, 2])
    z_max = float(raw_poses[0][0, 2])
    for step in range(T):
        t = (step + 1) * dt
        ramp = min(t / RAMP_TIME, 1.0)
        wv = WHEEL_VEL * ramp
        data.ctrl[:] = [wv, wv, wv]
        mujoco.mj_step(model, data)
        snap = _snapshot_poses(data, body_ids)
        if not np.all(np.isfinite(snap)) or np.any(np.abs(snap[:, :3]) > 100.0):
            has_nan = True
            break
        raw_poses.append(snap)
        raw_times.append(t)
        z_min = min(z_min, float(snap[0, 2]))
        z_max = max(z_max, float(snap[0, 2]))

    raw_poses_arr = np.stack(raw_poses, axis=0).astype(np.float32)
    raw_times_arr = np.asarray(raw_times, dtype=np.float32)
    x_final = float(raw_poses_arr[-1, 0, 0])
    is_stable = (
        not has_nan
        and z_min > 0.05
        and z_max < 2.0
        and x_final > OBSTACLE_X + 1.0
    )
    resampled = _resample_poses(raw_poses_arr, raw_times_arr, target_times)
    return resampled, is_stable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=pathlib.Path, required=True, help="Output .npz")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help=f"Target fps (default {DEFAULT_FPS})"
    )
    args = parser.parse_args()

    target_T = int(round(DURATION * args.fps))
    target_times = np.linspace(0.0, DURATION, target_T, dtype=np.float32)

    # Build shapes once (XML is identical apart from dt)
    setup_xml = HELHEST_OBSTACLE_XML.format(
        dt=DT_LIST[0],
        kv=KV,
        mu=MU,
        obstacle_mu=OBSTACLE_MU,
        obstacle_x=OBSTACLE_X,
        obstacle_height=OBSTACLE_HEIGHT,
        chassis_qw=1.0,
        chassis_qz=0.0,
    )
    model = mujoco.MjModel.from_xml_string(setup_xml)
    shapes = extract_shapes(model)
    print(f"Extracted {len(shapes)} shape descriptors.")

    pose_iters: list[np.ndarray] = []
    iter_labels: list[str] = []
    iter_stable: list[bool] = []
    for dt in DT_LIST:
        print(f"  simulating dt={dt}s ({int(DURATION/dt)} steps)...", end=" ", flush=True)
        poses, stable = simulate_dt(dt, target_times)
        pose_iters.append(poses)
        iter_stable.append(stable)
        suffix = "" if stable else "  (unstable)"
        iter_labels.append(f"dt = {dt*1000:.1f} ms{suffix}")
        print("STABLE" if stable else "UNSTABLE")

    body_pose_iters = np.stack(pose_iters, axis=0).astype(np.float32)
    iter_dt_values = np.array(DT_LIST, dtype=np.float32)
    iter_indices = np.arange(len(DT_LIST), dtype=np.int32)
    iter_stable_arr = np.array(iter_stable, dtype=bool)

    args.export.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.export,
        dt=np.float32(1.0 / args.fps),
        fps=np.float32(args.fps),
        body_pose_iters=body_pose_iters,
        iter_indices=iter_indices,
        iter_dt_values=iter_dt_values,
        iter_labels=np.array(iter_labels, dtype=object),
        iter_stable=iter_stable_arr,
        shapes=np.array(shapes, dtype=object),
    )
    print(f"\nSaved {args.export}  ({body_pose_iters.shape}, {len(shapes)} shapes)")


if __name__ == "__main__":
    main()
