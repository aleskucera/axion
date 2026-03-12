import numpy as np
import warp as wp
import newton

def generalized_to_maximal(
    model: newton.Model,
    state: newton.State,
    q0: float,
    q1: float,
    qd0: float = 0.0,
    qd1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert generalized pendulum coordinates (q0, q1, qd0, qd1) to maximal coordinates
    and write them into *state* via newton.eval_fk."""
    device = state.joint_q.device
    state.joint_q.assign(wp.array([q0, q1], dtype=wp.float32, device=device))
    state.joint_qd.assign(wp.array([qd0, qd1], dtype=wp.float32, device=device))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q_np = state.body_q.numpy().reshape(-1, 7)
    body_qd_np = state.body_qd.numpy().reshape(-1, 6)
    return body_q_np, body_qd_np

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