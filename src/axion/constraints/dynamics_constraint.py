import warp as wp
from axion.types import compute_world_inertia
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum


@wp.kernel
def unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Property Inputs ---
    body_M: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    if body_idx >= body_u.shape[1]:
        return

    M = body_M[world_idx, body_idx]
    u = body_u[world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    f_g = to_spatial_momentum(M, wp.spatial_vector(g_accel[0], wp.vec3()))

    # Gyroscopic term: w x (I @ w)
    w = wp.spatial_bottom(u)
    f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, M.inertia @ w))

    h_d[world_idx, body_idx] = to_spatial_momentum(M, u - u_prev) - (f + f_g + f_gyro) * dt


@wp.kernel
def batch_unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    batch_body_q: wp.array(dtype=wp.transform, ndim=3),
    batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Property Inputs ---
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()
    if body_idx >= batch_body_u.shape[2]:
        return
    q = batch_body_q[batch_idx, world_idx, body_idx]
    u = batch_body_u[batch_idx, world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    m = body_mass[world_idx, body_idx]
    I = body_inertia[world_idx, body_idx]
    M = compute_world_inertia(q, m, I)

    f_g = to_spatial_momentum(M, wp.spatial_vector(g_accel[0], wp.vec3()))

    # Gyroscopic term: w x (I @ w)
    w = wp.spatial_bottom(u)
    f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, M.inertia @ w))

    h_d[batch_idx, world_idx, body_idx] = (
        to_spatial_momentum(M, u - u_prev) - (f + f_g + f_gyro) * dt
    )
