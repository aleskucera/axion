import warp as wp
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

    h_d[world_idx, body_idx] = to_spatial_momentum(M, u - u_prev) - (f + f_g) * dt


@wp.kernel
def batch_unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Property Inputs ---
    body_M: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()
    if body_idx >= body_u.shape[1]:
        return

    M = body_M[world_idx, body_idx]
    u = body_u[batch_idx, world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    f_g = to_spatial_momentum(M, wp.spatial_vector(g_accel[0], wp.vec3()))

    h_d[batch_idx, world_idx, body_idx] = to_spatial_momentum(M, u - u_prev) - (f + f_g) * dt
