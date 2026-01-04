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


@wp.kernel
def fused_batch_unconstrained_dynamics_kernel(
    # --- Batched Inputs ---
    batch_body_q: wp.array(dtype=wp.transform, ndim=3),  # [Batch, World, Body]
    batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
    # --- Static Inputs (Loaded ONCE) ---
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Params ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    num_batches: int,
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
):
    # Launch dim: (World, Body) - NO Batch dim here!
    world_idx, body_idx = wp.tid()

    if body_idx >= batch_body_u.shape[2]:
        return

    # 1. Load Static Data into Registers
    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    # Precompute constant forces if possible (e.g. gravity depends on Mass, which is const)
    # But M (world inertia) depends on q, so f_g depends on q. We must recompute inside loop.

    # 2. Inner Loop over Batch Steps
    # We iterate through the 'Batch' dimension sequentially.
    # This allows us to reuse 'm', 'I_body', 'u_prev', 'f' without re-reading from DRAM.
    for b in range(num_batches):
        q = batch_body_q[b, world_idx, body_idx]
        u = batch_body_u[b, world_idx, body_idx]

        # Recompute World Inertia (q dependent)
        M = compute_world_inertia(q, m, I_body)

        f_g = to_spatial_momentum(M, wp.spatial_vector(g_accel[0], wp.vec3()))

        # Gyroscopic term
        w = wp.spatial_bottom(u)
        f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, M.inertia @ w))

        # Write Output
        h_d[b, world_idx, body_idx] = to_spatial_momentum(M, u - u_prev) - (f + f_g + f_gyro) * dt
