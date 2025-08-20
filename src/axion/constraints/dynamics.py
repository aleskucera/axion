import warp as wp
from axion.types import *
from axion.types import GeneralizedMass


@wp.kernel
def unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    # --- Body Property Inputs ---
    gen_mass: wp.array(dtype=GeneralizedMass),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.vec3,
    # --- Output ---
    g: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()
    if body_idx >= body_qd.shape[0]:
        return

    M = gen_mass[body_idx]
    u = body_qd[body_idx]
    u_prev = body_qd_prev[body_idx]
    f = body_f[body_idx]

    f_g = M * wp.spatial_vector(wp.vec3(), g_accel)

    g[body_idx] = M * (u - u_prev) - (f + f_g) * dt


@wp.kernel
def linesearch_dynamics_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    # --- Body Property Inputs ---
    gen_mass: wp.array(dtype=GeneralizedMass),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.vec3,
    # --- Output ---
    res_buffer: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, body_idx = wp.tid()
    if body_idx >= body_qd.shape[0]:
        return

    M = gen_mass[body_idx]
    u = body_qd[body_idx] + alphas[alpha_idx] * delta_body_qd[body_idx]
    u_prev = body_qd_prev[body_idx]
    f = body_f[body_idx]

    f_g = M * wp.spatial_vector(wp.vec3(), g_accel)

    g_b = M * (u - u_prev) - (f + f_g) * dt

    for i in range(wp.static(6)):
        st_i = wp.static(i)
        res_buffer[alpha_idx, body_idx * 6 + st_i] = g_b[st_i]
