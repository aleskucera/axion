import warp as wp
from axion.types import *
from axion.types import GeneralizedMass


# @wp.func
# def dynamics_equation(
#     u: wp.spatial_vector,
#     u_prev: wp.spatial_vector,
#     f: wp.spatial_vector,
#     m: wp.float32,
#     I: wp.mat33,
#     dt: wp.float32,
#     g_accel: wp.vec3,
# ):
#     w = wp.spatial_top(u)
#     v = wp.spatial_bottom(u)
#     w_prev = wp.spatial_top(u_prev)
#     v_prev = wp.spatial_bottom(u_prev)
#     f_ang = wp.spatial_top(f)
#     f_lin = wp.spatial_bottom(f)
#
#     # Angular momentum balance: I * Δω - τ_ext * dt
#     res_ang = I * (w - w_prev) - f_ang * dt
#
#     # Linear momentum balance: m * Δv - f_ext * dt
#     res_lin = m * (v - v_prev) - (f_lin + m * g_accel) * dt
#
#     return wp.spatial_vector(res_ang, res_lin)
#
#
# @wp.kernel
# def unconstrained_dynamics_kernel(
#     # --- Body State Inputs ---
#     body_qd: wp.array(dtype=wp.spatial_vector),
#     body_qd_prev: wp.array(dtype=wp.spatial_vector),
#     body_f: wp.array(dtype=wp.spatial_vector),
#     # --- Body Property Inputs ---
#     body_mass: wp.array(dtype=wp.float32),
#     body_inertia: wp.array(dtype=wp.mat33),
#     # --- Simulation Parameters ---
#     dt: wp.float32,
#     g_accel: wp.vec3,
#     # --- Output ---
#     g: wp.array(dtype=wp.spatial_vector),
# ):
#     body_idx = wp.tid()
#     if body_idx >= body_qd.shape[0]:
#         return
#
#     g_b = dynamics_equation(
#         u=body_qd[body_idx],
#         u_prev=body_qd_prev[body_idx],
#         f=body_f[body_idx],
#         m=body_mass[body_idx],
#         I=body_inertia[body_idx],
#         dt=dt,
#         g_accel=g_accel,
#     )
#
#     g[body_idx] = g_b
#
# @wp.kernel
# def linesearch_dynamics_residuals_kernel(
#     alphas: wp.array(dtype=wp.float32),
#     delta_body_qd: wp.array(dtype=wp.spatial_vector),
#     # --- Body State Inputs ---
#     body_qd: wp.array(dtype=wp.spatial_vector),
#     body_qd_prev: wp.array(dtype=wp.spatial_vector),
#     body_f: wp.array(dtype=wp.spatial_vector),
#     # --- Body Property Inputs ---
#     body_mass: wp.array(dtype=wp.float32),
#     body_inertia: wp.array(dtype=wp.mat33),
#     # --- Simulation Parameters ---
#     dt: wp.float32,
#     g_accel: wp.vec3,
#     # --- Output ---
#     res_buffer: wp.array(dtype=wp.float32, ndim=2),
# ):
#     alpha_idx, body_idx = wp.tid()
#     if body_idx >= body_qd.shape[0]:
#         return
#
#     g_b = dynamics_equation(
#         u=body_qd[body_idx] + alphas[alpha_idx] * delta_body_qd[body_idx],
#         u_prev=body_qd_prev[body_idx],
#         f=body_f[body_idx],
#         m=body_mass[body_idx],
#         I=body_inertia[body_idx],
#         dt=dt,
#         g_accel=g_accel,
#     )
#
#     for i in range(wp.static(6)):
#         st_i = wp.static(i)
#         res_buffer[alpha_idx, body_idx * 6 + st_i] = g_b[st_i]


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
