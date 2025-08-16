import time

import numpy as np
import warp as wp


@wp.func
def dynamics_equation(
    u: wp.spatial_vector,
    u_prev: wp.spatial_vector,
    f: wp.spatial_vector,
    m: wp.float32,
    I: wp.mat33,
    dt: wp.float32,
    g_accel: wp.vec3,
):
    w = wp.spatial_top(u)
    v = wp.spatial_bottom(u)
    w_prev = wp.spatial_top(u_prev)
    v_prev = wp.spatial_bottom(u_prev)
    f_ang = wp.spatial_top(f)
    f_lin = wp.spatial_bottom(f)

    # Angular momentum balance: I * Δω - τ_ext * dt
    res_ang = I * (w - w_prev) - f_ang * dt

    # Linear momentum balance: m * Δv - f_ext * dt
    res_lin = m * (v - v_prev) - (f_lin + m * g_accel) * dt

    return wp.spatial_vector(res_ang, res_lin)


@wp.kernel
def unconstrained_dynamics_kernel_2D(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_qd_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Parameters ---
    body_mass: wp.array(dtype=wp.float32),
    body_inertia: wp.array(dtype=wp.mat33),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.vec3,
    # --- Output ---
    g: wp.array(dtype=wp.float32, ndim=2),
):
    batch_idx, body_idx = wp.tid()
    if body_idx >= body_qd.shape[0]:
        return

    g_b = dynamics_equation(
        u=body_qd[batch_idx, body_idx],
        u_prev=body_qd_prev[batch_idx, body_idx],
        f=body_f[batch_idx, body_idx],
        m=body_mass[body_idx],
        I=body_inertia[body_idx],
        dt=dt,
        g_accel=g_accel,
    )

    for i in range(wp.static(6)):
        st_i = wp.static(i)
        g[batch_idx, body_idx * 6 + st_i] = g_b[st_i]
