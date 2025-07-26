import time

import numpy as np
import warp as wp
from axion.utils import scaled_fisher_burmeister
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.kernel
def frictional_constraint_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_friction_coeff: wp.array(dtype=wp.float32),
    # --- Velocity impulse variables ---
    lambda_n_offset: wp.int32,
    lambda_f_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    _lambda_prev: wp.array(dtype=wp.float32),
    # --- Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # Indices for outputs
    h_f_offset: wp.int32,
    J_f_offset: wp.int32,
    C_f_offset: wp.int32,
    # --- Outputs ---
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    if contact_gap[tid] >= 0.1:
        return

    body_a = contact_body_a[tid]
    body_b = contact_body_b[tid]

    body_qd_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]

    body_qd_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]

    # Calculate the relative velocity at the contact point
    v_t_a = wp.dot(J_contact_a[tid, 1], body_qd_a)
    v_t_b = wp.dot(J_contact_b[tid, 1], body_qd_b)

    v_b_a = wp.dot(J_contact_a[tid, 2], body_qd_a)
    v_b_b = wp.dot(J_contact_b[tid, 2], body_qd_b)

    v_t_rel = v_t_b + v_t_a
    v_b_rel = v_b_b + v_b_a

    v_rel = wp.vec2(v_t_rel, v_b_rel)
    v_rel_norm = wp.length(v_rel)

    lambda_f_t = _lambda[lambda_f_offset + 2 * tid]
    lambda_f_b = _lambda[lambda_f_offset + 2 * tid + 1]

    lambda_f = wp.vec2(lambda_f_t, lambda_f_b)
    lambda_f_norm = wp.length(lambda_f)

    lambda_n = wp.max(
        _lambda_prev[lambda_n_offset + tid], 50.0
    )  # TODO: Why this hack is necessary?

    lambda_n = _lambda_prev[lambda_n_offset + tid]
    mu = contact_friction_coeff[tid]
    phi_f, _, _ = scaled_fisher_burmeister(
        v_rel_norm, mu * lambda_n - lambda_f_norm, fb_alpha, fb_beta
    )

    # Compliance factor
    w = (v_rel_norm - phi_f) / (lambda_f_norm + phi_f + 1e-6)

    # --- g --- (momentum balance)
    if body_a >= 0:
        g_a = -J_contact_a[tid, 1] * lambda_f_t - J_contact_a[tid, 2] * lambda_f_b
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, body_a * 6 + st_i, g_a[st_i])

    if body_b >= 0:
        g_b = -J_contact_b[tid, 1] * lambda_f_t - J_contact_b[tid, 2] * lambda_f_b
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, body_b * 6 + st_i, g_b[st_i])

    # --- h --- (vector of the constraint errors)
    h[h_f_offset + 2 * tid] = v_t_rel + w * lambda_f_t
    h[h_f_offset + 2 * tid + 1] = v_b_rel + w * lambda_f_b

    # --- C --- (compliance block)
    C_values[C_f_offset + 2 * tid] = w
    C_values[C_f_offset + 2 * tid + 1] = w

    # --- J --- (constraint Jacobian block)
    if body_a >= 0:
        offset_t = J_f_offset + 2 * tid
        offset_b = J_f_offset + 2 * tid + 1
        J_values[offset_t, 0] = J_contact_a[tid, 1]
        J_values[offset_b, 0] = J_contact_a[tid, 2]

    if body_b >= 0:
        offset_t = J_f_offset + 2 * tid
        offset_b = J_f_offset + 2 * tid + 1
        J_values[offset_t, 1] = J_contact_b[tid, 1]
        J_values[offset_b, 1] = J_contact_b[tid, 2]
