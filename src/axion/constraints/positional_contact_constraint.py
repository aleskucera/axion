import warp as wp
from axion.math import scaled_fisher_burmeister_diff

from .utils import compute_effective_mass


@wp.func
def compute_signed_distance(
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    contact_point_a: wp.vec3,
    contact_point_b: wp.vec3,
    contact_thickness_a: wp.float32,
    contact_thickness_b: wp.float32,
    basis_n_a: wp.spatial_vector,
):
    # Extract normal from the stored Jacobian (first 3 components)
    n = wp.spatial_top(basis_n_a)

    offset_a = -contact_thickness_a * n
    p_a = wp.transform_point(body_q_1, contact_point_a) + offset_a

    offset_b = contact_thickness_b * n
    p_b = wp.transform_point(body_q_2, contact_point_b) + offset_b
    return wp.dot(n, p_a - p_b)


@wp.func
def compute_positional_local(
    body_a_idx: wp.int32,
    body_b_idx: wp.int32,
    contact_point_a: wp.vec3,
    contact_point_b: wp.vec3,
    contact_thickness_a: wp.float32,
    contact_thickness_b: wp.float32,
    basis_n_a: wp.spatial_vector,
    basis_n_b: wp.spatial_vector,
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    lambda_n: float,
    m_inv_1: float,
    I_inv_b_1: wp.mat33,
    m_inv_2: float,
    I_inv_b_2: wp.mat33,
    dt: float,
):
    # Compute Preconditioning
    effective_mass = compute_effective_mass(
        body_q_1,
        body_q_2,
        basis_n_a,
        basis_n_b,
        m_inv_1,
        I_inv_b_1,
        m_inv_2,
        I_inv_b_2,
        body_a_idx,
        body_b_idx,
    )
    precond = wp.pow(dt, 2.0) * effective_mass

    signed_distance = compute_signed_distance(
        body_q_1,
        body_q_2,
        contact_point_a,
        contact_point_b,
        contact_thickness_a,
        contact_thickness_b,
        basis_n_a,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(C_n, λ)
    phi_n, dphi_dc_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
        signed_distance,
        lambda_n,
        1.0,
        precond,
    )

    J_hat_n_1 = dphi_dc_n * basis_n_a
    J_hat_n_2 = dphi_dc_n * basis_n_b

    # Return as tuple:
    # (delta_h_d_1, delta_h_d_2, h_n_val, J_hat_n_1, J_hat_n_2, C_n_val, s_n_val)
    return (
        -J_hat_n_1 * lambda_n * dt,
        -J_hat_n_2 * lambda_n * dt,
        phi_n / dt,
        J_hat_n_1,
        J_hat_n_2,
        dphi_dlambda_n / wp.pow(dt, 2.0),
        dphi_dc_n,
    )


@wp.kernel
def positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    if contact_active_mask[world_idx, contact_idx] == 0.0:
        h_n[world_idx, contact_idx] = 0.0
        return

    lambda_n = body_lambda_n[world_idx, contact_idx]
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    body_q_1 = wp.transform_identity()
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    body_q_2 = wp.transform_identity()
    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    # Unpack only what's needed for residual
    d_h_d1, d_h_d2, h_n_val, unused_j1, unused_j2, unused_c, unused_s = compute_positional_local(
        body_1,
        body_2,
        contact_point_a[world_idx, contact_idx],
        contact_point_b[world_idx, contact_idx],
        contact_thickness_a[world_idx, contact_idx],
        contact_thickness_b[world_idx, contact_idx],
        contact_basis_n_a[world_idx, contact_idx],
        contact_basis_n_b[world_idx, contact_idx],
        body_q_1,
        body_q_2,
        lambda_n,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, d_h_d1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, d_h_d2)
    h_n[world_idx, contact_idx] = h_n_val


@wp.kernel
def positional_contact_constraint_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    stabilization_factor: wp.float32,
    compliance: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_n_values: wp.array(dtype=wp.float32, ndim=2),
    s_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    if contact_active_mask[world_idx, contact_idx] == 0.0:
        body_lambda_n[world_idx, contact_idx] = 0.0
        h_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        s_n[world_idx, contact_idx] = 0.0
        return

    lambda_n = body_lambda_n[world_idx, contact_idx]
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    body_q_1 = wp.transform_identity()
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    body_q_2 = wp.transform_identity()
    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    # Full unpack for solver
    d_h_d1, d_h_d2, h_n_val, J1, J2, c_val, s_val = compute_positional_local(
        body_1,
        body_2,
        contact_point_a[world_idx, contact_idx],
        contact_point_b[world_idx, contact_idx],
        contact_thickness_a[world_idx, contact_idx],
        contact_thickness_b[world_idx, contact_idx],
        contact_basis_n_a[world_idx, contact_idx],
        contact_basis_n_b[world_idx, contact_idx],
        body_q_1,
        body_q_2,
        lambda_n,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, d_h_d1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, d_h_d2)

    h_n[world_idx, contact_idx] = h_n_val
    C_n_values[world_idx, contact_idx] = c_val
    s_n[world_idx, contact_idx] = s_val
    J_hat_n_values[world_idx, contact_idx, 0] = J1
    J_hat_n_values[world_idx, contact_idx, 1] = J2


@wp.kernel
def batch_positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()

    if contact_active_mask[world_idx, contact_idx] == 0.0:
        h_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    lambda_n = body_lambda_n[batch_idx, world_idx, contact_idx]
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    body_q_1 = wp.transform_identity()
    if body_1 >= 0:
        body_q_1 = body_q[batch_idx, world_idx, body_1]
    body_q_2 = wp.transform_identity()
    if body_2 >= 0:
        body_q_2 = body_q[batch_idx, world_idx, body_2]

    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    d_h_d1, d_h_d2, h_n_val, unused_j1, unused_j2, unused_c, unused_s = compute_positional_local(
        body_1,
        body_2,
        contact_point_a[world_idx, contact_idx],
        contact_point_b[world_idx, contact_idx],
        contact_thickness_a[world_idx, contact_idx],
        contact_thickness_b[world_idx, contact_idx],
        contact_basis_n_a[world_idx, contact_idx],
        contact_basis_n_b[world_idx, contact_idx],
        body_q_1,
        body_q_2,
        lambda_n,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, d_h_d1)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, d_h_d2)
    h_n[batch_idx, world_idx, contact_idx] = h_n_val


@wp.kernel
def fused_batch_positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()
    if contact_idx >= contact_dist.shape[1]:
        return

    if contact_active_mask[world_idx, contact_idx] == 0.0:
        for b in range(num_batches):
            h_n[b, world_idx, contact_idx] = 0.0
        return

    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    # Pre-fetch contact data (invariant across batch)
    p_a = contact_point_a[world_idx, contact_idx]
    p_b = contact_point_b[world_idx, contact_idx]
    th_a = contact_thickness_a[world_idx, contact_idx]
    th_b = contact_thickness_b[world_idx, contact_idx]
    bn_a = contact_basis_n_a[world_idx, contact_idx]
    bn_b = contact_basis_n_b[world_idx, contact_idx]

    for b in range(num_batches):
        lambda_n = body_lambda_n[b, world_idx, contact_idx]
        body_q_1 = wp.transform_identity()
        if body_1 >= 0:
            body_q_1 = body_q[b, world_idx, body_1]
        body_q_2 = wp.transform_identity()
        if body_2 >= 0:
            body_q_2 = body_q[b, world_idx, body_2]

        d_h_d1, d_h_d2, h_n_val, unused_j1, unused_j2, unused_c, unused_s = (
            compute_positional_local(
                body_1,
                body_2,
                p_a,
                p_b,
                th_a,
                th_b,
                bn_a,
                bn_b,
                body_q_1,
                body_q_2,
                lambda_n,
                m_inv_1,
                I_inv_1,
                m_inv_2,
                I_inv_2,
                dt,
            )
        )

        if body_1 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_1, d_h_d1)
        if body_2 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_2, d_h_d2)
        h_n[b, world_idx, contact_idx] = h_n_val
