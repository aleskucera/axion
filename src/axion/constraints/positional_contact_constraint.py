import warp as wp
from axion.math import scaled_fisher_burmeister_diff
from axion.types import ContactInteraction

from .utils import compute_effective_mass


@wp.func
def compute_signed_distance(
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    interaction: ContactInteraction,
):
    # Extract normal from the stored Jacobian (first 3 components)
    n = wp.spatial_top(interaction.basis_a.normal)

    offset_a = -interaction.contact_thickness_a * n
    p_a = wp.transform_point(body_q_1, interaction.contact_point_a) + offset_a

    offset_b = interaction.contact_thickness_b * n
    p_b = wp.transform_point(body_q_2, interaction.contact_point_b) + offset_b
    return wp.dot(n, p_a - p_b)


@wp.func
def compute_positional_local(
    interaction: ContactInteraction,
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    lambda_n: float,
    m_inv_1: float,
    I_inv_b_1: wp.mat33,
    m_inv_2: float,
    I_inv_b_2: wp.mat33,
    dt: float,
):
    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    # Compute Preconditioning
    effective_mass = compute_effective_mass(
        body_q_1,
        body_q_2,
        J_n_1,
        J_n_2,
        m_inv_1,
        I_inv_b_1,
        m_inv_2,
        I_inv_b_2,
        interaction.body_a_idx,
        interaction.body_b_idx,
    )
    precond = wp.pow(dt, 2.0) * effective_mass

    signed_distance = compute_signed_distance(body_q_1, body_q_2, interaction)

    # Evaluate the Fisher-Burmeister complementarity function φ(C_n, λ)
    phi_n, dphi_dc_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
        signed_distance,
        lambda_n,
        1.0,
        precond,
    )

    J_hat_n_1 = dphi_dc_n * J_n_1
    J_hat_n_2 = dphi_dc_n * J_n_2

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
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    interaction = interactions[world_idx, contact_idx]

    if not interaction.is_active:
        h_n[world_idx, contact_idx] = 0.0
        return

    lambda_n = body_lambda_n[world_idx, contact_idx]
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

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
        interaction,
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
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    stabilization_factor: wp.float32,
    compliance: wp.float32,
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_n_values: wp.array(dtype=wp.float32, ndim=2),
    s_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    interaction = interactions[world_idx, contact_idx]

    if not interaction.is_active:
        constraint_active_mask[world_idx, contact_idx] = 0.0
        body_lambda_n[world_idx, contact_idx] = 0.0
        h_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        s_n[world_idx, contact_idx] = 0.0
        return

    constraint_active_mask[world_idx, contact_idx] = 1.0
    lambda_n = body_lambda_n[world_idx, contact_idx]
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

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
        interaction,
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
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()
    interaction = interactions[world_idx, contact_idx]

    if not interaction.is_active:
        h_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    lambda_n = body_lambda_n[batch_idx, world_idx, contact_idx]
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

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
        interaction,
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
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()
    if contact_idx >= interactions.shape[1]:
        return

    interaction = interactions[world_idx, contact_idx]
    if not interaction.is_active:
        for b in range(num_batches):
            h_n[b, world_idx, contact_idx] = 0.0
        return

    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

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
                interaction,
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
