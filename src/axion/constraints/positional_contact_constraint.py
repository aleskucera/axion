import warp as wp
from axion.math import scaled_fisher_burmeister_diff
from axion.types import ContactInteraction
from axion.types import SpatialInertia

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


@wp.struct
class PositionalLocalData:
    delta_h_d_1: wp.spatial_vector
    delta_h_d_2: wp.spatial_vector
    h_n_val: float
    # These are only strictly needed for the full solver,
    # but computing them together is cheaper/safer
    J_hat_n_1: wp.spatial_vector
    J_hat_n_2: wp.spatial_vector
    C_n_val: float
    s_n_val: float


@wp.func
def compute_positional_local(
    interaction: ContactInteraction,
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    lambda_n: float,
    M_inv_1: SpatialInertia,
    M_inv_2: SpatialInertia,
    dt: float,
) -> PositionalLocalData:
    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    # Compute Preconditioning
    effective_mass = compute_effective_mass(
        J_n_1, J_n_2, M_inv_1, M_inv_2, interaction.body_a_idx, interaction.body_b_idx
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

    # Pack result
    res = PositionalLocalData()
    res.delta_h_d_1 = -J_hat_n_1 * lambda_n * dt
    res.delta_h_d_2 = -J_hat_n_2 * lambda_n * dt
    res.h_n_val = phi_n / dt

    res.J_hat_n_1 = J_hat_n_1
    res.J_hat_n_2 = J_hat_n_2
    res.C_n_val = dphi_dlambda_n / wp.pow(dt, 2.0)
    res.s_n_val = dphi_dc_n

    return res


@wp.kernel
def positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),  # Unused, but kept for signature consistency
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),  # Unused
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
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
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, body_1]

    body_q_2 = wp.transform_identity()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, body_2]

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Call Shared Logic
    data = compute_positional_local(interaction, body_q_1, body_q_2, lambda_n, M_inv_1, M_inv_2, dt)

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, data.delta_h_d_1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, data.delta_h_d_2)

    h_n[world_idx, contact_idx] = data.h_n_val


@wp.kernel
def positional_contact_constraint_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
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
        # Zero out J, C, s_n...
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
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, body_1]

    body_q_2 = wp.transform_identity()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, body_2]

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Call Shared Logic
    data = compute_positional_local(interaction, body_q_1, body_q_2, lambda_n, M_inv_1, M_inv_2, dt)

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, data.delta_h_d_1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, data.delta_h_d_2)

    h_n[world_idx, contact_idx] = data.h_n_val

    # Store Extra Solver Terms
    C_n_values[world_idx, contact_idx] = data.C_n_val
    s_n[world_idx, contact_idx] = data.s_n_val
    J_hat_n_values[world_idx, contact_idx, 0] = data.J_hat_n_1
    J_hat_n_values[world_idx, contact_idx, 1] = data.J_hat_n_2


@wp.kernel
def batch_positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
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

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Call Shared Logic
    data = compute_positional_local(interaction, body_q_1, body_q_2, lambda_n, M_inv_1, M_inv_2, dt)

    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, data.delta_h_d_1)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, data.delta_h_d_2)

    h_n[batch_idx, world_idx, contact_idx] = data.h_n_val


@wp.kernel
def fused_batch_positional_contact_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
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

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    for b in range(num_batches):
        lambda_n = body_lambda_n[b, world_idx, contact_idx]

        body_q_1 = wp.transform_identity()
        if body_1 >= 0:
            body_q_1 = body_q[b, world_idx, body_1]

        body_q_2 = wp.transform_identity()
        if body_2 >= 0:
            body_q_2 = body_q[b, world_idx, body_2]

        # Call Shared Logic
        data = compute_positional_local(
            interaction, body_q_1, body_q_2, lambda_n, M_inv_1, M_inv_2, dt
        )

        if body_1 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_1, data.delta_h_d_1)
        if body_2 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_2, data.delta_h_d_2)

        h_n[b, world_idx, contact_idx] = data.h_n_val

