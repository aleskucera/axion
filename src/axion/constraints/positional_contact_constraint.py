import warp as wp
from axion.types import ContactInteraction
from axion.types import SpatialInertia
from axion.types import to_spatial_momentum

from .utils import scaled_fisher_burmeister


@wp.func
def compute_signed_distance(
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    interaction: ContactInteraction,
):
    # Extract normal from the stored Jacobian (first 3 components)
    n = wp.spatial_top(interaction.basis_a.normal)

    # Initialize world points (handles static body case correctly)
    p_a = interaction.contact_point_a
    p_b = interaction.contact_point_b

    offset_a = -interaction.contact_thickness_a * n
    p_a = wp.transform_point(body_q_1, interaction.contact_point_a) + offset_a

    offset_b = interaction.contact_thickness_b * n
    p_b = wp.transform_point(body_q_2, interaction.contact_point_b) + offset_b

    return wp.dot(n, p_a - p_b)


@wp.kernel
def positional_contact_constraint_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_n_values: wp.array(dtype=wp.float32, ndim=2),
    s_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    interaction = interactions[world_idx, contact_idx]

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[world_idx, contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        # The constraint residual is simply the impulse (h = λ),
        # which drives it to zero if unconstrained.
        h_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    body_q_1 = wp.transform()
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, interaction.body_a_idx]

    body_q_2 = wp.transform()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, interaction.body_b_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    precond = 0.0
    if body_1 >= 0:
        M_inv_1 = body_M_inv[world_idx, body_1]
        precond += wp.dot(J_n_1, to_spatial_momentum(M_inv_1, J_n_1))
    if body_2 >= 0:
        M_inv_2 = body_M_inv[world_idx, body_2]
        precond += wp.dot(J_n_2, to_spatial_momentum(M_inv_2, J_n_2))

    signed_distance = compute_signed_distance(body_q_1, body_q_2, interaction)

    # Evaluate the Fisher-Burmeister complementarity function φ(C_n, λ)
    phi_n, dphi_dc_n, dphi_dlambda_n = scaled_fisher_burmeister(
        signed_distance,
        lambda_n,
        1.0,
        wp.pow(dt, 2.0) * precond,
    )

    J_hat_n_1 = dphi_dc_n * J_n_1
    J_hat_n_2 = dphi_dc_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[world_idx, contact_idx] = phi_n / dt

    # 3. Update `C_n` (Compliance block)
    C_n_values[world_idx, contact_idx] = dphi_dlambda_n / wp.pow(dt, 2.0)

    # 4. Update `J_hat_n`
    J_hat_n_values[world_idx, contact_idx, 0] = J_hat_n_1
    J_hat_n_values[world_idx, contact_idx, 1] = J_hat_n_2

    # 5. Update lambda_n scale
    s_n[world_idx, contact_idx] = dphi_dc_n


@wp.kernel
def batch_positional_contact_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    body_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()
    interaction = interactions[world_idx, contact_idx]

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[batch_idx, world_idx, contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        h_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    body_q_1 = wp.transform()
    if body_1 >= 0:
        body_q_1 = body_q[batch_idx, world_idx, interaction.body_a_idx]

    body_q_2 = wp.transform()
    if body_2 >= 0:
        body_q_2 = body_q[batch_idx, world_idx, interaction.body_b_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    precond = 0.0
    if body_1 >= 0:
        M_inv_1 = body_M_inv[world_idx, body_1]
        precond += wp.dot(J_n_1, to_spatial_momentum(M_inv_1, J_n_1))
    if body_2 >= 0:
        M_inv_2 = body_M_inv[world_idx, body_2]
        precond += wp.dot(J_n_2, to_spatial_momentum(M_inv_2, J_n_2))

    signed_distance = compute_signed_distance(body_q_1, body_q_2, interaction)

    # Evaluate the Fisher-Burmeister complementarity function φ(C_n, λ)
    phi_n, dphi_dc_n, dphi_dlambda_n = scaled_fisher_burmeister(
        signed_distance,
        lambda_n,
        1.0,
        wp.pow(dt, 2.0) * precond,
    )

    J_hat_n_1 = dphi_dc_n * J_n_1
    J_hat_n_2 = dphi_dc_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[batch_idx, world_idx, contact_idx] = phi_n / dt
