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
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
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
        # The constraint residual is simply the impulse (h = λ),tems (indices k=0
        # which drives it to zero if unconstrained.
        constraint_active_mask[world_idx, contact_idx] = 0.0
        body_lambda_n[world_idx, contact_idx] = 0.0
        h_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        return

    constraint_active_mask[world_idx, contact_idx] = 1.0

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    body_q_1 = wp.transform_identity()
    if body_1 >= 0:
        body_q_1 = body_q[world_idx, interaction.body_a_idx]

    body_q_2 = wp.transform_identity()
    if body_2 >= 0:
        body_q_2 = body_q[world_idx, interaction.body_b_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    # precond = 0.0
    # ----- Compute Preconditioning -----
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]
        # precond += wp.dot(J_n_1, to_spatial_momentum(M_inv_1, J_n_1))
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]
        # precond += wp.dot(J_n_2, to_spatial_momentum(M_inv_2, J_n_2))

    effective_mass = compute_effective_mass(J_n_1, J_n_2, M_inv_1, M_inv_2, body_1, body_2)
    precond = wp.pow(dt, 2.0) * effective_mass + 1e-6

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

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[world_idx, contact_idx] = phi_n / dt

    # 3. Update `C_n` (Compliance block)
    C_n_values[world_idx, contact_idx] = dphi_dlambda_n / wp.pow(dt, 2.0) + 1e-1

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
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
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

    body_q_1 = wp.transform_identity()
    if body_1 >= 0:
        body_q_1 = body_q[batch_idx, world_idx, interaction.body_a_idx]

    body_q_2 = wp.transform_identity()
    if body_2 >= 0:
        body_q_2 = body_q[batch_idx, world_idx, interaction.body_b_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]

    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Compute effective mass
    effective_mass = compute_effective_mass(J_n_1, J_n_2, M_inv_1, M_inv_2, body_1, body_2)
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

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[batch_idx, world_idx, contact_idx] = phi_n / dt


@wp.kernel
def fused_batch_positional_contact_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),  # Not used in batch kernel?
    # Wait, check original batch kernel. It has body_u_prev but doesn't seem to use it in the body?
    # Original: body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    # It is NOT used in the logic. I will keep it to match signature or remove it if I can confirm it's unused.
    # The original kernel definition includes it. I will include it.
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= interactions.shape[1]:
        return

    interaction = interactions[world_idx, contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        for b in range(num_batches):
            h_n[b, world_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]

    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    effective_mass = compute_effective_mass(J_n_1, J_n_2, M_inv_1, M_inv_2, body_1, body_2)
    precond = wp.pow(dt, 2.0) * effective_mass

    for b in range(num_batches):
        # The normal impulse for this specific contact
        lambda_n = body_lambda_n[b, world_idx, contact_idx]

        body_q_1 = wp.transform_identity()
        if body_1 >= 0:
            body_q_1 = body_q[b, world_idx, interaction.body_a_idx]

        body_q_2 = wp.transform_identity()
        if body_2 >= 0:
            body_q_2 = body_q[b, world_idx, interaction.body_b_idx]

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

        # --- Update global system components ---
        # 1. Update `h_d`
        if body_1 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
        if body_2 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

        # 2. Update `h_n`
        h_n[b, world_idx, contact_idx] = phi_n / dt
