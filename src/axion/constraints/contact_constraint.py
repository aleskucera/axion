import warp as wp
from axion.types import *
from axion.types import ContactInteraction
from axion.types import GeneralizedMass

from .utils import scaled_fisher_burmeister
from .utils import scaled_fisher_burmeister_new


@wp.func
def compute_normal_constraint_term(
    interaction: ContactInteraction,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    J_n_a = interaction.basis_a.normal
    J_n_b = interaction.basis_b.normal

    # Relative normal velocity at the current time step (J * v), positive if separating.
    relative_velocity_n = wp.dot(J_n_a, body_qd_a) + wp.dot(J_n_b, body_qd_b)

    # Relative normal velocity at the previous time step (for restitution).
    relative_velocity_n_prev = wp.dot(J_n_a, body_qd_prev_a) + wp.dot(J_n_b, body_qd_prev_b)

    # --- Bias Terms ---

    # 1. Baumgarte stabilization bias to correct positional error (penetration) over time.
    #    We use wp.max to ensure we only correct for actual penetration (depth > 0).
    positional_correction_bias = -(stabilization_factor / dt) * wp.max(
        0.0, interaction.penetration_depth
    )

    # 2. Restitution bias based on pre-collision velocity.
    #    We only apply restitution if the pre-collision velocity is approaching (negative relative velocity).
    restitution_bias = interaction.restitution_coeff * wp.min(relative_velocity_n_prev, 0.0)

    # The final term for the complementarity function.
    # This represents the "effective" relative velocity after accounting for error correction and restitution.
    result = relative_velocity_n + positional_correction_bias + restitution_bias
    # wp.printf("Result: %f\n", result)
    return relative_velocity_n + positional_correction_bias + restitution_bias


@wp.kernel
def contact_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    lambda_n: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=ContactInteraction),
    gen_inv_mass: wp.array(dtype=GeneralizedMass),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.spatial_vector),
    h_n: wp.array(dtype=wp.float32),
    J_n_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_n_values: wp.array(dtype=wp.float32),
):
    contact_idx = wp.tid()
    interaction = interactions[contact_idx]

    # The normal impulse for this specific contact
    lambda_normal = lambda_n[contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        # The constraint residual is simply the impulse (h = λ),
        # which drives it to zero if unconstrained.
        h_n[contact_idx] = lambda_normal
        C_n_values[contact_idx] = 1.0
        return

    # Unpack body indices for clarity
    body_a_idx = interaction.body_a_idx
    body_b_idx = interaction.body_b_idx

    # Unpack Jacobian basis vectors
    J_n_a = interaction.basis_a.normal
    J_n_b = interaction.basis_b.normal

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a, body_qd_prev_a = wp.spatial_vector(), wp.spatial_vector()
    if body_a_idx >= 0:
        body_qd_a = body_qd[body_a_idx]
        body_qd_prev_a = body_qd_prev[body_a_idx]

    body_qd_b, body_qd_prev_b = wp.spatial_vector(), wp.spatial_vector()
    if body_b_idx >= 0:
        body_qd_b = body_qd[body_b_idx]
        body_qd_prev_b = body_qd_prev[body_b_idx]

    # Compute the velocity-level term for the complementarity function
    constraint_term_a = compute_normal_constraint_term(
        interaction,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        dt,
        stabilization_factor,
    )

    # TODO: Fix this
    Minv = gen_inv_mass[body_a_idx] + gen_inv_mass[body_b_idx]
    r = wp.dot(J_n_a, Minv * J_n_a)

    # Evaluate the Fisher-Burmeister complementarity function φ(a, λ)
    phi_n, dphi_da, dphi_dlambda = scaled_fisher_burmeister_new(constraint_term_a, lambda_normal, r)

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual): g -= J^T * λ
    if body_a_idx >= 0:
        g[body_a_idx] -= J_n_a * lambda_normal
    if body_b_idx >= 0:
        g[body_b_idx] -= J_n_b * lambda_normal

    # 2. Update `h` (constraint violation residual): h_n = φ(λ, b)
    h_n[contact_idx] = phi_n

    # 3. Update `C` (Compliance block): C = ∂h/∂λ = ∂φ/∂λ
    C_n_values[contact_idx] = dphi_dlambda + compliance

    # 4. Update `J` (Jacobian block): J = ∂h/∂v = (∂φ/∂b * ∂b/∂v) = dphi_db * J_n
    if body_a_idx >= 0:
        J_n_values[contact_idx, 0] = dphi_da * J_n_a
    if body_b_idx >= 0:
        J_n_values[contact_idx, 1] = dphi_da * J_n_b


@wp.kernel
def linesearch_contact_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda_n: wp.array(dtype=wp.float32),
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    lambda_n: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=ContactInteraction),
    gen_inv_mass: wp.array(dtype=GeneralizedMass),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g_alpha: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_alpha_n: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, contact_idx = wp.tid()
    interaction = interactions[contact_idx]

    alpha = alphas[alpha_idx]

    # The normal impulse for this specific contact
    lambda_normal = lambda_n[contact_idx] + alpha * delta_lambda_n[contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        h_alpha_n[alpha_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_a_idx = interaction.body_a_idx
    body_b_idx = interaction.body_b_idx

    # Unpack Jacobian basis vectors
    J_n_a = interaction.basis_a.normal
    J_n_b = interaction.basis_b.normal

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a, body_qd_prev_a = wp.spatial_vector(), wp.spatial_vector()
    if body_a_idx >= 0:
        body_qd_a = body_qd[body_a_idx] + alpha * delta_body_qd[body_a_idx]
        body_qd_prev_a = body_qd_prev[body_a_idx]

    body_qd_b, body_qd_prev_b = wp.spatial_vector(), wp.spatial_vector()
    if body_b_idx >= 0:
        body_qd_b = body_qd[body_b_idx] + alpha * delta_body_qd[body_b_idx]
        body_qd_prev_b = body_qd_prev[body_b_idx]

    # Compute the velocity-level term for the complementarity function
    constraint_term_a = compute_normal_constraint_term(
        interaction,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        dt,
        stabilization_factor,
    )

    # TODO: Check if this is correct
    Minv = gen_inv_mass[body_a_idx] + gen_inv_mass[body_b_idx]
    r = wp.dot(J_n_a, Minv * J_n_a)

    # Evaluate the Fisher-Burmeister complementarity function φ(λ, b)
    phi_n, _, _ = scaled_fisher_burmeister_new(constraint_term_a, lambda_normal, r)

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual): g -= J^T * λ
    if body_a_idx >= 0:
        g_alpha[alpha_idx, body_a_idx] -= J_n_a * lambda_normal
    if body_b_idx >= 0:
        g_alpha[alpha_idx, body_b_idx] -= J_n_b * lambda_normal

    # 2. Update `h` (constraint violation residual): h_n = φ(λ, b)
    h_alpha_n[alpha_idx, contact_idx] = phi_n
