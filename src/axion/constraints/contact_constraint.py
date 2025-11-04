import warp as wp
from axion.types import ContactInteraction
from axion.types import SpatialInertia

from .utils import scaled_fisher_burmeister


@wp.func
def compute_target_v_n(
    interaction: ContactInteraction,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    u_1_prev: wp.spatial_vector,
    u_2_prev: wp.spatial_vector,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal
    c = interaction.penetration_depth
    e = interaction.restitution_coeff

    # Relative normal velocity at the current time step positive if separating.
    v_n_curr = wp.dot(J_n_1, u_1) + wp.dot(J_n_2, u_2)

    # Relative normal velocity at the previous time step (for restitution).
    v_n_prev = wp.dot(J_n_1, u_1_prev) + wp.dot(J_n_2, u_2_prev)

    # --- Bias Terms ---

    # 1. Baumgarte stabilization bias to correct positional error (penetration) over time.
    #    We use wp.max to ensure we only correct for actual penetration (depth > 0).
    positional_correction_bias = -(stabilization_factor / dt) * wp.max(0.0, c)

    # 2. Restitution bias based on pre-collision velocity.
    #    We only apply restitution if the pre-collision velocity is approaching (negative relative velocity).
    restitution_bias = e * wp.min(v_n_prev, 0.0)

    # The final term for the complementarity function.
    # This represents the "effective" relative velocity after accounting for error correction and restitution.
    target_v_n = v_n_curr + positional_correction_bias + restitution_bias
    return target_v_n


@wp.kernel
def contact_constraint_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector),
    body_u_prev: wp.array(dtype=wp.spatial_vector),
    body_lambda_n: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=ContactInteraction),
    body_M_inv: wp.array(dtype=SpatialInertia),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector),
    h_n: wp.array(dtype=wp.float32),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_n_values: wp.array(dtype=wp.float32),
    s_n: wp.array(dtype=wp.float32),
):
    contact_idx = wp.tid()
    interaction = interactions[contact_idx]

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        # The constraint residual is simply the impulse (h = λ),
        # which drives it to zero if unconstrained.
        h_n[contact_idx] = lambda_n
        J_hat_n_values[contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[contact_idx, 1] = wp.spatial_vector()
        C_n_values[contact_idx] = 1.0
        return

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    # Safely get body velocities (handles fixed bodies with index -1)
    u_1, u_1_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[body_1]
        u_1_prev = body_u_prev[body_1]

    u_2, u_2_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[body_2]
        u_2_prev = body_u_prev[body_2]

    # Compute the velocity-level term for the complementarity function
    target_v_n = compute_target_v_n(
        interaction,
        u_1,
        u_2,
        u_1_prev,
        u_2_prev,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
    phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister(
        target_v_n,
        lambda_n,
        fb_alpha,
        fb_beta,
    )

    J_hat_n_1 = dphi_dtarget_v_n * J_n_1
    J_hat_n_2 = dphi_dtarget_v_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, body_1, -J_hat_n_1 * lambda_n)
        # g[body_1] -= dphi_dtarget_v_n * J_n_1 * lambda_n
    if body_2 >= 0:
        wp.atomic_add(h_d, body_2, -J_hat_n_2 * lambda_n)
        # g[body_2] -= dphi_dtarget_v_n * J_n_2 * lambda_n

    # 2. Update `h_n`
    h_n[contact_idx] = phi_n

    # 3. Update `C_n` (Compliance block)
    C_n_values[contact_idx] = dphi_dlambda_n + compliance

    # 4. Update `J_hat_n`
    J_hat_n_values[contact_idx, 0] = J_hat_n_1
    J_hat_n_values[contact_idx, 1] = J_hat_n_2

    # 5. Update lambda_n scale
    s_n[contact_idx] = dphi_dtarget_v_n


@wp.kernel
def batch_contact_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction),
    body_M_inv: wp.array(dtype=SpatialInertia),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
):
    batch_idx, contact_idx = wp.tid()
    interaction = interactions[contact_idx]

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[batch_idx, contact_idx]

    # Early exit for inactive contacts.
    if not interaction.is_active:
        h_n[batch_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    # Unpack Jacobian basis vectors
    J_n_1 = interaction.basis_a.normal
    J_n_2 = interaction.basis_b.normal

    # Safely get body velocities (handles fixed bodies with index -1)
    u_1, u_1_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[batch_idx, body_1]
        u_1_prev = body_u_prev[body_1]

    u_2, u_2_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[batch_idx, body_2]
        u_2_prev = body_u_prev[body_2]

    # Compute the velocity-level term for the complementarity function
    target_v_n = compute_target_v_n(
        interaction,
        u_1,
        u_2,
        u_1_prev,
        u_2_prev,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
    phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister(
        target_v_n,
        lambda_n,
        fb_alpha,
        fb_beta,
    )

    J_hat_n_1 = dphi_dtarget_v_n * J_n_1
    J_hat_n_2 = dphi_dtarget_v_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, body_1, -J_hat_n_1 * lambda_n)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, body_2, -J_hat_n_2 * lambda_n)

    # 2. Update `h_n`
    h_n[batch_idx, contact_idx] = phi_n


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
    M_inv: wp.array(dtype=SpatialInertia),
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
    constraint_term_a = compute_target_v_n(
        interaction,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        dt,
        stabilization_factor,
    )

    r = 1.0

    # Evaluate the Fisher-Burmeister complementarity function φ(λ, b)
    phi_n, _, _ = scaled_fisher_burmeister(
        constraint_term_a,
        lambda_normal,
        fb_alpha,
        r * fb_beta,
    )

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual): g -= J^T * λ
    if body_a_idx >= 0:
        g_alpha[alpha_idx, body_a_idx] -= J_n_a * lambda_normal
    if body_b_idx >= 0:
        g_alpha[alpha_idx, body_b_idx] -= J_n_b * lambda_normal

    # 2. Update `h` (constraint violation residual): h_n = φ(λ, b)
    h_alpha_n[alpha_idx, contact_idx] = phi_n
