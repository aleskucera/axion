import warp as wp
from axion.types import ContactInteraction

from .utils import scaled_fisher_burmeister


@wp.struct
class FrictionModelResult:
    slip_velocity: wp.vec2
    slip_coupling_factor: wp.float32


@wp.func
def compute_friction_model(
    interaction: ContactInteraction,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    friction_impulse: wp.vec2,
    normal_impulse: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
) -> FrictionModelResult:
    # Unpack Jacobian basis vectors
    J_t1_a, J_t2_a = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_b, J_t2_b = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # --- Calculate Slip Velocity ---
    v_t1 = wp.dot(J_t1_a, body_qd_a) + wp.dot(J_t1_b, body_qd_b)
    v_t2 = wp.dot(J_t2_a, body_qd_a) + wp.dot(J_t2_b, body_qd_b)
    slip_velocity = wp.vec2(v_t1, v_t2)
    slip_speed = wp.length(slip_velocity)

    # --- Calculate Friction Impulse Magnitude & Cone Limit ---
    friction_impulse_magnitude = wp.length(friction_impulse)
    friction_cone_limit = interaction.friction_coeff * normal_impulse

    # --- Evaluate Friction Cone Complementarity ---
    complementarity_gap, _, _ = scaled_fisher_burmeister(
        slip_speed,
        friction_cone_limit - friction_impulse_magnitude,
        fb_alpha,
        fb_beta,
    )

    # --- Compute the Slip Coupling Factor 'w' ---
    slip_coupling_factor = (slip_speed - complementarity_gap) / (
        friction_impulse_magnitude + complementarity_gap + 1e-6
    )

    # --- Package and return the results ---
    result = FrictionModelResult()
    result.slip_velocity = slip_velocity
    result.slip_coupling_factor = slip_coupling_factor
    return result


@wp.kernel
def friction_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_f: wp.array(dtype=wp.float32),
    lambda_f_prev: wp.array(dtype=wp.float32),
    lambda_n_prev: wp.array(dtype=wp.float32),
    lambda_n_scale_prev: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=ContactInteraction),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.spatial_vector),
    h_f: wp.array(dtype=wp.float32),
    J_f_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_f_values: wp.array(dtype=wp.float32),
):
    contact_idx = wp.tid()
    constr1_idx = 2 * contact_idx
    constr2_idx = 2 * contact_idx + 1

    interaction = interactions[contact_idx]

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    normal_impulse = lambda_n_prev[contact_idx] / (lambda_n_scale_prev[contact_idx] + 1e-3)
    # normal_impulse = 100.0
    friction_cone_limit = interaction.friction_coeff * normal_impulse

    if not interaction.is_active or friction_cone_limit <= 1e-4:
        # Unconstrained: h = λ, C = 1, J = 0
        h_f[constr1_idx] = lambda_f[constr1_idx]
        h_f[constr2_idx] = lambda_f[constr2_idx]
        C_f_values[constr1_idx] = 1.0
        C_f_values[constr2_idx] = 1.0
        return

    # --- 2. Gather Inputs for the Friction Model ---
    body_a_idx = interaction.body_a_idx
    body_b_idx = interaction.body_b_idx

    body_qd_a = wp.spatial_vector()
    if body_a_idx >= 0:
        body_qd_a = body_qd[body_a_idx]
    body_qd_b = wp.spatial_vector()
    if body_b_idx >= 0:
        body_qd_b = body_qd[body_b_idx]

    lambda_t1 = lambda_f[constr1_idx]
    lambda_t2 = lambda_f[constr2_idx]
    lambda_t1_prev = lambda_f[constr1_idx]
    lambda_t2_prev = lambda_f_prev[constr2_idx]
    friction_impulse = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        body_qd_a,
        body_qd_b,
        friction_impulse,
        normal_impulse,
        fb_alpha,
        fb_beta,
    )
    v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
    w = model_result.slip_coupling_factor

    # --- 4. Assemble System Matrix Components ---
    J_t1_a, J_t2_a = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_b, J_t2_b = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # Update g (forces): g -= J^T * λ
    if body_a_idx >= 0:
        wp.atomic_add(g, body_a_idx, -J_t1_a * lambda_t1 - J_t2_a * lambda_t2)
    if body_b_idx >= 0:
        wp.atomic_add(g, body_b_idx, -J_t1_b * lambda_t1 - J_t2_b * lambda_t2)

    # Update h (constraint violation): h_t = v_t + w * λ_t
    h_f[constr1_idx] = v_t1 + w * lambda_t1
    h_f[constr2_idx] = v_t2 + w * lambda_t2

    # Update C (compliance): C_ff
    C_f_values[constr1_idx] = w
    C_f_values[constr2_idx] = w

    # Update J (Jacobian): J_f = [J_t1; J_t2]
    if body_a_idx >= 0:
        J_f_values[constr1_idx, 0] = J_t1_a
        J_f_values[constr2_idx, 0] = J_t2_a
    if body_b_idx >= 0:
        J_f_values[constr1_idx, 1] = J_t1_b
        J_f_values[constr2_idx, 1] = J_t2_b


@wp.kernel
def linesearch_friction_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda_f: wp.array(dtype=wp.float32),
    delta_lambda_n: wp.array(dtype=wp.float32),
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_f: wp.array(dtype=wp.float32),
    lambda_n: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=ContactInteraction),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g_alpha: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_alpha_f: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, contact_idx = wp.tid()
    constr1_idx = 2 * contact_idx
    constr2_idx = 2 * contact_idx + 1

    interaction = interactions[contact_idx]

    alpha = alphas[alpha_idx]

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    normal_impulse = lambda_n[contact_idx] + alpha * delta_lambda_n[contact_idx]
    friction_cone_limit = interaction.friction_coeff * normal_impulse

    if not interaction.is_active or friction_cone_limit <= 1e-4:
        h_alpha_f[alpha_idx, constr1_idx] = 0.0
        h_alpha_f[alpha_idx, constr2_idx] = 0.0
        return

    # --- 2. Gather Inputs for the Friction Model ---
    body_a_idx = interaction.body_a_idx
    body_b_idx = interaction.body_b_idx

    body_qd_a = wp.spatial_vector()
    if body_a_idx >= 0:
        body_qd_a = body_qd[body_a_idx] + alpha * delta_body_qd[body_a_idx]
    body_qd_b = wp.spatial_vector()
    if body_b_idx >= 0:
        body_qd_b = body_qd[body_b_idx] + alpha * delta_body_qd[body_b_idx]

    lambda_t1 = lambda_f[constr1_idx] + alpha * delta_lambda_f[constr1_idx]
    lambda_t2 = lambda_f[constr2_idx] + alpha * delta_lambda_f[constr2_idx]
    friction_impulse = wp.vec2(lambda_t1, lambda_t2)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        body_qd_a,
        body_qd_b,
        friction_impulse,
        normal_impulse,
        fb_alpha,
        fb_beta,
    )
    v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
    w = model_result.slip_coupling_factor

    # --- 4. Assemble System Matrix Components ---
    J_t1_a, J_t2_a = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_b, J_t2_b = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # Update g (forces): g -= J^T * λ
    if body_a_idx >= 0:
        wp.atomic_add(g_alpha, alpha_idx, body_a_idx, -J_t1_a * lambda_t1 - J_t2_a * lambda_t2)
    if body_b_idx >= 0:
        wp.atomic_add(g_alpha, alpha_idx, body_b_idx, -J_t1_b * lambda_t1 - J_t2_b * lambda_t2)

    # Update h (constraint violation): h_t = v_t + w * λ_t
    h_alpha_f[alpha_idx, constr1_idx] = v_t1 + w * lambda_t1
    h_alpha_f[alpha_idx, constr2_idx] = v_t2 + w * lambda_t2
