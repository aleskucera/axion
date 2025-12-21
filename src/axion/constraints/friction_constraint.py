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
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    impulse_f_prev: wp.vec2,
    impulse_n_prev: wp.float32,
    dt: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
) -> FrictionModelResult:
    mu = interaction.friction_coeff
    J_t1_1, J_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_2, J_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # --- Calculate Slip Velocity ---
    v_t_1 = wp.dot(J_t1_1, u_1) + wp.dot(J_t1_2, u_2)
    v_t_2 = wp.dot(J_t2_1, u_1) + wp.dot(J_t2_2, u_2)
    v_t = wp.vec2(v_t_1, v_t_2)
    v_t_norm = wp.length(v_t)

    # --- Calculate Friction Impulse Magnitude & Cone Limit ---
    impulse_f_norm = wp.length(impulse_f_prev)

    # --- Evaluate Friction Cone Complementarity ---
    # beta converts Force to Velocity (Compliance * dt or similar scaling)
    beta = dt * fb_beta
    phi_f, _, _ = scaled_fisher_burmeister(
        v_t_norm,
        mu * impulse_n_prev - impulse_f_norm,
        fb_alpha,
        beta,
    )

    # --- Compute the Slip Coupling Factor 'w' ---
    # Denominator: beta * F (Velocity) + phi (Velocity)
    # Result w: beta * (V/V) = beta (Compliance units V/F)
    w = beta * wp.max((v_t_norm - phi_f) / (beta * impulse_f_norm + phi_f + 1e-8), 0.0)

    # --- Package and return the results ---
    result = FrictionModelResult()
    result.slip_velocity = v_t
    result.slip_coupling_factor = w
    return result


@wp.kernel
def friction_constraint_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_f: wp.array(dtype=wp.float32, ndim=2),
    J_hat_f_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_f_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    interaction = interactions[world_idx, contact_idx]
    mu = interaction.friction_coeff

    lambda_n_prev = body_lambda_n_prev[world_idx, contact_idx]
    s_prev = s_n_prev[world_idx, contact_idx]

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    impulse_n_prev = lambda_n_prev / (s_prev + 1e-6)

    if mu * impulse_n_prev <= 1e-3:
        # Unconstrained: h = λ, C = 1, J = 0
        h_f[world_idx, constr_idx1] = 0.0
        h_f[world_idx, constr_idx2] = 0.0
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx2, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx2, 1] = wp.spatial_vector()
        C_f_values[world_idx, constr_idx1] = 0.0
        C_f_values[world_idx, constr_idx2] = 0.0
        return

    # wp.printf("Penetration depth: %f, mu: %f\n", interaction.penetration_depth, mu)
    # --- 2. Gather Inputs for the Friction Model ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    u_1 = wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[world_idx, body_1]
    u_2 = wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[world_idx, body_2]

    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    impulse_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        impulse_f_prev,
        impulse_n_prev,
        dt,
        fb_alpha,
        fb_beta,
    )
    v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
    w = model_result.slip_coupling_factor

    # --- 4. Assemble System Matrix Components ---
    J_hat_t1_1, J_hat_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_hat_t1_2, J_hat_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    lambda_t1 = body_lambda_f[world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[world_idx, constr_idx2]

    # Update g
    if body_1 >= 0:
        wp.atomic_add(
            h_d, world_idx, body_1, -dt * (J_hat_t1_1 * lambda_t1 + J_hat_t2_1 * lambda_t2)
        )
    if body_2 >= 0:
        wp.atomic_add(
            h_d, world_idx, body_2, -dt * (J_hat_t1_2 * lambda_t1 + J_hat_t2_2 * lambda_t2)
        )

    # Update h (constraint violation)
    h_f[world_idx, constr_idx1] = 0.03 * (v_t1 + w * lambda_t1)
    h_f[world_idx, constr_idx2] = 0.03 * (v_t2 + w * lambda_t2)

    # Update J (Jacobian)
    J_hat_f_values[world_idx, constr_idx1, 0] = J_hat_t1_1
    J_hat_f_values[world_idx, constr_idx2, 0] = J_hat_t2_1
    J_hat_f_values[world_idx, constr_idx1, 1] = J_hat_t1_2
    J_hat_f_values[world_idx, constr_idx2, 1] = J_hat_t2_2

    # Update C (compliance)
    C_f_values[world_idx, constr_idx1] = w
    C_f_values[world_idx, constr_idx2] = w


@wp.kernel
def batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_f: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()
    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    interaction = interactions[world_idx, contact_idx]
    mu = interaction.friction_coeff

    lambda_n_prev = body_lambda_n_prev[world_idx, contact_idx]
    s_prev = s_n_prev[world_idx, contact_idx]

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    impulse_n_prev = lambda_n_prev / (s_prev + 1e-6)

    if interaction.penetration_depth <= 0 or mu * impulse_n_prev <= 1e-3:
        h_f[batch_idx, world_idx, constr_idx1] = 0.0
        h_f[batch_idx, world_idx, constr_idx2] = 0.0
        return

    # --- 2. Gather Inputs for the Friction Model ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    u_1 = wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[batch_idx, world_idx, body_1]
    u_2 = wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[batch_idx, world_idx, body_2]

    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    impulse_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        impulse_f_prev,
        impulse_n_prev,
        dt,
        fb_alpha,
        fb_beta,
    )
    v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
    w = model_result.slip_coupling_factor

    # --- 4. Assemble System Matrix Components ---
    J_hat_t1_1, J_hat_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_hat_t1_2, J_hat_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    lambda_t1 = body_lambda_f[batch_idx, world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[batch_idx, world_idx, constr_idx2]

    # Update h_d
    if body_1 >= 0:
        wp.atomic_add(
            h_d,
            batch_idx,
            world_idx,
            body_1,
            -dt * (J_hat_t1_1 * lambda_t1 + J_hat_t2_1 * lambda_t2),
        )
    if body_2 >= 0:
        wp.atomic_add(
            h_d,
            batch_idx,
            world_idx,
            body_2,
            -dt * (J_hat_t1_2 * lambda_t1 + J_hat_t2_2 * lambda_t2),
        )

    # Update h_f (constraint violation)
    h_f[batch_idx, world_idx, constr_idx1] = 0.001 * (v_t1 + w * lambda_t1)
    h_f[batch_idx, world_idx, constr_idx2] = 0.001 * (v_t2 + w * lambda_t2)
