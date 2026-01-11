import warp as wp
from axion.math import scaled_fisher_burmeister
from axion.types import ContactInteraction
from axion.types import to_spatial_momentum
from axion.types.spatial_inertia import compute_world_inertia

from .utils import compute_constraint_compliance
from .utils import compute_constraint_compliance_batched


@wp.struct
class FrictionModelResult:
    slip_velocity: wp.vec2
    slip_coupling_factor: wp.float32


@wp.func
def compute_friction_model(
    interaction: ContactInteraction,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    force_f_prev: wp.vec2,
    force_n_prev: wp.float32,
    dt: wp.float32,
    fb_alpha: wp.float32,
    beta: wp.float32,  # <--- Assume this is already (dt * precond)
) -> FrictionModelResult:
    mu = interaction.friction_coeff
    J_t1_1, J_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_2, J_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    v_t_1 = wp.dot(J_t1_1, u_1) + wp.dot(J_t1_2, u_2)
    v_t_2 = wp.dot(J_t2_1, u_1) + wp.dot(J_t2_2, u_2)
    v_t = wp.vec2(v_t_1, v_t_2)
    v_t_norm = wp.length(v_t)

    force_f_norm = wp.length(force_f_prev)

    # FIXED: Use beta directly. Do not multiply by dt again.
    # beta is passed in as (dt * precond) from the kernel.
    phi_f = scaled_fisher_burmeister(
        v_t_norm,
        mu * force_n_prev - force_f_norm,
        fb_alpha,
        beta,
    )

    # FIXED: Use 1e-8 epsilon (1e-4 is too large for velocity/force ratios)
    w = beta * wp.max((v_t_norm - phi_f) / (beta * force_f_norm + phi_f + 1e-8), 0.0)

    result = FrictionModelResult()
    result.slip_velocity = v_t
    result.slip_coupling_factor = w
    return result


@wp.kernel
def friction_constraint_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    # --- Simulation & Solver Parameters ---
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    dt: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
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
    # force_n_prev = lambda_n_prev / (s_prev)
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-4:
        # Unconstrained: h = λ, C = 1, J = 0
        constraint_active_mask[world_idx, constr_idx1] = 0.0
        constraint_active_mask[world_idx, constr_idx2] = 0.0
        body_lambda_f[world_idx, constr_idx1] = 0.0
        body_lambda_f[world_idx, constr_idx2] = 0.0
        h_f[world_idx, constr_idx1] = 0.0
        h_f[world_idx, constr_idx2] = 0.0
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx2, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx2, 1] = wp.spatial_vector()
        C_f_values[world_idx, constr_idx1] = 0.0
        C_f_values[world_idx, constr_idx2] = 0.0
        return

    constraint_active_mask[world_idx, constr_idx1] = 1.0
    constraint_active_mask[world_idx, constr_idx2] = 1.0

    # --- 2. Gather Inputs for the Friction Model ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    u_1 = wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[world_idx, body_1]
    u_2 = wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[world_idx, body_2]

    w_t1 = compute_constraint_compliance(
        interaction.basis_a.tangent1,
        interaction.basis_b.tangent1,
        body_1,
        body_2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        world_idx,
    )

    # Tangent 2
    w_t2 = compute_constraint_compliance(
        interaction.basis_a.tangent2,
        interaction.basis_b.tangent2,
        body_1,
        body_2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        world_idx,
    )

    # Average for Isotropic Friction
    precond = (w_t1 + w_t2) * 0.5

    # Scaling for Velocity Level (dt^1)
    beta = dt * precond
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        force_f_prev,
        force_n_prev,
        dt,
        fb_alpha,
        beta,
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
    h_f[world_idx, constr_idx1] = 1.0 * (v_t1 + w * lambda_t1)
    h_f[world_idx, constr_idx2] = 1.0 * (v_t2 + w * lambda_t2)

    # Update J (Jacobian)
    J_hat_f_values[world_idx, constr_idx1, 0] = J_hat_t1_1
    J_hat_f_values[world_idx, constr_idx2, 0] = J_hat_t2_1
    J_hat_f_values[world_idx, constr_idx1, 1] = J_hat_t1_2
    J_hat_f_values[world_idx, constr_idx2, 1] = J_hat_t2_2

    # Update C (compliance)
    C_f_values[world_idx, constr_idx1] = w / dt
    C_f_values[world_idx, constr_idx2] = w / dt


@wp.kernel
def batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    # --- Simulation & Solver Parameters ---
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
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
    # force_n_prev = lambda_n_prev / (s_prev)
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-4:
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

    w_t1 = compute_constraint_compliance_batched(
        interaction.basis_a.tangent1,
        interaction.basis_b.tangent1,
        body_1,
        body_2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        batch_idx,
        world_idx,
    )

    # Tangent 2
    w_t2 = compute_constraint_compliance_batched(
        interaction.basis_a.tangent2,
        interaction.basis_b.tangent2,
        body_1,
        body_2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        batch_idx,
        world_idx,
    )

    # Average for Isotropic Friction
    precond = (w_t1 + w_t2) * 0.5

    # Scaling for Velocity Level (dt^1)
    beta = dt * precond
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms by Calling the Model ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        force_f_prev,
        force_n_prev,
        dt,
        fb_alpha,
        beta,
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
    h_f[batch_idx, world_idx, constr_idx1] = 1.0 * (v_t1 + w * lambda_t1)
    h_f[batch_idx, world_idx, constr_idx2] = 1.0 * (v_t2 + w * lambda_t2)


@wp.kernel
def fused_batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    # --- Body Property Inputs ---
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_f: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= interactions.shape[1]:
        return

    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    interaction = interactions[world_idx, contact_idx]
    mu = interaction.friction_coeff

    lambda_n_prev = body_lambda_n_prev[world_idx, contact_idx]
    s_prev = s_n_prev[world_idx, contact_idx]

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-4:
        for b in range(num_batches):
            h_f[b, world_idx, constr_idx1] = 0.0
            h_f[b, world_idx, constr_idx2] = 0.0
        return

    # --- 2. Gather Inputs for the Friction Model ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # Pre-load Basis Vectors (Static)
    J_hat_t1_1, J_hat_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_hat_t1_2, J_hat_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # Load Static Body Properties
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33(0.0)
    if body_1 >= 0:
        m_inv_1 = body_inv_mass[world_idx, body_1]
        I_inv_1 = body_inv_inertia[world_idx, body_1]

    m_inv_2 = 0.0
    I_inv_2 = wp.mat33(0.0)
    if body_2 >= 0:
        m_inv_2 = body_inv_mass[world_idx, body_2]
        I_inv_2 = body_inv_inertia[world_idx, body_2]

    for b in range(num_batches):
        u_1 = wp.spatial_vector()
        if body_1 >= 0:
            u_1 = body_u[b, world_idx, body_1]
        u_2 = wp.spatial_vector()
        if body_2 >= 0:
            u_2 = body_u[b, world_idx, body_2]
        
        # Load body transforms
        body_q_1 = wp.transform_identity()
        if body_1 >= 0:
            body_q_1 = body_q[b, world_idx, body_1]

        body_q_2 = wp.transform_identity()
        if body_2 >= 0:
            body_q_2 = body_q[b, world_idx, body_2]
        
        # Compute Preconditioning (w_t1)
        w_t1 = 0.0
        if body_1 >= 0:
            M_inv_1 = compute_world_inertia(body_q_1, m_inv_1, I_inv_1)
            w_t1 += wp.dot(J_hat_t1_1, to_spatial_momentum(M_inv_1, J_hat_t1_1))
        if body_2 >= 0:
            M_inv_2 = compute_world_inertia(body_q_2, m_inv_2, I_inv_2)
            w_t1 += wp.dot(J_hat_t1_2, to_spatial_momentum(M_inv_2, J_hat_t1_2))
            
        # Compute Preconditioning (w_t2)
        w_t2 = 0.0
        if body_1 >= 0:
            M_inv_1 = compute_world_inertia(body_q_1, m_inv_1, I_inv_1)
            w_t2 += wp.dot(J_hat_t2_1, to_spatial_momentum(M_inv_1, J_hat_t2_1))
        if body_2 >= 0:
            M_inv_2 = compute_world_inertia(body_q_2, m_inv_2, I_inv_2)
            w_t2 += wp.dot(J_hat_t2_2, to_spatial_momentum(M_inv_2, J_hat_t2_2))

        precond = (w_t1 + w_t2) * 0.5
        beta = dt * precond

        # --- 3. Compute Friction Terms by Calling the Model ---
        model_result = compute_friction_model(
            interaction,
            u_1,
            u_2,
            force_f_prev,
            force_n_prev,
            dt,
            fb_alpha,
            beta,
        )
        v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
        w = model_result.slip_coupling_factor

        lambda_t1 = body_lambda_f[b, world_idx, constr_idx1]
        lambda_t2 = body_lambda_f[b, world_idx, constr_idx2]

        # Update h_d
        if body_1 >= 0:
            wp.atomic_add(
                h_d,
                b,
                world_idx,
                body_1,
                -dt * (J_hat_t1_1 * lambda_t1 + J_hat_t2_1 * lambda_t2),
            )
        if body_2 >= 0:
            wp.atomic_add(
                h_d,
                b,
                world_idx,
                body_2,
                -dt * (J_hat_t1_2 * lambda_t1 + J_hat_t2_2 * lambda_t2),
            )

        # Update h_f (constraint violation)
        h_f[b, world_idx, constr_idx1] = 1.0 * (v_t1 + w * lambda_t1)
        h_f[b, world_idx, constr_idx2] = 1.0 * (v_t2 + w * lambda_t2)
