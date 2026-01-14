import warp as wp
from axion.math import scaled_fisher_burmeister
from axion.types import ContactInteraction
from axion.types import SpatialInertia

from .utils import compute_effective_mass


@wp.struct
class FrictionModelResult:
    slip_velocity: wp.vec2
    slip_coupling_factor: wp.float32


# @wp.func
# def compute_friction_model(
#     interaction: ContactInteraction,
#     u_1: wp.spatial_vector,
#     u_2: wp.spatial_vector,
#     force_f_prev: wp.vec2,
#     force_n_prev: wp.float32,
#     dt: wp.float32,
#     precond: wp.float32,  # Represents (dt * effective_mass)
# ) -> FrictionModelResult:
#     mu = interaction.friction_coeff
#     J_t1_1, J_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
#     J_t1_2, J_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2
#
#     v_t_1 = wp.dot(J_t1_1, u_1) + wp.dot(J_t1_2, u_2)
#     v_t_2 = wp.dot(J_t2_1, u_1) + wp.dot(J_t2_2, u_2)
#     v_t = wp.vec2(v_t_1, v_t_2)
#     v_t_norm = wp.length(v_t)
#
#     force_f_norm = wp.length(force_f_prev)
#
#     # Solve 1D complementarity for friction magnitude
#     phi_f = scaled_fisher_burmeister(
#         v_t_norm,
#         mu * force_n_prev - force_f_norm,
#         1.0,  # alpha for friction usually 1.0 (pure force/velocity)
#         0.1 * precond,
#     )
#
#     # Compute coupling factor w
#     # (v_t_norm - phi_f) is the "clamped" velocity magnitude
#     denom = precond * force_f_norm + phi_f + 1e-6  # Increased epsilon
#     w = 0.1 * precond * wp.max((v_t_norm - phi_f) / denom, 0.1)
#
#     # Clamp w to avoid exploding gradients if effective mass is weird
#     w = wp.min(w, 1e1)
#     w = wp.max(w, 1e-4)
#
#     result = FrictionModelResult()
#     result.slip_velocity = v_t
#     result.slip_coupling_factor = w
#     return result


@wp.func
def compute_friction_model(
    interaction: ContactInteraction,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    force_f_prev: wp.vec2,
    force_n_prev: wp.float32,
    dt: wp.float32,
    precond: wp.float32,
) -> FrictionModelResult:
    mu = interaction.friction_coeff
    J_t1_1, J_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_t1_2, J_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    v_t_1 = wp.dot(J_t1_1, u_1) + wp.dot(J_t1_2, u_2)
    v_t_2 = wp.dot(J_t2_1, u_1) + wp.dot(J_t2_2, u_2)
    v_t = wp.vec2(v_t_1, v_t_2)
    v_t_norm = wp.length(v_t)

    force_f_norm = wp.length(force_f_prev)

    # 1. Define the scaling factor r
    # The paper suggests r = h / effective_mass.
    # Your 'precond' is (dt * effective_mass), which is roughly correct
    # for mapping Force -> Velocity.
    r = precond

    # 2. Compute Scaled Residual (phi)
    # NO 0.1 factor here. Use r directly.
    gap = mu * force_n_prev - force_f_norm
    phi_f = scaled_fisher_burmeister(v_t_norm, gap, 1.0, r)

    # 3. Compute Compliance (w)
    # Using the fixed point derivation derived from the scaled phi
    # The 'r' factor must match EXACTLY what was used in scaled_fisher_burmeister

    # Robust denominator with consistent scaling
    denom_eps = 1e-6
    denominator = r * force_f_norm + phi_f + denom_eps

    # Calculate w
    # Note: If phi_f is close to v_t_norm, w approaches 0 (slip)
    # If phi_f is negative (stick), w becomes large.
    numerator = v_t_norm - phi_f
    w = r * (numerator / denominator)

    # 4. Safety Clamping
    # We must clamp w to prevent the condition number from exploding.
    # A max value of 1e4 or 1e5 is safe for float32. 10.0 (1e1) is too soft.
    w = wp.max(w, 0.0)
    w = wp.min(w, 1e5)

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
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
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

    # --- 1. Handle Early Exit for Inactive/Non-Frictional Contacts ---
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-6:
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
    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        u_1 = body_u[world_idx, body_1]
        M_inv_1 = world_M_inv[world_idx, body_1]

    u_2 = wp.spatial_vector()
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        u_2 = body_u[world_idx, body_2]
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Compute effective mass (J M^-1 J^T) using pre-computed world inertia
    w_t1 = compute_effective_mass(
        interaction.basis_a.tangent1,
        interaction.basis_b.tangent1,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    w_t2 = compute_effective_mass(
        interaction.basis_a.tangent2,
        interaction.basis_b.tangent2,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    # Average for Isotropic Friction Model
    effective_mass = (w_t1 + w_t2) * 0.5
    precond = dt * effective_mass

    # Previous State
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        force_f_prev,
        force_n_prev,
        dt,
        precond,
    )
    v_t1, v_t2 = model_result.slip_velocity.x, model_result.slip_velocity.y
    w = model_result.slip_coupling_factor

    # # DEBUG: Friction Convergence
    # if world_idx == 0 and contact_idx == 0:
    #     v_t_norm = wp.length(model_result.slip_velocity)
    #     f_prev_norm = wp.length(force_f_prev)
    #     gap = mu * force_n_prev - f_prev_norm
    #     # Re-compute phi for logging (it's internal to the function otherwise)
    #     phi_debug = scaled_fisher_burmeister(v_t_norm, gap, 1.0, precond)
    #
    #     wp.printf(
    #         "Fric[0]: v_t=%.6f | mu*N=%.6f | |f_prev|=%.6f | phi=%.6f | w=%.6f | pre=%.6f\n",
    #         v_t_norm,
    #         mu * force_n_prev,
    #         f_prev_norm,
    #         phi_debug,
    #         w,
    #         precond,
    #     )

    # --- 4. Assemble System Matrix Components ---
    J_hat_t1_1, J_hat_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_hat_t1_2, J_hat_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    lambda_t1 = body_lambda_f[world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[world_idx, constr_idx2]

    # Update h_d (dynamics residual)
    if body_1 >= 0:
        wp.atomic_add(
            h_d, world_idx, body_1, -dt * (J_hat_t1_1 * lambda_t1 + J_hat_t2_1 * lambda_t2)
        )
    if body_2 >= 0:
        wp.atomic_add(
            h_d, world_idx, body_2, -dt * (J_hat_t1_2 * lambda_t1 + J_hat_t2_2 * lambda_t2)
        )

    # Update h_f (constraint violation)
    h_f[world_idx, constr_idx1] = v_t1 + w * lambda_t1
    h_f[world_idx, constr_idx2] = v_t2 + w * lambda_t2

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
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
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

    # --- 1. Handle Early Exit ---
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-6:
        h_f[batch_idx, world_idx, constr_idx1] = 0.0
        h_f[batch_idx, world_idx, constr_idx2] = 0.0
        return

    # --- 2. Gather Inputs for the Friction Model ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    u_1 = wp.spatial_vector()
    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        u_1 = body_u[batch_idx, world_idx, body_1]
        M_inv_1 = world_M_inv[world_idx, body_1]

    u_2 = wp.spatial_vector()
    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        u_2 = body_u[batch_idx, world_idx, body_2]
        M_inv_2 = world_M_inv[world_idx, body_2]

    # Compute effective mass using frozen inertia
    effective_mass_t1 = compute_effective_mass(
        interaction.basis_a.tangent1,
        interaction.basis_b.tangent1,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    effective_mass_t2 = compute_effective_mass(
        interaction.basis_a.tangent2,
        interaction.basis_b.tangent2,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    effective_mass = (effective_mass_t1 + effective_mass_t2) * 0.5
    precond = dt * effective_mass

    # Previous lambda
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # --- 3. Compute Friction Terms ---
    model_result = compute_friction_model(
        interaction,
        u_1,
        u_2,
        force_f_prev,
        force_n_prev,
        dt,
        precond,
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

    # Update h_f
    h_f[batch_idx, world_idx, constr_idx1] = v_t1 + w * lambda_t1
    h_f[batch_idx, world_idx, constr_idx2] = v_t2 + w * lambda_t2


@wp.kernel
def fused_batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
    world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
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

    # --- 1. Handle Early Exit ---
    force_n_prev = lambda_n_prev

    if mu * force_n_prev <= 1e-6:
        for b in range(num_batches):
            h_f[b, world_idx, constr_idx1] = 0.0
            h_f[b, world_idx, constr_idx2] = 0.0
        return

    # --- 2. Gather Inputs (Static across batches) ---
    body_1 = interaction.body_a_idx
    body_2 = interaction.body_b_idx

    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # Pre-load Basis Vectors
    J_hat_t1_1, J_hat_t2_1 = interaction.basis_a.tangent1, interaction.basis_a.tangent2
    J_hat_t1_2, J_hat_t2_2 = interaction.basis_b.tangent1, interaction.basis_b.tangent2

    # Pre-load Spatial Inertia (Frozen)
    M_inv_1 = SpatialInertia()
    if body_1 >= 0:
        M_inv_1 = world_M_inv[world_idx, body_1]

    M_inv_2 = SpatialInertia()
    if body_2 >= 0:
        M_inv_2 = world_M_inv[world_idx, body_2]

    effective_mass_t1 = compute_effective_mass(
        interaction.basis_a.tangent1,
        interaction.basis_b.tangent1,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    effective_mass_t2 = compute_effective_mass(
        interaction.basis_a.tangent2,
        interaction.basis_b.tangent2,
        M_inv_1,
        M_inv_2,
        body_1,
        body_2,
    )

    effective_mass = (effective_mass_t1 + effective_mass_t2) * 0.5
    precond = dt * effective_mass

    # --- 3. Iterate over Batches ---
    for b in range(num_batches):
        u_1 = wp.spatial_vector()
        if body_1 >= 0:
            u_1 = body_u[b, world_idx, body_1]
        u_2 = wp.spatial_vector()
        if body_2 >= 0:
            u_2 = body_u[b, world_idx, body_2]

        # Compute Friction Terms (only velocity dependent part varies)
        model_result = compute_friction_model(
            interaction,
            u_1,
            u_2,
            force_f_prev,
            force_n_prev,
            dt,
            precond,
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

        # Update h_f
        h_f[b, world_idx, constr_idx1] = v_t1 + w * lambda_t1
        h_f[b, world_idx, constr_idx2] = v_t2 + w * lambda_t2
