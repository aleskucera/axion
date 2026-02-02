import warp as wp
from axion.math import scaled_fisher_burmeister_diff


@wp.func
def compute_target_v_n(
    contact_dist: wp.float32,
    contact_restitution_coeff: wp.float32,
    basis_n_a: wp.spatial_vector,
    basis_n_b: wp.spatial_vector,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    u_1_prev: wp.spatial_vector,
    u_2_prev: wp.spatial_vector,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    J_n_1 = basis_n_a
    J_n_2 = basis_n_b
    c = contact_dist
    e = contact_restitution_coeff

    # Relative normal velocity at the current time step positive if separating.
    v_n_curr = wp.dot(J_n_1, u_1) + wp.dot(J_n_2, u_2)

    # Relative normal velocity at the previous time step (for restitution).
    v_n_prev = wp.dot(J_n_1, u_1_prev) + wp.dot(J_n_2, u_2_prev)

    # --- Bias Terms ---
    # 1. Restitution bias based on pre-collision velocity.
    #    We only apply restitution if the pre-collision velocity is approaching (negative relative velocity).
    restitution_bias = e * wp.min(v_n_prev, 0.0)

    # 2. Baumgarte stabilization bias to correct positional error (penetration) over time.
    #    We use wp.max to ensure we only correct for actual penetration (depth > 0).
    # TODO: Correct the positional correction bias
    positional_correction_bias = -(stabilization_factor / dt) * wp.max(0.0, c)

    # The final term for the complementarity function.
    # This represents the "effective" relative velocity after accounting for error correction and restitution.
    target_v_n = v_n_curr + positional_correction_bias + restitution_bias
    return target_v_n


@wp.kernel
def velocity_contact_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
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
    world_idx, contact_idx = wp.tid()

    # Early exit for inactive contacts.
    if contact_dist[world_idx, contact_idx] <= 0.0:
        return

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[world_idx, contact_idx]

    # Unpack body indices for clarity
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = contact_basis_n_a[world_idx, contact_idx]
    J_n_2 = contact_basis_n_b[world_idx, contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    u_1, u_1_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[world_idx, body_1]
        u_1_prev = body_u_prev[world_idx, body_1]

    u_2, u_2_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[world_idx, body_2]
        u_2_prev = body_u_prev[world_idx, body_2]

    # Compute the velocity-level term for the complementarity function
    target_v_n = compute_target_v_n(
        contact_dist[world_idx, contact_idx],
        contact_restitution_coeff[world_idx, contact_idx],
        J_n_1,
        J_n_2,
        u_1,
        u_2,
        u_1_prev,
        u_2_prev,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
    phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
        target_v_n,
        lambda_n,
        fb_alpha,
        1e-5 * dt * fb_beta,
    )

    J_hat_n_1 = dphi_dtarget_v_n * J_n_1
    J_hat_n_2 = dphi_dtarget_v_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[world_idx, contact_idx] = phi_n


@wp.kernel
def velocity_contact_constraint_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_n: wp.array(dtype=wp.float32, ndim=2),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
    s_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[world_idx, contact_idx]

    # Early exit for inactive contacts.
    if contact_dist[world_idx, contact_idx] <= 0.0:
        constraint_active_mask[world_idx, contact_idx] = 0.0
        body_lambda_n[world_idx, contact_idx] = 0.0
        h_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_j_values[world_idx, contact_idx] = 0.0
        return

    constraint_active_mask[world_idx, contact_idx] = 1.0

    # Unpack body indices for clarity
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = contact_basis_n_a[world_idx, contact_idx]
    J_n_2 = contact_basis_n_b[world_idx, contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    u_1, u_1_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[world_idx, body_1]
        u_1_prev = body_u_prev[world_idx, body_1]

    u_2, u_2_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[world_idx, body_2]
        u_2_prev = body_u_prev[world_idx, body_2]

    # Compute the velocity-level term for the complementarity function
    target_v_n = compute_target_v_n(
        contact_dist[world_idx, contact_idx],
        contact_restitution_coeff[world_idx, contact_idx],
        J_n_1,
        J_n_2,
        u_1,
        u_2,
        u_1_prev,
        u_2_prev,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
    phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
        target_v_n,
        lambda_n,
        fb_alpha,
        1e-5 * dt * fb_beta,
    )

    J_hat_n_1 = dphi_dtarget_v_n * J_n_1
    J_hat_n_2 = dphi_dtarget_v_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, -J_hat_n_1 * lambda_n * dt)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, -J_hat_n_2 * lambda_n * dt)

    # 2. Update `h_n`
    h_n[world_idx, contact_idx] = phi_n

    # 3. Update `C_n` (Compliance block)
    C_j_values[world_idx, contact_idx] = dphi_dlambda_n / 1000000.0 + 1e-5

    # 4. Update `J_hat_n`
    J_hat_n_values[world_idx, contact_idx, 0] = J_hat_n_1
    J_hat_n_values[world_idx, contact_idx, 1] = J_hat_n_2

    # 5. Update lambda_n scale
    s_n[world_idx, contact_idx] = dphi_dtarget_v_n


@wp.kernel
def batch_velocity_contact_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
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

    # The normal impulse for this specific contact
    lambda_n = body_lambda_n[batch_idx, world_idx, contact_idx]

    # Early exit for inactive contacts.
    if contact_dist[world_idx, contact_idx] <= 0.0:
        h_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = contact_basis_n_a[world_idx, contact_idx]
    J_n_2 = contact_basis_n_b[world_idx, contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    u_1, u_1_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_1 >= 0:
        u_1 = body_u[batch_idx, world_idx, body_1]
        u_1_prev = body_u_prev[world_idx, body_1]

    u_2, u_2_prev = wp.spatial_vector(), wp.spatial_vector()
    if body_2 >= 0:
        u_2 = body_u[batch_idx, world_idx, body_2]
        u_2_prev = body_u_prev[world_idx, body_2]

    # Compute the velocity-level term for the complementarity function
    target_v_n = compute_target_v_n(
        contact_dist[world_idx, contact_idx],
        contact_restitution_coeff[world_idx, contact_idx],
        J_n_1,
        J_n_2,
        u_1,
        u_2,
        u_1_prev,
        u_2_prev,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
    phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
        target_v_n,
        lambda_n,
        fb_alpha,
        1e-5 * dt * fb_beta,
    )

    J_hat_n_1 = dphi_dtarget_v_n * J_n_1
    J_hat_n_2 = dphi_dtarget_v_n * J_n_2

    # --- Update global system components ---
    # 1. Update `h_d`
    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, -dt * J_hat_n_1 * lambda_n)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, -dt * J_hat_n_2 * lambda_n)

    # 2. Update `h_n`
    h_n[batch_idx, world_idx, contact_idx] = phi_n


@wp.kernel
def fused_batch_velocity_contact_residual_kernel(
    # --- Body State Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_n: wp.array(dtype=wp.float32, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_n: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_dist.shape[1]:
        return

    # Early exit for inactive contacts.
    if contact_dist[world_idx, contact_idx] <= 0.0:
        for b in range(num_batches):
            h_n[b, world_idx, contact_idx] = 0.0
        return

    # Unpack body indices for clarity
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Unpack Jacobian basis vectors
    J_n_1 = contact_basis_n_a[world_idx, contact_idx]
    J_n_2 = contact_basis_n_b[world_idx, contact_idx]

    # Pre-fetch contact params
    restitution = contact_restitution_coeff[world_idx, contact_idx]
    dist = contact_dist[world_idx, contact_idx]

    # Pre-load Static Previous Velocities
    u_1_prev = wp.spatial_vector()
    if body_1 >= 0:
        u_1_prev = body_u_prev[world_idx, body_1]

    u_2_prev = wp.spatial_vector()
    if body_2 >= 0:
        u_2_prev = body_u_prev[world_idx, body_2]

    for b in range(num_batches):
        # The normal impulse for this specific contact
        lambda_n = body_lambda_n[b, world_idx, contact_idx]

        # Safely get body velocities (handles fixed bodies with index -1)
        u_1 = wp.spatial_vector()
        if body_1 >= 0:
            u_1 = body_u[b, world_idx, body_1]

        u_2 = wp.spatial_vector()
        if body_2 >= 0:
            u_2 = body_u[b, world_idx, body_2]

        # Compute the velocity-level term for the complementarity function
        target_v_n = compute_target_v_n(
            dist,
            restitution,
            J_n_1,
            J_n_2,
            u_1,
            u_2,
            u_1_prev,
            u_2_prev,
            dt,
            stabilization_factor,
        )

        # Evaluate the Fisher-Burmeister complementarity function φ(v_n, λ)
        phi_n, dphi_dtarget_v_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(
            target_v_n,
            lambda_n,
            fb_alpha,
            1e-5 * dt * fb_beta,
        )

        J_hat_n_1 = dphi_dtarget_v_n * J_n_1
        J_hat_n_2 = dphi_dtarget_v_n * J_n_2

        # --- Update global system components ---
        # 1. Update `h_d`
        if body_1 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_1, -dt * J_hat_n_1 * lambda_n)
        if body_2 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_2, -dt * J_hat_n_2 * lambda_n)

        # 2. Update `h_n`
        h_n[b, world_idx, contact_idx] = phi_n

