import warp as wp

from .utils import scaled_fisher_burmeister


@wp.kernel
def frictional_constraint_kernel_2D(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32, ndim=2),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=3),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    # --- Velocity Impulse Variables ---
    lambda_n_offset: wp.int32,
    lambda_f_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32, ndim=2),
    _lambda_prev: wp.array(dtype=wp.float32, ndim=2),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # --- Offsets for Output Arrays ---
    h_f_offset: wp.int32,
    J_f_offset: wp.int32,
    C_f_offset: wp.int32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.float32, ndim=2),
    h: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_values: wp.array(dtype=wp.float32, ndim=2),
):
    # TODO: Not finished
    contact_idx = wp.tid()
    mu = contact_friction_coeff[contact_idx]
    lambda_n = _lambda_prev[lambda_n_offset + contact_idx]

    # Early exit for inactive contacts
    if contact_gap[contact_idx] >= 0.0 or lambda_n * mu <= 1e-2:
        h[h_f_offset + 2 * contact_idx] = _lambda[lambda_f_offset + 2 * contact_idx]
        h[h_f_offset + 2 * contact_idx + 1] = _lambda[
            lambda_f_offset + 2 * contact_idx + 1
        ]

        C_values[C_f_offset + 2 * contact_idx] = 1.0
        C_values[C_f_offset + 2 * contact_idx + 1] = 1.0

        J_values[J_f_offset + 2 * contact_idx, 0] = wp.spatial_vector()
        J_values[J_f_offset + 2 * contact_idx, 1] = wp.spatial_vector()
        J_values[J_f_offset + 2 * contact_idx + 1, 0] = wp.spatial_vector()
        J_values[J_f_offset + 2 * contact_idx + 1, 1] = wp.spatial_vector()
        return

    body_a = contact_body_a[contact_idx]
    body_b = contact_body_b[contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]

    body_qd_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]

    # Tangent vectors are at index 1 and 2
    grad_c_t1_a = J_contact_a[contact_idx, 1]
    grad_c_t2_a = J_contact_a[contact_idx, 2]
    grad_c_t1_b = J_contact_b[contact_idx, 1]
    grad_c_t2_b = J_contact_b[contact_idx, 2]

    # Relative tangential velocity at the contact point
    v_t1_rel = wp.dot(grad_c_t1_a, body_qd_a) + wp.dot(grad_c_t1_b, body_qd_b)
    v_t2_rel = wp.dot(grad_c_t2_a, body_qd_a) + wp.dot(grad_c_t2_b, body_qd_b)
    v_rel = wp.vec2(v_t1_rel, v_t2_rel)
    v_rel_norm = wp.length(v_rel)

    # Current friction impulse from the global impulse vector
    lambda_f_t1 = _lambda[lambda_f_offset + 2 * contact_idx]
    lambda_f_t2 = _lambda[lambda_f_offset + 2 * contact_idx + 1]
    lambda_f = wp.vec2(lambda_f_t1, lambda_f_t2)
    lambda_f_norm = wp.length(lambda_f)

    # REGULARIZATION: Use the normal impulse from the previous Newton iteration
    # to define the friction cone size. We clamp it to a minimum value to
    # prevent the cone from collapsing on new contacts, which causes instability.
    # lambda_n = wp.max(
    #     _lambda_prev[lambda_n_offset + tid], 100.0
    # )  # TODO: Resolve this problem
    lambda_n = _lambda_prev[lambda_n_offset + contact_idx]
    friction_cone_limit = mu * lambda_n

    # Use a non-linear complementarity function to relate slip speed and friction force
    phi_f, _, _ = scaled_fisher_burmeister(
        v_rel_norm, friction_cone_limit - lambda_f_norm, fb_alpha, fb_beta
    )

    # Compliance factor `w` relates the direction of slip to the friction impulse direction.
    # It becomes the off-diagonal block in the system matrix.
    w = (v_rel_norm - phi_f) / (lambda_f_norm + phi_f + 1e-6)

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual: -J^T * λ)
    if body_a >= 0:
        g_a = -grad_c_t1_a * lambda_f_t1 - grad_c_t2_a * lambda_f_t2
        for i in range(wp.static(6)):
            wp.atomic_add(g, body_a * 6 + i, g_a[i])

    if body_b >= 0:
        g_b = -grad_c_t1_b * lambda_f_t1 - grad_c_t2_b * lambda_f_t2
        for i in range(wp.static(6)):
            wp.atomic_add(g, body_b * 6 + i, g_b[i])

    # 2. Update `h` (constraint violation residual)
    h[h_f_offset + 2 * contact_idx] = v_t1_rel + w * lambda_f_t1
    h[h_f_offset + 2 * contact_idx + 1] = v_t2_rel + w * lambda_f_t2

    # 3. Update `C` (diagonal compliance block of the system matrix)
    # This `w` value forms the coupling between the two tangential directions.
    C_values[C_f_offset + 2 * contact_idx] = w + 1e-5
    C_values[C_f_offset + 2 * contact_idx + 1] = w + 1e-5

    # 4. Update `J` (constraint Jacobian block of the system matrix)
    if body_a >= 0:
        offset_t1 = J_f_offset + 2 * contact_idx
        offset_t2 = J_f_offset + 2 * contact_idx + 1
        J_values[offset_t1, 0] = grad_c_t1_a
        J_values[offset_t2, 0] = grad_c_t2_a

    if body_b >= 0:
        offset_t1 = J_f_offset + 2 * contact_idx
        offset_t2 = J_f_offset + 2 * contact_idx + 1
        J_values[offset_t1, 1] = grad_c_t1_b
        J_values[offset_t2, 1] = grad_c_t2_b
