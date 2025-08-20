import warp as wp

from .utils import get_random_idx_to_res_buffer
from .utils import scaled_fisher_burmeister


@wp.kernel
def linesearch_frictional_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_friction_coeff: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables ---
    lambda_n_offset: wp.int32,
    lambda_f_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # --- Outputs ---
    res_buffer: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, contact_idx = wp.tid()
    mu = contact_friction_coeff[contact_idx]
    lambda_n = _lambda[lambda_n_offset + contact_idx]

    # Early exit for inactive contacts
    if contact_gap[contact_idx] >= 0.0 or lambda_n * mu <= 1e-3:
        return

    alpha = alphas[alpha_idx]

    body_a = contact_body_a[contact_idx]
    body_b = contact_body_b[contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a] + alpha * delta_body_qd[body_a]

    body_qd_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b] + alpha * delta_body_qd[body_b]

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
    lambda_f_t1 = (
        _lambda[lambda_f_offset + 2 * contact_idx]
        + alpha * delta_lambda[lambda_f_offset + 2 * contact_idx]
    )
    lambda_f_t2 = (
        _lambda[lambda_f_offset + 2 * contact_idx + 1]
        + alpha * delta_lambda[lambda_f_offset + 2 * contact_idx + 1]
    )
    lambda_f = wp.vec2(lambda_f_t1, lambda_f_t2)
    lambda_f_norm = wp.length(lambda_f)

    # Use a non-linear complementarity function to relate slip speed and friction force
    phi_f, _, _ = scaled_fisher_burmeister(
        v_rel_norm, mu * lambda_n - lambda_f_norm, fb_alpha, fb_beta
    )

    # Compliance factor `w` relates the direction of slip to the friction impulse direction.
    # It becomes the off-diagonal block in the system matrix.
    w = (v_rel_norm - phi_f) / (lambda_f_norm + phi_f + 1e-6)

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual: -J^T * λ)
    if body_a >= 0:
        g_a = -grad_c_t1_a * lambda_f_t1 - grad_c_t2_a * lambda_f_t2
        for i in range(wp.static(6)):
            res_buffer[alpha_idx, body_a * 6 + i] += g_a[i]

    if body_b >= 0:
        g_b = -grad_c_t1_b * lambda_f_t1 - grad_c_t2_b * lambda_f_t2
        for i in range(wp.static(6)):
            res_buffer[alpha_idx, body_b * 6 + i] += g_b[i]

    # 2. Update `h` (constraint violation residual)
    h_t1 = v_t1_rel + w * lambda_f_t1
    h_t2 = v_t2_rel + w * lambda_f_t2
    h_sq_sum = wp.pow(h_t1, 2.0) + wp.pow(h_t2, 2.0)
    buff_idx = get_random_idx_to_res_buffer(alpha_idx + contact_idx)
    res_buffer[alpha_idx, buff_idx] += h_sq_sum
