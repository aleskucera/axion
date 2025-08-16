import warp as wp

from .utils import get_random_idx_to_res_buffer
from .utils import scaled_fisher_burmeister


@wp.func
def _compute_complementarity_argument(
    grad_c_n_a: wp.spatial_vector,
    grad_c_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    c_n: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    # Relative normal velocity at the current time step (J * v), positive if separating
    delta_v_n = wp.dot(grad_c_n_a, body_qd_a) + wp.dot(grad_c_n_b, body_qd_b)

    # Relative normal velocity at the previous time step (for restitution)
    delta_v_n_prev = wp.dot(grad_c_n_a, body_qd_prev_a) + wp.dot(
        grad_c_n_b, body_qd_prev_b
    )

    # Baumgarte stabilization bias to correct penetration depth over time
    b_err = stabilization_factor / dt * c_n

    # Restitution bias based on pre-collision velocity
    # We only apply restitution if the pre-collision velocity is approaching.
    b_rest = restitution * wp.min(delta_v_n_prev, 0.0)

    return delta_v_n + b_err + b_rest


@wp.kernel
def contact_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_restitution_coeff: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables (from current Newton iterate) ---
    lambda_n: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,  # alpha for scaled_fisher_burmeister
    fb_beta: wp.float32,  # beta for scaled_fisher_burmeister
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.spatial_vector),
    h_n: wp.array(dtype=wp.float32),
    J_n_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_n_values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # Contact that are not penetrating
    if contact_gap[tid] >= 0.0:
        h_n[tid] = lambda_n[tid]
        C_n_values[tid] = 1.0
        J_n_values[tid, 0] = wp.spatial_vector()
        J_n_values[tid, 1] = wp.spatial_vector()
        return

    c_n = contact_gap[tid]
    body_a = contact_body_a[tid]
    body_b = contact_body_b[tid]

    # The normal direction Jacobian is the first of the three (normal, tangent1, tangent2)
    grad_c_n_a = J_contact_a[tid, 0]
    grad_c_n_b = J_contact_b[tid, 0]

    e = contact_restitution_coeff[tid]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    body_qd_prev_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]
        body_qd_prev_a = body_qd_prev[body_a]

    body_qd_b = wp.spatial_vector()
    body_qd_prev_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]
        body_qd_prev_b = body_qd_prev[body_b]

    # Compute the velocity-level term for the complementarity function
    complementarity_arg = _compute_complementarity_argument(
        grad_c_n_a,
        grad_c_n_b,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        c_n,
        e,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister function and its derivatives
    phi_n, dphi_dlambda_n, dphi_db = scaled_fisher_burmeister(
        lambda_n[tid], complementarity_arg, fb_alpha, fb_beta
    )

    # Jacobian of the constraint w.r.t body velocities (∂φ/∂v = ∂φ/∂b * ∂b/∂v)
    J_n_a = dphi_db * grad_c_n_a
    J_n_b = dphi_db * grad_c_n_b

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual)
    if body_a >= 0:
        g[body_a] -= grad_c_n_a * lambda_n[tid]

    if body_b >= 0:
        g[body_b] -= grad_c_n_b * lambda_n[tid]

    # 2. Update `h` (constraint violation residual)
    h_n[tid] = phi_n

    # 3. Update `C` (diagonal compliance block of the system matrix: ∂h/∂λ)
    C_n_values[tid] = dphi_dlambda_n + 1e-5

    # 4. Update `J` (constraint Jacobian block of the system matrix: ∂h/∂u)
    if body_a >= 0:
        J_n_values[tid, 0] = J_n_a

    if body_b >= 0:
        J_n_values[tid, 1] = J_n_b


@wp.kernel
def linesearch_contact_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_restitution_coeff: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables (from current Newton iterate) ---
    lambda_n_offset: wp.int32,  # Start index for normal impulses in `_lambda`
    _lambda: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,  # alpha for scaled_fisher_burmeister
    fb_beta: wp.float32,  # beta for scaled_fisher_burmeister
    # --- Outputs ---
    res_buffer: wp.array(dtype=wp.float32, ndim=2),
):
    alpha_idx, contact_idx = wp.tid()

    # Contact that are not penetrating
    if contact_gap[contact_idx] >= 0.0:
        return

    alpha = alphas[alpha_idx]
    # Get the current normal impulse from the global impulse vector
    lambda_n = (
        _lambda[lambda_n_offset + contact_idx]
        + alpha * delta_lambda[lambda_n_offset + contact_idx]
    )

    c_n = contact_gap[contact_idx]
    body_a = contact_body_a[contact_idx]
    body_b = contact_body_b[contact_idx]

    # The normal direction Jacobian is the first of the three (normal, tangent1, tangent2)
    grad_c_n_a = J_contact_a[contact_idx, 0]
    grad_c_n_b = J_contact_b[contact_idx, 0]

    e = contact_restitution_coeff[contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    body_qd_prev_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a] + alpha * delta_body_qd[body_a]
        body_qd_prev_a = body_qd_prev[body_a]

    body_qd_b = wp.spatial_vector()
    body_qd_prev_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b] + alpha * delta_body_qd[body_b]
        body_qd_prev_b = body_qd_prev[body_b]

    # Compute the velocity-level term for the complementarity function
    complementarity_arg = _compute_complementarity_argument(
        grad_c_n_a,
        grad_c_n_b,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        c_n,
        e,
        dt,
        stabilization_factor,
    )

    # Evaluate the Fisher-Burmeister function and its derivatives
    phi_n, dphi_dlambda_n, dphi_db = scaled_fisher_burmeister(
        lambda_n, complementarity_arg, fb_alpha, fb_beta
    )

    # 1. Update `g` (momentum balance residual)
    if body_a >= 0:
        g_a = -grad_c_n_a * lambda_n
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            res_buffer[alpha_idx, body_a * 6 + st_i] += g_a[st_i]

    if body_b >= 0:
        g_b = -grad_c_n_b * lambda_n
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            res_buffer[alpha_idx, body_b * 6 + st_i] += g_b[st_i]

    # 2. Update `h` (constraint violation residual)
    buff_idx = get_random_idx_to_res_buffer(alpha_idx + contact_idx)
    res_buffer[alpha_idx, buff_idx] += wp.pow(phi_n, 2.0)
