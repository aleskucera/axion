import warp as wp
from axion.types import ContactManifold

from .utils import scaled_fisher_burmeister


@wp.kernel
def frictional_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    manifolds: wp.array(dtype=ContactManifold),
    # --- Velocity Impulse Variables ---
    lambda_f: wp.array(dtype=wp.float32),
    lambda_n_prev: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.spatial_vector),
    h_f: wp.array(dtype=wp.float32),
    J_f_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_f_values: wp.array(dtype=wp.float32),
):
    contact_idx = wp.tid()

    manifold = manifolds[contact_idx]

    mu = manifold.friction
    lambda_n = lambda_n_prev[contact_idx]

    # Early exit for inactive contacts
    if not manifold.is_active or lambda_n * mu <= 1e-2:
        h_f[2 * contact_idx] = lambda_f[2 * contact_idx]
        h_f[2 * contact_idx + 1] = lambda_f[2 * contact_idx + 1]

        C_f_values[2 * contact_idx] = 1.0
        C_f_values[2 * contact_idx + 1] = 1.0

        J_f_values[2 * contact_idx, 0] = wp.spatial_vector()
        J_f_values[2 * contact_idx, 1] = wp.spatial_vector()
        J_f_values[2 * contact_idx + 1, 0] = wp.spatial_vector()
        J_f_values[2 * contact_idx + 1, 1] = wp.spatial_vector()
        return

    body_a = manifold.point_a.body_idx
    body_b = manifold.point_b.body_idx

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]

    body_qd_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]

    # Tangent vectors are at index 1 and 2
    grad_c_t1_a = manifold.point_a.jacobian.t1
    grad_c_t2_a = manifold.point_a.jacobian.t2
    grad_c_t1_b = manifold.point_b.jacobian.t1
    grad_c_t2_b = manifold.point_b.jacobian.t2

    # Relative tangential velocity at the contact point
    v_t1_rel = wp.dot(grad_c_t1_a, body_qd_a) + wp.dot(grad_c_t1_b, body_qd_b)
    v_t2_rel = wp.dot(grad_c_t2_a, body_qd_a) + wp.dot(grad_c_t2_b, body_qd_b)
    v_rel = wp.vec2(v_t1_rel, v_t2_rel)
    v_rel_norm = wp.length(v_rel)

    # Current friction impulse from the global impulse vector
    lambda_f_t1 = lambda_f[2 * contact_idx]
    lambda_f_t2 = lambda_f[2 * contact_idx + 1]
    lambda_f_norm = wp.length(wp.vec2(lambda_f_t1, lambda_f_t2))

    # REGULARIZATION: Use the normal impulse from the previous Newton iteration
    # to define the friction cone size. We clamp it to a minimum value to
    # prevent the cone from collapsing on new contacts, which causes instability.
    # lambda_n = wp.max(
    #     _lambda_prev[lambda_n_offset + tid], 100.0
    # )  # TODO: Resolve this problem
    friction_cone_limit = mu * lambda_n_prev[contact_idx]

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
        g[body_a] += -grad_c_t1_a * lambda_f_t1 - grad_c_t2_a * lambda_f_t2

    if body_b >= 0:
        g[body_b] += -grad_c_t1_b * lambda_f_t1 - grad_c_t2_b * lambda_f_t2

    # 2. Update `h` (constraint violation residual)
    h_f[2 * contact_idx] = v_t1_rel + w * lambda_f_t1
    h_f[2 * contact_idx + 1] = v_t2_rel + w * lambda_f_t2

    # 3. Update `C` (diagonal compliance block of the system matrix)
    # This `w` value forms the coupling between the two tangential directions.
    C_f_values[2 * contact_idx] = w + 1e-5
    C_f_values[2 * contact_idx + 1] = w + 1e-5

    # 4. Update `J` (constraint Jacobian block of the system matrix)
    if body_a >= 0:
        J_f_values[2 * contact_idx, 0] = grad_c_t1_a
        J_f_values[2 * contact_idx + 1, 0] = grad_c_t2_a

    if body_b >= 0:
        J_f_values[2 * contact_idx, 1] = grad_c_t1_b
        J_f_values[2 * contact_idx + 1, 1] = grad_c_t2_b
