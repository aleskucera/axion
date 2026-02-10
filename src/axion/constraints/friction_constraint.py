import warp as wp
from axion.math import scaled_fisher_burmeister

from .utils import compute_effective_mass


@wp.func
def compute_friction_model(
    mu: wp.float32,
    J_t1_1: wp.spatial_vector,
    J_t2_1: wp.spatial_vector,
    J_t1_2: wp.spatial_vector,
    J_t2_2: wp.spatial_vector,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    force_f_prev: wp.vec2,
    force_n_prev: wp.float32,
    dt: wp.float32,
    precond: wp.float32,
):  # Returns (slip_velocity, slip_coupling_factor)
    v_t_1 = wp.dot(J_t1_1, u_1) + wp.dot(J_t1_2, u_2)
    v_t_2 = wp.dot(J_t2_1, u_1) + wp.dot(J_t2_2, u_2)
    v_t = wp.vec2(v_t_1, v_t_2)

    # eps = 1e-8
    # v_t_norm = wp.sqrt(wp.dot(v_t, v_t) + eps)
    #
    # force_f_norm = wp.sqrt(wp.dot(force_f_prev, force_f_prev) + eps)

    v_t_norm = wp.length(v_t)

    force_f_norm = wp.length(force_f_prev)

    r = precond
    gap = mu * force_n_prev - force_f_norm
    phi_f = scaled_fisher_burmeister(v_t_norm, gap, 1.0, r)

    denom_eps = 1e-6
    denominator = r * force_f_norm + phi_f + denom_eps
    numerator = v_t_norm - phi_f

    w = r * (numerator / denominator)
    w = wp.max(w, 0.0)
    w = wp.min(w, 1e5)

    return v_t, w


@wp.func
def compute_friction_local(
    body_a_idx: wp.int32,
    body_b_idx: wp.int32,
    contact_friction_coeff: wp.float32,
    J_t1_1: wp.spatial_vector,
    J_t2_1: wp.spatial_vector,
    J_t1_2: wp.spatial_vector,
    J_t2_2: wp.spatial_vector,
    q_1: wp.transform,
    q_2: wp.transform,
    u_1: wp.spatial_vector,
    u_2: wp.spatial_vector,
    m_inv_1: wp.float32,
    I_inv_b_1: wp.mat33,
    m_inv_2: wp.float32,
    I_inv_b_2: wp.mat33,
    # M_inv_1: SpatialInertia,
    # M_inv_2: SpatialInertia,
    lambda_t1: float,
    lambda_t2: float,
    lambda_t1_prev: float,
    lambda_t2_prev: float,
    force_n_prev: float,
    dt: float,
):
    # Effective mass logic remains same
    w_t1 = compute_effective_mass(
        q_1,
        q_2,
        J_t1_1,
        J_t1_2,
        m_inv_1,
        I_inv_b_1,
        m_inv_2,
        I_inv_b_2,
        body_a_idx,
        body_b_idx,
    )
    w_t2 = compute_effective_mass(
        q_1,
        q_2,
        J_t2_1,
        J_t2_2,
        m_inv_1,
        I_inv_b_1,
        m_inv_2,
        I_inv_b_2,
        body_a_idx,
        body_b_idx,
    )

    effective_mass = (w_t1 + w_t2) * 0.5
    precond = dt * effective_mass
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    # Unpack tuple from model
    v_t, w = compute_friction_model(
        contact_friction_coeff,
        J_t1_1,
        J_t2_1,
        J_t1_2,
        J_t2_2,
        u_1,
        u_2,
        force_f_prev,
        force_n_prev,
        dt,
        precond,
    )

    v_t1, v_t2 = v_t.x, v_t.y

    # Residuals
    delta_h_d_1 = -dt * (J_t1_1 * lambda_t1 + J_t2_1 * lambda_t2)
    delta_h_d_2 = -dt * (J_t1_2 * lambda_t1 + J_t2_2 * lambda_t2)
    h_f_val_1 = v_t1 + w * lambda_t1
    h_f_val_2 = v_t2 + w * lambda_t2
    C_f_val = w / dt

    # Return order: delta_h1, delta_h2, hf1, hf2, Jt1_1, Jt2_1, Jt1_2, Jt2_2, Cf
    return (delta_h_d_1, delta_h_d_2, h_f_val_1, h_f_val_2, J_t1_1, J_t2_1, J_t1_2, J_t2_2, C_f_val)


@wp.kernel
def friction_constraint_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # world_M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
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

    mu = contact_friction_coeff[world_idx, contact_idx]
    force_n_prev = body_lambda_n_prev[world_idx, contact_idx]

    # --- 1. Handle Early Exit ---
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

    # --- 2. Gather Inputs ---
    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Basis
    J_t1_1 = contact_basis_t1_a[world_idx, contact_idx]
    J_t2_1 = contact_basis_t2_a[world_idx, contact_idx]
    J_t1_2 = contact_basis_t1_b[world_idx, contact_idx]
    J_t2_2 = contact_basis_t2_b[world_idx, contact_idx]

    q_1 = wp.transform()
    u_1 = wp.spatial_vector()
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        q_1 = body_q[world_idx, body_1]
        u_1 = body_u[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    q_2 = wp.transform()
    u_2 = wp.spatial_vector()
    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        q_2 = body_q[world_idx, body_2]
        u_2 = body_u[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    lambda_t1 = body_lambda_f[world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[world_idx, constr_idx2]
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]

    # --- 3. Compute Local Logic ---
    (d_h1, d_h2, hf1, hf2, J11, J21, J12, J22, cf) = compute_friction_local(
        body_1,
        body_2,
        mu,
        J_t1_1,
        J_t2_1,
        J_t1_2,
        J_t2_2,
        q_1,
        q_2,
        u_1,
        u_2,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        lambda_t1,
        lambda_t2,
        lambda_t1_prev,
        lambda_t2_prev,
        force_n_prev,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, d_h1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, d_h2)

    h_f[world_idx, constr_idx1] = hf1
    h_f[world_idx, constr_idx2] = hf2

    J_hat_f_values[world_idx, constr_idx1, 0] = J11
    J_hat_f_values[world_idx, constr_idx2, 0] = J21
    J_hat_f_values[world_idx, constr_idx1, 1] = J12
    J_hat_f_values[world_idx, constr_idx2, 1] = J22

    C_f_values[world_idx, constr_idx1] = cf
    C_f_values[world_idx, constr_idx2] = cf


@wp.kernel
def friction_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_f: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    mu = contact_friction_coeff[world_idx, contact_idx]
    force_n_prev = body_lambda_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        h_f[world_idx, constr_idx1] = 0.0
        h_f[world_idx, constr_idx2] = 0.0
        return

    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Basis
    J_t1_1 = contact_basis_t1_a[world_idx, contact_idx]
    J_t2_1 = contact_basis_t2_a[world_idx, contact_idx]
    J_t1_2 = contact_basis_t1_b[world_idx, contact_idx]
    J_t2_2 = contact_basis_t2_b[world_idx, contact_idx]

    q_1 = wp.transform()
    u_1 = wp.spatial_vector()
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        q_1 = body_q[world_idx, body_1]
        u_1 = body_u[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    q_2 = wp.transform()
    u_2 = wp.spatial_vector()
    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        q_2 = body_q[world_idx, body_2]
        u_2 = body_u[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    lambda_t1 = body_lambda_f[world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[world_idx, constr_idx2]
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]

    # --- Call Shared Logic ---
    (d_h1, d_h2, hf1, hf2, J11, J21, J12, J22, cf) = compute_friction_local(
        body_1,
        body_2,
        mu,
        J_t1_1,
        J_t2_1,
        J_t1_2,
        J_t2_2,
        q_1,
        q_2,
        u_1,
        u_2,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        lambda_t1,
        lambda_t2,
        lambda_t1_prev,
        lambda_t2_prev,
        force_n_prev,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, world_idx, body_1, d_h1)
    if body_2 >= 0:
        wp.atomic_add(h_d, world_idx, body_2, d_h2)

    h_f[world_idx, constr_idx1] = hf1
    h_f[world_idx, constr_idx2] = hf2


@wp.kernel
def batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_f: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()
    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    mu = contact_friction_coeff[world_idx, contact_idx]
    force_n_prev = body_lambda_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        h_f[batch_idx, world_idx, constr_idx1] = 0.0
        h_f[batch_idx, world_idx, constr_idx2] = 0.0
        return

    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Basis
    J_t1_1 = contact_basis_t1_a[world_idx, contact_idx]
    J_t2_1 = contact_basis_t2_a[world_idx, contact_idx]
    J_t1_2 = contact_basis_t1_b[world_idx, contact_idx]
    J_t2_2 = contact_basis_t2_b[world_idx, contact_idx]

    q_1 = wp.transform()
    u_1 = wp.spatial_vector()
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        q_1 = body_q[batch_idx, world_idx, body_1]
        u_1 = body_u[batch_idx, world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    q_2 = wp.transform()
    u_2 = wp.spatial_vector()
    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        q_2 = body_q[batch_idx, world_idx, body_2]
        u_2 = body_u[batch_idx, world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    lambda_t1 = body_lambda_f[batch_idx, world_idx, constr_idx1]
    lambda_t2 = body_lambda_f[batch_idx, world_idx, constr_idx2]
    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]

    # --- Call Shared Logic ---
    (d_h1, d_h2, hf1, hf2, J11, J21, J12, J22, cf) = compute_friction_local(
        body_1,
        body_2,
        mu,
        J_t1_1,
        J_t2_1,
        J_t1_2,
        J_t2_2,
        q_1,
        q_2,
        u_1,
        u_2,
        m_inv_1,
        I_inv_1,
        m_inv_2,
        I_inv_2,
        lambda_t1,
        lambda_t2,
        lambda_t1_prev,
        lambda_t2_prev,
        force_n_prev,
        dt,
    )

    if body_1 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_1, d_h1)
    if body_2 >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, body_2, d_h2)

    h_f[batch_idx, world_idx, constr_idx1] = hf1
    h_f[batch_idx, world_idx, constr_idx2] = hf2


@wp.kernel
def fused_batch_friction_residual_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_f: wp.array(dtype=wp.float32, ndim=3),
    body_lambda_f_prev: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_n_prev: wp.array(dtype=wp.float32, ndim=2),
    s_n_prev: wp.array(dtype=wp.float32, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    num_batches: int,
    # --- Outputs (contributions to the linear system) ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_f: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_friction_coeff.shape[1]:
        return

    constr_idx1 = 2 * contact_idx
    constr_idx2 = 2 * contact_idx + 1

    mu = contact_friction_coeff[world_idx, contact_idx]
    force_n_prev = body_lambda_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        for b in range(num_batches):
            h_f[b, world_idx, constr_idx1] = 0.0
            h_f[b, world_idx, constr_idx2] = 0.0
        return

    body_1 = contact_body_a[world_idx, contact_idx]
    body_2 = contact_body_b[world_idx, contact_idx]

    # Basis
    J_t1_1 = contact_basis_t1_a[world_idx, contact_idx]
    J_t2_1 = contact_basis_t2_a[world_idx, contact_idx]
    J_t1_2 = contact_basis_t1_b[world_idx, contact_idx]
    J_t2_2 = contact_basis_t2_b[world_idx, contact_idx]

    lambda_t1_prev = body_lambda_f_prev[world_idx, constr_idx1]
    lambda_t2_prev = body_lambda_f_prev[world_idx, constr_idx2]

    # Pre-load Spatial Inertia (Frozen)
    m_inv_1 = 0.0
    I_inv_1 = wp.mat33()
    if body_1 >= 0:
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_1 = body_I_inv[world_idx, body_1]

    m_inv_2 = 0.0
    I_inv_2 = wp.mat33()
    if body_2 >= 0:
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_2 = body_I_inv[world_idx, body_2]

    # --- 3. Iterate over Batches ---
    for b in range(num_batches):
        q_1 = wp.transform()
        u_1 = wp.spatial_vector()
        if body_1 >= 0:
            q_1 = body_q[b, world_idx, body_1]
            u_1 = body_u[b, world_idx, body_1]

        q_2 = wp.transform()
        u_2 = wp.spatial_vector()
        if body_2 >= 0:
            q_2 = body_q[b, world_idx, body_2]
            u_2 = body_u[b, world_idx, body_2]

        lambda_t1 = body_lambda_f[b, world_idx, constr_idx1]
        lambda_t2 = body_lambda_f[b, world_idx, constr_idx2]

        # --- Call Shared Logic ---
        (d_h1, d_h2, hf1, hf2, J11, J21, J12, J22, cf) = compute_friction_local(
            body_1,
            body_2,
            mu,
            J_t1_1,
            J_t2_1,
            J_t1_2,
            J_t2_2,
            q_1,
            q_2,
            u_1,
            u_2,
            m_inv_1,
            I_inv_1,
            m_inv_2,
            I_inv_2,
            lambda_t1,
            lambda_t2,
            lambda_t1_prev,
            lambda_t2_prev,
            force_n_prev,
            dt,
        )

        if body_1 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_1, d_h1)
        if body_2 >= 0:
            wp.atomic_add(h_d, b, world_idx, body_2, d_h2)

        h_f[b, world_idx, constr_idx1] = hf1
        h_f[b, world_idx, constr_idx2] = hf2
