import warp as wp
from axion.math import orthogonal_basis
from axion.math import scaled_fisher_burmeister

from .utils import compute_effective_mass

# -----------------------------------------------------------------------------
# 1. Low-Level Helpers
# -----------------------------------------------------------------------------


@wp.func
def resolve_body_indices(
    world_idx: int,
    shape0: int,
    shape1: int,
    shape_body: wp.array(dtype=wp.int32, ndim=2),
):
    """Resolves shape indices to body indices. Returns -1 for static shapes."""
    body0 = -1
    if shape0 >= 0:
        body0 = shape_body[world_idx, shape0]

    body1 = -1
    if shape1 >= 0:
        body1 = shape_body[world_idx, shape1]

    return body0, body1


@wp.func
def compute_friction_model(
    mu: wp.float32,
    J_t1_0: wp.spatial_vector,
    J_t2_0: wp.spatial_vector,
    J_t1_1: wp.spatial_vector,
    J_t2_1: wp.spatial_vector,
    vel0: wp.spatial_vector,
    vel1: wp.spatial_vector,
    force_f_prev: wp.vec2,
    force_n_prev: wp.float32,
    dt: wp.float32,
    precond: wp.float32,
):
    """Algebraic Fisher-Burmeister formulation for friction."""
    v_t_0 = wp.dot(J_t1_0, vel0) + wp.dot(J_t1_1, vel1)
    v_t_1 = wp.dot(J_t2_0, vel0) + wp.dot(J_t2_1, vel1)
    v_t = wp.vec2(v_t_0, v_t_1)

    eps = 1e-8
    v_t_norm = wp.sqrt(wp.dot(v_t, v_t) + eps)

    # 1. Capture Raw Norm (Represents the actual scale of forces in the system)
    raw_f_norm = wp.length(force_f_prev)

    # 2. Compute Clamped Norm (Represents the physical limit)
    limit = mu * force_n_prev
    clamped_f_norm = wp.min(raw_f_norm, limit)

    # 3. GAP Calculation: ALWAYS use Clamped
    gap = limit - clamped_f_norm

    r = precond
    phi_f = scaled_fisher_burmeister(v_t_norm, gap, 1.0, r)

    denom_eps = 1e-6

    # 4. DENOMINATOR Calculation: Use RAW Norm (The Fix)
    denominator = r * raw_f_norm + phi_f + denom_eps
    numerator = v_t_norm - phi_f

    w = r * (numerator / denominator)
    w = wp.max(w, 0.0)
    w = wp.min(w, 1e5)

    return v_t, w


# -----------------------------------------------------------------------------
# 2. The SINGLE Core Logic Function
# -----------------------------------------------------------------------------


@wp.func
def compute_friction_core(
    body0: int,
    body1: int,
    n: wp.vec3,
    t1: wp.vec3,
    t2: wp.vec3,
    mu: float,
    p0_local: wp.vec3,
    p1_local: wp.vec3,
    thickness0: float,
    thickness1: float,
    vel0: wp.spatial_vector,
    pose0_prev: wp.transform,
    m_inv0: float,
    I_inv0: wp.mat33,
    com0: wp.vec3,
    vel1: wp.spatial_vector,
    pose1_prev: wp.transform,
    m_inv1: float,
    I_inv1: wp.mat33,
    com1: wp.vec3,
    lambda_t1: float,
    lambda_t2: float,
    lambda_t1_prev: float,
    lambda_t2_prev: float,
    force_n_prev: float,
    dt: float,
):
    """
    Computes all Jacobians and friction residuals dynamically.
    """
    J_t1_0 = wp.spatial_vector()
    J_t2_0 = wp.spatial_vector()

    if body0 >= 0:
        p0_world_prev = wp.transform_point(pose0_prev, p0_local)
        p0_adj_prev = p0_world_prev - (thickness0 * n)
        com0_world_prev = wp.transform_point(pose0_prev, com0)
        r0 = p0_adj_prev - com0_world_prev
        J_t1_0 = wp.spatial_vector(t1, wp.cross(r0, t1))
        J_t2_0 = wp.spatial_vector(t2, wp.cross(r0, t2))

    J_t1_1 = wp.spatial_vector()
    J_t2_1 = wp.spatial_vector()

    if body1 >= 0:
        p1_world_prev = wp.transform_point(pose1_prev, p1_local)
        p1_adj_prev = p1_world_prev + (thickness1 * n)
        com1_world_prev = wp.transform_point(pose1_prev, com1)
        r1 = p1_adj_prev - com1_world_prev
        J_t1_1 = wp.spatial_vector(-t1, -wp.cross(r1, t1))
        J_t2_1 = wp.spatial_vector(-t2, -wp.cross(r1, t2))

    w_t1 = compute_effective_mass(
        pose0_prev, pose1_prev, J_t1_0, J_t1_1, m_inv0, I_inv0, m_inv1, I_inv1, body0, body1
    )
    w_t2 = compute_effective_mass(
        pose0_prev, pose1_prev, J_t2_0, J_t2_1, m_inv0, I_inv0, m_inv1, I_inv1, body0, body1
    )

    effective_mass = (w_t1 + w_t2) * 0.5
    precond = dt * effective_mass
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    v_t, w = compute_friction_model(
        mu, J_t1_0, J_t2_0, J_t1_1, J_t2_1, vel0, vel1, force_f_prev, force_n_prev, dt, precond
    )

    d_res_d0 = -dt * (J_t1_0 * lambda_t1 + J_t2_0 * lambda_t2)
    d_res_d1 = -dt * (J_t1_1 * lambda_t1 + J_t2_1 * lambda_t2)
    res_f0 = v_t.x + w * lambda_t1
    res_f1 = v_t.y + w * lambda_t2
    c_f = w / dt

    return d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f


# -----------------------------------------------------------------------------
# 3. Standard Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def friction_residual_kernel(
    # State variables
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    # Outputs
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_f: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    mu = (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1]) * 0.5
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    # EARLY EXIT: Save memory bandwidth
    if mu * force_n_prev <= 1e-6:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    t1, t2 = orthogonal_basis(n)

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    lam_t1 = constr_force[world_idx, constr_idx0]
    lam_t2 = constr_force[world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_f[world_idx, constr_idx0] = res_f0
    res_f[world_idx, constr_idx1] = res_f1


@wp.kernel
def friction_constraint_kernel(
    # State variables
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    # Outputs
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_f: wp.array(dtype=wp.float32, ndim=2),
    J_hat_f_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_f_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    mu = (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1]) * 0.5
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    constr_active_mask[world_idx, constr_idx0] = 1.0
    constr_active_mask[world_idx, constr_idx1] = 1.0

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)

    constr_body_idx[world_idx, constr_idx0, 0] = body0
    constr_body_idx[world_idx, constr_idx0, 1] = body1
    constr_body_idx[world_idx, constr_idx1, 0] = body0
    constr_body_idx[world_idx, constr_idx1, 1] = body1

    n = contact_normal[world_idx, contact_idx]
    t1, t2 = orthogonal_basis(n)

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    lam_t1 = constr_force[world_idx, constr_idx0]
    lam_t2 = constr_force[world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_f[world_idx, constr_idx0] = res_f0
    res_f[world_idx, constr_idx1] = res_f1

    J_hat_f_values[world_idx, constr_idx0, 0] = J_t1_0
    J_hat_f_values[world_idx, constr_idx1, 0] = J_t2_0
    J_hat_f_values[world_idx, constr_idx0, 1] = J_t1_1
    J_hat_f_values[world_idx, constr_idx1, 1] = J_t2_1

    C_f_values[world_idx, constr_idx0] = c_f
    C_f_values[world_idx, constr_idx1] = c_f


# -----------------------------------------------------------------------------
# 4. Batched Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def batch_friction_residual_kernel(
    # State variables (3D)
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),  # Prev step remains 2D
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_f: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    mu = (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1]) * 0.5
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    t1, t2 = orthogonal_basis(n)

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[batch_idx, world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[batch_idx, world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    lam_t1 = constr_force[batch_idx, world_idx, constr_idx0]
    lam_t2 = constr_force[batch_idx, world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body1, d_res_d1)

    res_f[batch_idx, world_idx, constr_idx0] = res_f0
    res_f[batch_idx, world_idx, constr_idx1] = res_f1


@wp.kernel
def fused_batch_friction_residual_kernel(
    # State variables (3D)
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    num_batches: int,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_f: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    # Load shared contact and mass parameters exactly ONCE
    mu = (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1]) * 0.5
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    if mu * force_n_prev <= 1e-6:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    t1, t2 = orthogonal_basis(n)

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    m_inv0, I_inv0, com0 = 0.0, wp.mat33(0.0), wp.vec3()
    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]

    m_inv1, I_inv1, com1 = 0.0, wp.mat33(0.0), wp.vec3()
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]

    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    # --- Iterate through batches utilizing the preloaded memory ---
    for b in range(num_batches):
        vel0 = wp.spatial_vector()
        if body0 >= 0:
            vel0 = body_vel[b, world_idx, body0]

        vel1 = wp.spatial_vector()
        if body1 >= 0:
            vel1 = body_vel[b, world_idx, body1]

        lam_t1 = constr_force[b, world_idx, constr_idx0]
        lam_t2 = constr_force[b, world_idx, constr_idx1]

        d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f = (
            compute_friction_core(
                body0,
                body1,
                n,
                t1,
                t2,
                mu,
                p0,
                p1,
                thickness0,
                thickness1,
                vel0,
                pose0_prev,
                m_inv0,
                I_inv0,
                com0,
                vel1,
                pose1_prev,
                m_inv1,
                I_inv1,
                com1,
                lam_t1,
                lam_t2,
                lam_t1_p,
                lam_t2_p,
                force_n_prev,
                dt,
            )
        )

        if body0 >= 0:
            wp.atomic_add(res_d, b, world_idx, body0, d_res_d0)
        if body1 >= 0:
            wp.atomic_add(res_d, b, world_idx, body1, d_res_d1)

        res_f[b, world_idx, constr_idx0] = res_f0
        res_f[b, world_idx, constr_idx1] = res_f1
