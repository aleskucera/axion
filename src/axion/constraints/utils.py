import warp as wp
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials

MAX_BODIES = 10
RES_BUFFER_DIM = MAX_BODIES * 6 + 50


@wp.func
def get_constraint_body_index(
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: int,
    J_n_offset: int,
    J_f_offset: int,
    constraint_idx: int,
):
    """Maps a constraint index to the two body indices it affects."""
    body_a = -1
    body_b = -1
    if constraint_idx < J_n_offset:
        joint_index = (constraint_idx - J_j_offset) // 5
        body_a = joint_parent[joint_index]
        body_b = joint_child[joint_index]
    elif constraint_idx < J_f_offset:
        contact_index = constraint_idx - J_n_offset
        body_a = contact_body_a[contact_index]
        body_b = contact_body_b[contact_index]
    else:
        contact_index = (constraint_idx - J_f_offset) // 2
        body_a = contact_body_a[contact_index]
        body_b = contact_body_b[contact_index]
    return body_a, body_b


@wp.func
def orthogonal_basis(axis: wp.vec3):
    # Choose v as the unit vector along the axis with the smallest absolute component
    if wp.abs(axis.x) <= wp.abs(axis.y) and wp.abs(axis.x) <= wp.abs(axis.z):
        v = wp.vec3(1.0, 0.0, 0.0)
    elif wp.abs(axis.y) <= wp.abs(axis.z):
        v = wp.vec3(0.0, 1.0, 0.0)
    else:
        v = wp.vec3(0.0, 0.0, 1.0)

    # Compute b1 as the normalized cross product of axis and v
    b1 = wp.normalize(wp.cross(axis, v))

    # Compute b2 as the cross product of axis and b1
    b2 = wp.cross(axis, b1)

    return b1, b2


@wp.func
def compute_restitution_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
    """Computes the average coefficient of restitution for a contact pair."""
    e = 0.0
    if shape_a >= 0 and shape_b >= 0:
        e_a = shape_materials.restitution[shape_a]
        e_b = shape_materials.restitution[shape_b]
        e = (e_a + e_b) * 0.5
    elif shape_a >= 0:
        e = shape_materials.restitution[shape_a]
    elif shape_b >= 0:
        e = shape_materials.restitution[shape_b]
    return e


@wp.func
def compute_friction_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
    mu = 0.0
    if shape_a >= 0 and shape_b >= 0:
        mu_a = shape_materials.mu[shape_a]
        mu_b = shape_materials.mu[shape_b]
        mu = (mu_a + mu_b) * 0.5
    elif shape_a >= 0:
        mu = shape_materials.mu[shape_a]
    elif shape_b >= 0:
        mu = shape_materials.mu[shape_b]
    return mu


@wp.kernel
def contact_kinematics_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_geo: ModelShapeGeometry,
    shape_materials: ModelShapeMaterials,
    contact_count: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    # --- Outputs ---
    gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_restitution_coeff: wp.array(dtype=wp.float32),
    contact_friction_coeff: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if tid >= contact_count[0] or shape_a == shape_b:
        gap[tid] = 0.0
        J_contact_a[tid, 0] = wp.spatial_vector()
        J_contact_a[tid, 1] = wp.spatial_vector()
        J_contact_a[tid, 2] = wp.spatial_vector()
        J_contact_b[tid, 0] = wp.spatial_vector()
        J_contact_b[tid, 1] = wp.spatial_vector()
        J_contact_b[tid, 2] = wp.spatial_vector()
        contact_body_a[tid] = -1
        contact_body_b[tid] = -1
        contact_restitution_coeff[tid] = 0.0
        contact_friction_coeff[tid] = 0.0
        return

    # Contact body indices (default to -1)
    body_a = -1
    body_b = -1

    # Contact thickness (default to 0.0)
    thickness_a = 0.0
    thickness_b = 0.0

    # Get body indices and thickness
    if shape_a >= 0:
        body_a = shape_body[shape_a]
        thickness_a = shape_geo.thickness[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
        thickness_b = shape_geo.thickness[shape_b]

    # Contact normal in world space
    n = contact_normal[tid]

    # Contact points in world space
    p_a = contact_point0[tid]
    p_b = contact_point1[tid]

    # Contact residual (contact_point - body_com)
    r_a = wp.vec3()
    r_b = wp.vec3()

    # Get world-space contact points and lever arms (r_a, r_b)
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        offset_a = -thickness_a * n
        p_a = wp.transform_point(X_wb_a, contact_point0[tid]) + offset_a
        r_a = p_a - wp.transform_point(X_wb_a, body_com[body_a])
    if body_b >= 0:
        X_wb_b = body_q[body_b]
        offset_b = thickness_b * n
        p_b = wp.transform_point(X_wb_b, contact_point1[tid]) + offset_b
        r_b = p_b - wp.transform_point(X_wb_b, body_com[body_b])

    # Compute penetration depth
    gap[tid] = wp.dot(n, p_a - p_b)

    # Compute contact Jacobians
    t, b = orthogonal_basis(n)

    J_contact_a[tid, 0] = wp.spatial_vector(wp.cross(r_a, n), n)
    J_contact_a[tid, 1] = wp.spatial_vector(wp.cross(r_a, t), t)
    J_contact_a[tid, 2] = wp.spatial_vector(wp.cross(r_a, b), b)

    J_contact_b[tid, 0] = wp.spatial_vector(-wp.cross(r_b, n), -n)
    J_contact_b[tid, 1] = wp.spatial_vector(-wp.cross(r_b, t), -t)
    J_contact_b[tid, 2] = wp.spatial_vector(-wp.cross(r_b, b), -b)

    contact_body_a[tid] = body_a
    contact_body_b[tid] = body_b

    # Compute coefficients of restitution and friction
    contact_restitution_coeff[tid] = compute_restitution_coefficient(
        shape_a, shape_b, shape_materials
    )
    contact_friction_coeff[tid] = compute_friction_coefficient(
        shape_a, shape_b, shape_materials
    )


@wp.func
def scaled_fisher_burmeister(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
):
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    value = scaled_a + scaled_b - norm

    # Avoid division by zero
    if norm < 1e-6:
        return value, 0.0, 1.0

    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db


@wp.func
def get_random_idx_to_res_buffer(state: wp.int32):
    low = wp.static(wp.uint32(MAX_BODIES * 6))
    high = wp.static(wp.uint32(RES_BUFFER_DIM))
    idx = wp.randu(wp.uint32(state), low, high)
    return idx
