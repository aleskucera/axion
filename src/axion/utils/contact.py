import warp as wp
from axion.utils import orthogonal_basis
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.func
def compute_restitution_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
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
