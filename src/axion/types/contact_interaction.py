import warp as wp
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.struct
class ContactBasis:
    """The spatial vector basis (n, t1, t2) for a single body in a contact constraint."""

    normal: wp.spatial_vector
    tangent1: wp.spatial_vector
    tangent2: wp.spatial_vector


@wp.struct
class ContactInteraction:
    """A complete description of a single contact constraint for the solver."""

    is_active: wp.bool

    # Indices of the two interacting bodies (-1 for static objects)
    body_a_idx: wp.int32
    body_b_idx: wp.int32

    # Per-body constraint data
    basis_a: ContactBasis
    basis_b: ContactBasis

    # Shared geometric and material properties
    penetration_depth: wp.float32
    restitution_coeff: wp.float32
    friction_coeff: wp.float32


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
def contact_interaction_kernel(
    # --- Inputs ---
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_geo: ModelShapeGeometry,
    shape_materials: ModelShapeMaterials,
    contact_count: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    # --- Single Output ---
    interactions: wp.array(dtype=ContactInteraction),
):
    tid = wp.tid()

    interaction = ContactInteraction()
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if tid >= contact_count[0] or shape_a == shape_b:
        interaction.is_active = False
        interactions[tid] = interaction
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

    # Compute contact Jacobians
    t1, t2 = orthogonal_basis(n)

    # Fill the manifold struct
    interaction.is_active = True
    interaction.body_a_idx = body_a
    interaction.body_b_idx = body_b

    interaction.penetration_depth = wp.dot(n, p_b - p_a)

    if interaction.penetration_depth <= 0:
        interaction.is_active = False

    interaction.basis_a.normal = wp.spatial_vector(wp.cross(r_a, n), n)
    interaction.basis_a.tangent1 = wp.spatial_vector(wp.cross(r_a, t1), t1)
    interaction.basis_a.tangent2 = wp.spatial_vector(wp.cross(r_a, t2), t2)

    interaction.basis_b.normal = wp.spatial_vector(-wp.cross(r_b, n), -n)
    interaction.basis_b.tangent1 = wp.spatial_vector(-wp.cross(r_b, t1), -t1)
    interaction.basis_b.tangent2 = wp.spatial_vector(-wp.cross(r_b, t2), -t2)

    interaction.restitution_coeff = compute_restitution_coefficient(
        shape_a, shape_b, shape_materials
    )
    interaction.friction_coeff = compute_friction_coefficient(
        shape_a, shape_b, shape_materials
    )

    # Write the complete struct to the output array
    interactions[tid] = interaction
