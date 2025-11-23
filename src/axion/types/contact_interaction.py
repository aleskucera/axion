import warp as wp

from .utils import orthogonal_basis


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


# @wp.kernel
# def contact_interaction_kernel(
#     # --- Inputs ---
#     body_q: wp.array(dtype=wp.transform),
#     body_com: wp.array(dtype=wp.vec3),
#     shape_body: wp.array(dtype=wp.int32),
#     shape_thickness: wp.array(dtype=wp.float32),
#     shape_material_mu: wp.array(dtype=wp.float32),
#     shape_material_restitution: wp.array(dtype=wp.float32),
#     contact_count: wp.array(dtype=wp.int32),
#     contact_point0: wp.array(dtype=wp.vec3),
#     contact_point1: wp.array(dtype=wp.vec3),
#     contact_normal: wp.array(dtype=wp.vec3),
#     contact_shape0: wp.array(dtype=wp.int32),
#     contact_shape1: wp.array(dtype=wp.int32),
#     contact_thickness0: wp.array(dtype=wp.float32),
#     contact_thickness1: wp.array(dtype=wp.float32),
#     # --- Single Output ---
#     interactions: wp.array(dtype=ContactInteraction),
# ):
#     tid = wp.tid()
#
#     interaction = ContactInteraction()
#     shape_a = contact_shape0[tid]
#     shape_b = contact_shape1[tid]
#
#     if tid >= contact_count[0] or shape_a == shape_b:
#         interaction.is_active = False
#         interactions[tid] = interaction
#         return
#
#     # Contact body indices (default to -1)
#     body_a = -1
#     body_b = -1
#
#     # Get body indices and thickness
#     if shape_a >= 0:
#         body_a = shape_body[shape_a]
#     if shape_b >= 0:
#         body_b = shape_body[shape_b]
#
#     # Contact normal in world space
#     n = contact_normal[tid]
#
#     # Contact points in world space
#     p_a = contact_point0[tid]
#     p_b = contact_point1[tid]
#
#     # Contact residual (contact_point - body_com)
#     r_a = wp.vec3()
#     r_b = wp.vec3()
#
#     # Get world-space contact points and lever arms (r_a, r_b)
#     if body_a >= 0:
#         X_wb_a = body_q[body_a]
#         offset_a = -contact_thickness0[tid] * n
#         p_a = wp.transform_point(X_wb_a, contact_point0[tid]) + offset_a
#         r_a = p_a - wp.transform_point(X_wb_a, body_com[body_a])
#     if body_b >= 0:
#         X_wb_b = body_q[body_b]
#         offset_b = contact_thickness1[tid] * n
#         p_b = wp.transform_point(X_wb_b, contact_point1[tid]) + offset_b
#         r_b = p_b - wp.transform_point(X_wb_b, body_com[body_b])
#
#     # Compute contact Jacobians
#     t1, t2 = orthogonal_basis(n)
#
#     # Fill the manifold struct
#     interaction.is_active = True
#     interaction.body_a_idx = body_a
#     interaction.body_b_idx = body_b
#
#     # interaction.penetration_depth = d
#     interaction.penetration_depth = wp.dot(n, p_b - p_a)
#
#     if interaction.penetration_depth <= 0:
#         interaction.is_active = False
#
#     interaction.basis_a.normal = wp.spatial_vector(n, wp.cross(r_a, n))
#     interaction.basis_a.tangent1 = wp.spatial_vector(t1, wp.cross(r_a, t1))
#     interaction.basis_a.tangent2 = wp.spatial_vector(t2, wp.cross(r_a, t2))
#
#     interaction.basis_b.normal = wp.spatial_vector(-n, -wp.cross(r_b, n))
#     interaction.basis_b.tangent1 = wp.spatial_vector(-t1, -wp.cross(r_b, t1))
#     interaction.basis_b.tangent2 = wp.spatial_vector(-t2, -wp.cross(r_b, t2))
#
#     mu_a = shape_material_mu[shape_a]
#     mu_b = shape_material_mu[shape_b]
#     interaction.friction_coeff = (mu_a + mu_b) * 0.5
#
#     e_a = shape_material_restitution[shape_a]
#     e_b = shape_material_restitution[shape_b]
#     interaction.restitution_coeff = (e_a + e_b) * 0.5
#
#     # Write the complete struct to the output array
#     interactions[tid] = interaction
#


@wp.kernel
def contact_interaction_kernel(
    # --- Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_thickness: wp.array(dtype=wp.float32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_material_restitution: wp.array(dtype=wp.float32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    # --- Single Output ---
    interactions: wp.array(dtype=ContactInteraction, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    interaction = ContactInteraction()
    shape_a = contact_shape0[world_idx, contact_idx]
    shape_b = contact_shape1[world_idx, contact_idx]

    if contact_idx >= contact_count[world_idx] or shape_a == shape_b:
        interaction.is_active = False
        interactions[world_idx, contact_idx] = interaction
        return

    # Contact body indices (default to -1)
    body_a = -1
    body_b = -1

    # Get body indices and thickness
    if shape_a >= 0:
        body_a = shape_body[world_idx, shape_a]
    if shape_b >= 0:
        body_b = shape_body[world_idx, shape_b]

    # Contact normal in world space
    n = contact_normal[world_idx, contact_idx]

    # Contact points in world space
    p_a = contact_point0[world_idx, contact_idx]
    p_b = contact_point1[world_idx, contact_idx]

    # Contact residual (contact_point - body_com)
    r_a = wp.vec3()
    r_b = wp.vec3()

    # Get world-space contact points and lever arms (r_a, r_b)
    if body_a >= 0:
        X_wb_a = body_q[world_idx, body_a]
        offset_a = -contact_thickness0[world_idx, contact_idx] * n
        p_a = wp.transform_point(X_wb_a, contact_point0[world_idx, contact_idx]) + offset_a
        r_a = p_a - wp.transform_point(X_wb_a, body_com[world_idx, body_a])
    if body_b >= 0:
        X_wb_b = body_q[world_idx, body_b]
        offset_b = contact_thickness1[world_idx, contact_idx] * n
        p_b = wp.transform_point(X_wb_b, contact_point1[world_idx, contact_idx]) + offset_b
        r_b = p_b - wp.transform_point(X_wb_b, body_com[world_idx, body_b])

    # Compute contact Jacobians
    t1, t2 = orthogonal_basis(n)

    # Fill the manifold struct
    interaction.is_active = True
    interaction.body_a_idx = body_a
    interaction.body_b_idx = body_b

    # interaction.penetration_depth = d
    interaction.penetration_depth = wp.dot(n, p_b - p_a)

    if interaction.penetration_depth <= 0:
        interaction.is_active = False

    interaction.basis_a.normal = wp.spatial_vector(n, wp.cross(r_a, n))
    interaction.basis_a.tangent1 = wp.spatial_vector(t1, wp.cross(r_a, t1))
    interaction.basis_a.tangent2 = wp.spatial_vector(t2, wp.cross(r_a, t2))

    interaction.basis_b.normal = wp.spatial_vector(-n, -wp.cross(r_b, n))
    interaction.basis_b.tangent1 = wp.spatial_vector(-t1, -wp.cross(r_b, t1))
    interaction.basis_b.tangent2 = wp.spatial_vector(-t2, -wp.cross(r_b, t2))

    mu_a = shape_material_mu[world_idx, shape_a]
    mu_b = shape_material_mu[world_idx, shape_b]
    interaction.friction_coeff = (mu_a + mu_b) * 0.5

    e_a = shape_material_restitution[world_idx, shape_a]
    e_b = shape_material_restitution[world_idx, shape_b]
    interaction.restitution_coeff = (e_a + e_b) * 0.5

    # Write the complete struct to the output array
    interactions[world_idx, contact_idx] = interaction
