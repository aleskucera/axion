import warp as wp
from axion.math import orthogonal_basis


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
    # --- Outputs ---
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    shape_a = contact_shape0[world_idx, contact_idx]
    shape_b = contact_shape1[world_idx, contact_idx]

    if contact_idx >= contact_count[world_idx] or shape_a == shape_b:
        contact_dist[world_idx, contact_idx] = -1.0
        return

    # Contact body indices (default to -1)
    body_a = -1
    body_b = -1

    # Get body indices
    if shape_a >= 0:
        body_a = shape_body[world_idx, shape_a]
    if shape_b >= 0:
        body_b = shape_body[world_idx, shape_b]

    # Contact normal in world space
    n = contact_normal[world_idx, contact_idx]

    # Contact points from Newton are in Body-Local space
    p_a_local = contact_point0[world_idx, contact_idx]
    p_b_local = contact_point1[world_idx, contact_idx]

    # Transforms
    X_a = wp.transform_identity()
    if body_a >= 0:
        X_a = body_q[world_idx, body_a]

    X_b = wp.transform_identity()
    if body_b >= 0:
        X_b = body_q[world_idx, body_b]

    # World points
    p_a_world = wp.transform_point(X_a, p_a_local)
    p_b_world = wp.transform_point(X_b, p_b_local)

    # Thickness adjustment in world space
    offset_a = -contact_thickness0[world_idx, contact_idx] * n
    offset_b = contact_thickness1[world_idx, contact_idx] * n

    p_a_world_adj = p_a_world + offset_a
    p_b_world_adj = p_b_world + offset_b

    # Contact residuals (lever arms) in World space
    r_a = wp.vec3()
    if body_a >= 0:
        com_a_world = wp.transform_point(X_a, body_com[world_idx, body_a])
        r_a = p_a_world_adj - com_a_world

    r_b = wp.vec3()
    if body_b >= 0:
        com_b_world = wp.transform_point(X_b, body_com[world_idx, body_b])
        r_b = p_b_world_adj - com_b_world

    # Compute contact Jacobians
    t1, t2 = orthogonal_basis(n)

    # Fill the output arrays
    contact_body_a[world_idx, contact_idx] = body_a
    contact_body_b[world_idx, contact_idx] = body_b

    contact_point_a[world_idx, contact_idx] = p_a_local
    contact_point_b[world_idx, contact_idx] = p_b_local

    contact_thickness_a[world_idx, contact_idx] = contact_thickness0[world_idx, contact_idx]
    contact_thickness_b[world_idx, contact_idx] = contact_thickness1[world_idx, contact_idx]

    # Penetration depth (positive for penetration)
    contact_dist[world_idx, contact_idx] = wp.dot(n, p_b_world_adj - p_a_world_adj)

    contact_basis_n_a[world_idx, contact_idx] = wp.spatial_vector(n, wp.cross(r_a, n))
    contact_basis_t1_a[world_idx, contact_idx] = wp.spatial_vector(t1, wp.cross(r_a, t1))
    contact_basis_t2_a[world_idx, contact_idx] = wp.spatial_vector(t2, wp.cross(r_a, t2))

    contact_basis_n_b[world_idx, contact_idx] = wp.spatial_vector(-n, -wp.cross(r_b, n))
    contact_basis_t1_b[world_idx, contact_idx] = wp.spatial_vector(-t1, -wp.cross(r_b, t1))
    contact_basis_t2_b[world_idx, contact_idx] = wp.spatial_vector(-t2, -wp.cross(r_b, t2))

    mu_a = shape_material_mu[world_idx, shape_a]
    mu_b = shape_material_mu[world_idx, shape_b]
    contact_friction_coeff[world_idx, contact_idx] = (mu_a + mu_b) * 0.5

    e_a = shape_material_restitution[world_idx, shape_a]
    e_b = shape_material_restitution[world_idx, shape_b]
    contact_restitution_coeff[world_idx, contact_idx] = (e_a + e_b) * 0.5
