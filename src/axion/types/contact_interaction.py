import warp as wp
from axion.math import orthogonal_basis


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

    # Contact point vectors relative to body frames
    contact_point_a: wp.vec3
    contact_point_b: wp.vec3

    # ContactThickness
    contact_thickness_a: wp.float32
    contact_thickness_b: wp.float32

    # Shared geometric and material properties
    penetration_depth: wp.float32
    restitution_coeff: wp.float32
    friction_coeff: wp.float32


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

    # Fill the manifold struct
    interaction.is_active = True
    interaction.body_a_idx = body_a
    interaction.body_b_idx = body_b

    interaction.contact_point_a = p_a_local
    interaction.contact_point_b = p_b_local

    interaction.contact_thickness_a = contact_thickness0[world_idx, contact_idx]
    interaction.contact_thickness_b = contact_thickness1[world_idx, contact_idx]

    # Penetration depth (positive for penetration)
    interaction.penetration_depth = wp.dot(n, p_b_world_adj - p_a_world_adj)

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

@wp.kernel
def contact_penetration_depth_kernel(
    # --- Inputs (reordered 2D arrays: body always in position 0, ground in position 1) ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),  # Always body (reordered)
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),  # Always ground (reordered)
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),  # From body to ground (reordered)
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),  # Always body (reordered)
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),  # Always ground (reordered)
    body_shape: wp.array(dtype=wp.int32, ndim=2),  # Shape index of body for each contact
    # --- Single Output ---
    depths: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Compute penetration depth for ground-pendulum contacts.
    Assumes reordered contact data where:
    - position 0 is always the body (body >= 0)
    - position 1 is always the ground (body == -1)
    """
    world_idx, contact_idx = wp.tid()

    # contact_count is shape (1,) from Newton, use [0]
    if contact_idx >= contact_count[0]:
        depths[world_idx, contact_idx] = -1.0  # Negative indicates inactive/invalid contact
        return

    # Get body index from body shape
    body_shape_idx = body_shape[world_idx, contact_idx]
    if body_shape_idx < 0:
        depths[world_idx, contact_idx] = -1.0  # Invalid body shape
        return

    body_idx = shape_body[world_idx, body_shape_idx]
    if body_idx < 0:
        depths[world_idx, contact_idx] = -1.0  # Invalid body
        return

    # Contact normal (already reordered to point from body to ground)
    n = contact_normal[world_idx, contact_idx]

    # Contact points (already reordered: 0=body, 1=ground)
    p_body_local = contact_point0[world_idx, contact_idx]  # Body point in body-local space
    p_ground_local = contact_point1[world_idx, contact_idx]  # Ground point (in ground-local space, which is world space)

    # Transform body point to world space
    X_body = body_q[world_idx, body_idx]
    p_body_world = wp.transform_point(X_body, p_body_local)

    # Ground point is already in world space (ground has identity transform)
    p_ground_world = p_ground_local

    # Thickness adjustment in world space (already reordered)
    offset_body = -contact_thickness0[world_idx, contact_idx] * n  # Move body point outward
    offset_ground = contact_thickness1[world_idx, contact_idx] * n  # Move ground point outward

    p_body_world_adj = p_body_world + offset_body
    p_ground_world_adj = p_ground_world + offset_ground

    # Penetration depth (positive for penetration)
    # Normal points from body to ground, so depth = dot(n, ground - body)
    depths[world_idx, contact_idx] = wp.dot(n, p_ground_world_adj - p_body_world_adj)


@wp.kernel
def reorder_ground_contacts_kernel(
    # --- Inputs from Newton (1D arrays) ---
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=1),
    contact_shape1: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=1),
    contact_point1: wp.array(dtype=wp.vec3, ndim=1),
    contact_normal: wp.array(dtype=wp.vec3, ndim=1),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=1),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=1),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    # --- Outputs (reordered, 2D arrays for neural network) ---
    reordered_point0: wp.array(dtype=wp.vec3, ndim=2),  # Always body
    reordered_point1: wp.array(dtype=wp.vec3, ndim=2),  # Always ground
    reordered_normal: wp.array(dtype=wp.vec3, ndim=2),
    reordered_thickness0: wp.array(dtype=wp.float32, ndim=2),  # Always body
    reordered_thickness1: wp.array(dtype=wp.float32, ndim=2),  # Always ground
    reordered_body_shape: wp.array(dtype=wp.int32, ndim=2),  # Shape index of body (for getting body_q)
):
    """
    Reorder contact data so that:
    - shape0/point0/thickness0 always belong to the body (body >= 0)
    - shape1/point1/thickness1 always belong to the ground (body == -1)
    - Normal is flipped if needed to point from body to ground
    """
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[0]:
        return

    shape_a = contact_shape0[contact_idx]
    shape_b = contact_shape1[contact_idx]

    # Convert global shape indices to per-world indices (for multi-world support)
    # For single world, this is a no-op, but keeps the code correct
    shapes_per_world = shape_body.shape[1]
    shape_a_per_world = shape_a % shapes_per_world if shape_a >= 0 else shape_a
    shape_b_per_world = shape_b % shapes_per_world if shape_b >= 0 else shape_b

    # Get body indices using per-world shape indices
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[world_idx, shape_a_per_world]
    if shape_b >= 0:
        body_b = shape_body[world_idx, shape_b_per_world]

    # Check if this is a ground contact (exactly one body is -1)
    is_ground_contact = (body_a == -1 and body_b >= 0) or (body_a >= 0 and body_b == -1)
    if not is_ground_contact:
        # Not a ground contact, skip
        return

    # Determine if we need to swap (if body_a is ground, swap everything)
    need_swap = (body_a == -1)

    if need_swap:
        # Swap: body is in position 1, ground is in position 0
        # So we put body data in output position 0, ground data in output position 1
        reordered_point0[world_idx, contact_idx] = contact_point1[contact_idx]  # Body point
        reordered_point1[world_idx, contact_idx] = contact_point0[contact_idx]  # Ground point
        reordered_normal[world_idx, contact_idx] = -contact_normal[contact_idx]  # Flip normal (from body to ground)
        reordered_thickness0[world_idx, contact_idx] = contact_thickness1[contact_idx]  # Body thickness
        reordered_thickness1[world_idx, contact_idx] = contact_thickness0[contact_idx]  # Ground thickness
        reordered_body_shape[world_idx, contact_idx] = shape_b_per_world  # Body shape index (per-world)
    else:
        # Already correct: body is in position 0, ground is in position 1
        reordered_point0[world_idx, contact_idx] = contact_point0[contact_idx]  # Body point
        reordered_point1[world_idx, contact_idx] = contact_point1[contact_idx]  # Ground point
        reordered_normal[world_idx, contact_idx] = contact_normal[contact_idx]  # Normal already correct
        reordered_thickness0[world_idx, contact_idx] = contact_thickness0[contact_idx]  # Body thickness
        reordered_thickness1[world_idx, contact_idx] = contact_thickness1[contact_idx]  # Ground thickness
        reordered_body_shape[world_idx, contact_idx] = shape_a_per_world  # Body shape index (per-world)
