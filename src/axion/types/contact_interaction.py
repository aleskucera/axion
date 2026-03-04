import warp as wp
from axion.math import orthogonal_basis

CAPSULE_MAX_CONTACTS_PER_BODY = 2

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
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    shape_a = contact_shape0[world_idx, contact_idx]
    shape_b = contact_shape1[world_idx, contact_idx]

    if contact_idx >= contact_count[world_idx] or shape_a == shape_b:
        contact_active_mask[world_idx, contact_idx] = 0.0
        return

    contact_active_mask[world_idx, contact_idx] = 1.0

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
    # --- Per-body contact counter (zeroed before launch), used to assign output slots ---
    body_contact_count: wp.array(dtype=wp.int32, ndim=1),
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
    - Output order: link1_contact1, link1_contact2, link2_contact1, link2_contact2, ...
      (slot = link_index * MAX_CONTACTS_PER_BODY + contact_index_within_body)
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

    # Body index (the non-ground body); assign output slot by (link, contact index)
    body_idx = body_a if body_a >= 0 else body_b
    contact_slot_within_body = wp.atomic_add(body_contact_count, body_idx, 1)
    link_index = body_idx
    if link_index < 0:
        return
    slot = link_index * CAPSULE_MAX_CONTACTS_PER_BODY + contact_slot_within_body

    max_slots = reordered_point0.shape[1]
    if slot >= max_slots:
        return

    # Determine if we need to swap (if body_a is ground, swap everything)
    need_swap = (body_a == -1)

    if need_swap:
        # Swap: body is in position 1, ground is in position 0
        reordered_point0[world_idx, slot] = contact_point1[contact_idx]  # Body point
        reordered_point1[world_idx, slot] = contact_point0[contact_idx]  # Ground point
        reordered_normal[world_idx, slot] = -contact_normal[contact_idx]  # Flip normal (from body to ground)
        reordered_thickness0[world_idx, slot] = contact_thickness1[contact_idx]  # Body thickness
        reordered_thickness1[world_idx, slot] = contact_thickness0[contact_idx]  # Ground thickness
        reordered_body_shape[world_idx, slot] = shape_b_per_world  # Body shape index (per-world)
    else:
        # Already correct: body is in position 0, ground is in position 1
        reordered_point0[world_idx, slot] = contact_point0[contact_idx]  # Body point
        reordered_point1[world_idx, slot] = contact_point1[contact_idx]  # Ground point
        reordered_normal[world_idx, slot] = contact_normal[contact_idx]  # Normal already correct
        reordered_thickness0[world_idx, slot] = contact_thickness0[contact_idx]  # Body thickness
        reordered_thickness1[world_idx, slot] = contact_thickness1[contact_idx]  # Ground thickness
        reordered_body_shape[world_idx, slot] = shape_a_per_world  # Body shape index (per-world)

@wp.kernel
def contact_penetration_depth_kernel(
    # --- Inputs (reordered 2D arrays: body always in position 0, ground in position 1) ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
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

    Sign convention:
    - Negative depth: penetrating (bodies overlap).
    - Positive depth: not touching (separation or inactive slot).
    - Inactive/invalid contact slots are set to NON_TOUCHING_DEPTH (e.g. 1000.0).
    """
    NON_TOUCHING_DEPTH = 1000.0  # Large positive value for inactive or non-touching contacts

    world_idx, contact_idx = wp.tid()

    # Get body index from body shape
    body_shape_idx = body_shape[world_idx, contact_idx]
    if body_shape_idx < 0:
        depths[world_idx, contact_idx] = NON_TOUCHING_DEPTH  # Invalid body shape
        return

    body_idx = shape_body[world_idx, body_shape_idx]
    if body_idx < 0:
        depths[world_idx, contact_idx] = NON_TOUCHING_DEPTH  # Invalid body
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

    # Signed depth: dot(n, ground - body) is positive when penetrating.
    # Negate so that penetrating -> negative depth, separated -> positive depth.
    raw = wp.dot(n, p_ground_world_adj - p_body_world_adj)
    depths[world_idx, contact_idx] = -raw