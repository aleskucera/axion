import warp as wp


@wp.kernel
def compute_contact_world_indices(
    shape_world: wp.array(dtype=wp.int32),
    # Contact info
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    world_counters: wp.array(dtype=wp.int32),
    # OUTPUT: This array holds the assigned column index for every contact
    out_write_slots: wp.array(dtype=wp.int32),
):
    contact_idx = wp.tid()

    shape_0 = contact_shape0[contact_idx]
    shape_1 = contact_shape1[contact_idx]

    if contact_idx >= contact_count[0] or shape_0 == shape_1:
        return

    world_id = -1
    if shape_0 >= 0:
        world_id = shape_world[shape_0]
    if shape_1 >= 0:
        world_id = shape_world[shape_1]

    if world_id >= 0:
        # Reserve a slot atomically
        # We do this ONCE. Ideally, this is fast because we aren't moving heavy data.
        slot = wp.atomic_add(world_counters, world_id, 1)
    else:
        slot = -1

    # Store the reservation. Even if slot > 100, we store it to check later.
    out_write_slots[contact_idx] = slot


@wp.kernel
def scatter_contact_data(
    # Maps
    world_indices: wp.array(dtype=wp.int32),
    write_slots: wp.array(dtype=wp.int32),
    # Source Data
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_thickness0: wp.array(dtype=wp.float32),
    contact_thickness1: wp.array(dtype=wp.float32),
    # Destination Data
    mapped_contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    mapped_contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    mapped_contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    mapped_contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    mapped_contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    mapped_contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    mapped_contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
):
    contact_idx = wp.tid()

    num_worlds = mapped_contact_point0.shape[0]
    max_contacts_per_world = mapped_contact_point0.shape[1]
    contact_num_in_world = write_slots[contact_idx]

    # Bounds check
    if contact_num_in_world >= max_contacts_per_world:
        return

    world = world_indices[contact_idx]

    mapped_contact_point0[world, contact_num_in_world] = contact_point0[contact_idx]
    mapped_contact_point1[world, contact_num_in_world] = contact_point1[contact_idx]
    mapped_contact_normal[world, contact_num_in_world] = contact_normal[contact_idx]
    mapped_contact_shape0[world, contact_num_in_world] = contact_shape0[contact_idx] // num_worlds
    mapped_contact_shape1[world, contact_num_in_world] = contact_shape1[contact_idx] // num_worlds
    mapped_contact_thickness0[world, contact_num_in_world] = contact_thickness0[contact_idx]
    mapped_contact_thickness1[world, contact_num_in_world] = contact_thickness1[contact_idx]
