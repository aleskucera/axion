import warp as wp


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


@wp.kernel
def update_constraint_body_idx_kernel(
    shape_body: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    # --- Parameters ---
    joint_count: wp.uint32,
    max_contact_count: wp.uint32,
    # --- Outputs ---
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
):
    constraint_idx = wp.tid()
    nj = wp.int32(joint_count)
    nc = wp.int32(max_contact_count)

    body_a = -1
    body_b = -1

    if constraint_idx < 5 * nj:
        joint_index = constraint_idx // 5
        body_a = joint_parent[joint_index]
        body_b = joint_child[joint_index]
    elif constraint_idx < 5 * nj + nc:
        offset = 5 * nj
        contact_index = (constraint_idx - offset) // 1

        shape_a = contact_shape0[contact_index]
        shape_b = contact_shape1[contact_index]

        if shape_a != shape_b:
            if shape_a >= 0:
                body_a = shape_body[shape_a]
            if shape_b >= 0:
                body_b = shape_body[shape_b]

    else:
        offset = 5 * nj + nc
        contact_index = (constraint_idx - offset) // 2
        shape_a = contact_shape0[contact_index]
        shape_b = contact_shape1[contact_index]

        if shape_a != shape_b:
            if shape_a >= 0:
                body_a = shape_body[shape_a]
            if shape_b >= 0:
                body_b = shape_body[shape_b]

    constraint_body_idx[constraint_idx, 0] = body_a
    constraint_body_idx[constraint_idx, 1] = body_b
