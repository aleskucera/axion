import warp as wp


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
