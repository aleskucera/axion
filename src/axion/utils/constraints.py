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
    joint_index = (constraint_idx - J_j_offset) // 5
    normal_index = constraint_idx - J_n_offset
    friction_index = (constraint_idx - J_f_offset) // 2

    joint_body_a = joint_parent[joint_index]
    joint_body_b = joint_child[joint_index]

    contact_body_a_n = contact_body_a[normal_index]
    contact_body_b_n = contact_body_b[normal_index]

    contact_body_a_f = contact_body_a[friction_index]
    contact_body_b_f = contact_body_b[friction_index]

    body_a = wp.where(
        constraint_idx < J_n_offset,
        joint_body_a,
        wp.where(constraint_idx < J_f_offset, contact_body_a_n, contact_body_a_f),
    )
    body_b = wp.where(
        constraint_idx < J_n_offset,
        joint_body_b,
        wp.where(constraint_idx < J_f_offset, contact_body_b_n, contact_body_b_f),
    )

    return body_a, body_b


# @wp.func
# def get_constraint_body_index(
#     joint_parent: wp.array(dtype=wp.int32),
#     joint_child: wp.array(dtype=wp.int32),
#     contact_body_a: wp.array(dtype=wp.int32),
#     contact_body_b: wp.array(dtype=wp.int32),
#     J_j_offset: int,
#     J_n_offset: int,
#     J_f_offset: int,
#     constraint_idx: int,
# ):
#     """Maps a constraint index to the two body indices it affects."""
#     body_a = -1
#     body_b = -1
#     if constraint_idx < J_n_offset:
#         joint_index = (constraint_idx - J_j_offset) // 5
#         body_a = joint_parent[joint_index]
#         body_b = joint_child[joint_index]
#     elif constraint_idx < J_f_offset:
#         contact_index = constraint_idx - J_n_offset
#         body_a = contact_body_a[contact_index]
#         body_b = contact_body_b[contact_index]
#     else:
#         contact_index = (constraint_idx - J_f_offset) // 2
#         body_a = contact_body_a[contact_index]
#         body_b = contact_body_b[contact_index]
#     return body_a, body_b
