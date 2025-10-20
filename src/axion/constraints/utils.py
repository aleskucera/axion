import warp as wp
from axion.types import ContactInteraction
from axion.types import JointConstraintData


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


@wp.func
def scaled_fisher_burmeister_new(
    a: wp.float32,
    b: wp.float32,
    r: wp.float32 = 1.0,
):
    scaled_b = r * b
    norm = wp.sqrt(a**2.0 + scaled_b**2.0)

    value = a + scaled_b - norm

    # Avoid division by zero
    if norm < 1e-6:
        return value, 0.0, 1.0

    dvalue_da = 1.0 - a / norm
    dvalue_db = r * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db


@wp.kernel
def fill_joint_constraint_body_idx_kernel(
    joint_constraints: wp.array(dtype=JointConstraintData),
    joint_constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
):
    constraint_idx = wp.tid()

    if constraint_idx >= len(joint_constraints):
        return

    c = joint_constraints[constraint_idx]
    joint_constraint_body_idx[constraint_idx, 0] = c.parent_idx
    joint_constraint_body_idx[constraint_idx, 1] = c.child_idx


@wp.kernel
def fill_contact_constraint_body_idx_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction),
    contact_constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
):
    contact_idx = wp.tid()

    if contact_idx >= len(contact_interaction):
        return

    interaction = contact_interaction[contact_idx]
    contact_constraint_body_idx[contact_idx, 0] = interaction.body_a_idx
    contact_constraint_body_idx[contact_idx, 1] = interaction.body_b_idx


@wp.kernel
def fill_friction_constraint_body_idx_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction),
    friction_constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
):
    constraint_idx = wp.tid()
    contact_idx = constraint_idx // 2

    if contact_idx >= len(contact_interaction):
        return

    interaction = contact_interaction[contact_idx]
    friction_constraint_body_idx[constraint_idx, 0] = interaction.body_a_idx
    friction_constraint_body_idx[constraint_idx, 1] = interaction.body_b_idx


# @wp.kernel
# def update_constraint_body_idx_kernel(
#     joint_constraints: wp.array(dtype=JointConstraintData),
#     contact_interaction: wp.array(dtype=ContactInteraction),
#     # --- Outputs ---
#     constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
# ):
#     constraint_idx = wp.tid()
#     N_jc = len(joint_constraints)
#     N_n = len(contact_interaction)
#
#     body_a = -1
#     body_b = -1
#
#     if constraint_idx < N_jc:
#         c = joint_constraints[constraint_idx]
#         body_a = c.parent_idx
#         body_b = c.child_idx
#     elif constraint_idx < N_jc + N_n:
#         offset = N_jc
#         contact_idx = constraint_idx - offset
#
#         body_a = contact_interaction[contact_idx].body_a_idx
#         body_b = contact_interaction[contact_idx].body_b_idx
#     else:
#         offset = N_jc + N_n
#         contact_idx = (constraint_idx - offset) // 2
#
#         body_a = contact_interaction[contact_idx].body_a_idx
#         body_b = contact_interaction[contact_idx].body_b_idx
#
#     constraint_body_idx[constraint_idx, 0] = body_a
#     constraint_body_idx[constraint_idx, 1] = body_b
