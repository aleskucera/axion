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
    if norm < 1e-5:
        return value, 0.0, 1.0

    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db


@wp.kernel
def fill_joint_constraint_body_idx_kernel(
    joint_constraints: wp.array(dtype=JointConstraintData, ndim=2),
    joint_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, constraint_idx = wp.tid()

    if world_idx >= joint_constraints.shape[0] or constraint_idx >= joint_constraints.shape[1]:
        return

    c = joint_constraints[world_idx, constraint_idx]
    joint_constraint_body_idx[world_idx, constraint_idx, 0] = c.parent_idx
    joint_constraint_body_idx[world_idx, constraint_idx, 1] = c.child_idx


@wp.kernel
def fill_contact_constraint_body_idx_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction, ndim=2),
    contact_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if world_idx >= contact_interaction.shape[0] or contact_idx >= contact_interaction.shape[1]:
        return

    interaction = contact_interaction[world_idx, contact_idx]
    contact_constraint_body_idx[world_idx, contact_idx, 0] = interaction.body_a_idx
    contact_constraint_body_idx[world_idx, contact_idx, 1] = interaction.body_b_idx


@wp.kernel
def fill_friction_constraint_body_idx_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction, ndim=2),
    friction_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, constraint_idx = wp.tid()
    contact_idx = constraint_idx // 2

    if world_idx >= contact_interaction.shape[0] or contact_idx >= contact_interaction.shape[1]:
        return

    interaction = contact_interaction[world_idx, contact_idx]
    friction_constraint_body_idx[world_idx, constraint_idx, 0] = interaction.body_a_idx
    friction_constraint_body_idx[world_idx, constraint_idx, 1] = interaction.body_b_idx


@wp.kernel
def fill_joint_constraint_active_mask_kernel(
    joint_constraints: wp.array(dtype=JointConstraintData, ndim=2),
    joint_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    if world_idx >= joint_constraints.shape[0] or constraint_idx >= joint_constraints.shape[1]:
        return

    c = joint_constraints[world_idx, constraint_idx]

    if c.is_active:
        joint_constraint_active_mask[world_idx, constraint_idx] = 1.0
    else:
        joint_constraint_active_mask[world_idx, constraint_idx] = 0.0


@wp.kernel
def fill_contact_constraint_active_mask_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction, ndim=2),
    contact_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    if world_idx >= contact_interaction.shape[0] or contact_idx >= contact_interaction.shape[1]:
        return

    interaction = contact_interaction[world_idx, contact_idx]
    if interaction.is_active:
        contact_constraint_active_mask[world_idx, contact_idx] = 1.0
    else:
        contact_constraint_active_mask[world_idx, contact_idx] = 0.0


@wp.kernel
def fill_friction_constraint_active_mask_kernel(
    contact_interaction: wp.array(dtype=ContactInteraction, ndim=2),
    friction_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()
    contact_idx = constraint_idx // 2

    if world_idx >= contact_interaction.shape[0] or contact_idx >= contact_interaction.shape[1]:
        return

    interaction = contact_interaction[world_idx, contact_idx]
    if interaction.is_active:
        friction_constraint_active_mask[world_idx, constraint_idx] = 1.0
    else:
        friction_constraint_active_mask[world_idx, constraint_idx] = 0.0
