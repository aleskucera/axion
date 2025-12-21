import warp as wp
from axion.types import ContactInteraction


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
    # --- Joint Definition ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Output ---
    joint_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, joint_idx = wp.tid()
    
    # Check bounds (though dims should match)
    if world_idx >= joint_type.shape[0] or joint_idx >= joint_type.shape[1]:
        return

    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Determine constraint count
    count = 0
    if j_type == 1: # REVOLUTE
        count = 5
    elif j_type == 2: # BALL
        count = 3
    elif j_type == 3: # FIXED
        count = 6
        
    for k in range(count):
        offset = start_offset + k
        # Safety check for output bounds
        if offset < joint_constraint_body_idx.shape[1]:
            joint_constraint_body_idx[world_idx, offset, 0] = p_idx
            joint_constraint_body_idx[world_idx, offset, 1] = c_idx


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
    # --- Joint Definition ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Output ---
    joint_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if world_idx >= joint_type.shape[0] or joint_idx >= joint_type.shape[1]:
        return

    j_type = joint_type[world_idx, joint_idx]
    is_enabled = (joint_enabled[world_idx, joint_idx] != 0)
    
    # Check if valid child
    if joint_child[world_idx, joint_idx] < 0:
        is_enabled = False

    start_offset = constraint_offsets[world_idx, joint_idx]

    count = 0
    if j_type == 1: # REVOLUTE
        count = 5
    elif j_type == 2: # BALL
        count = 3
    elif j_type == 3: # FIXED
        count = 6
        
    val = 1.0 if is_enabled else 0.0

    for k in range(count):
        offset = start_offset + k
        if offset < joint_constraint_active_mask.shape[1]:
            joint_constraint_active_mask[world_idx, offset] = val


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