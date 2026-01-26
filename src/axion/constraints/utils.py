import numpy as np
import warp as wp
from axion.types import ContactInteraction


@wp.func
def compute_spatial_momentum(
    mass: wp.float32,
    inertia: wp.mat33,
    velocity: wp.spatial_vector,
) -> wp.spatial_vector:
    top = mass * wp.spatial_top(velocity)
    bot = inertia @ wp.spatial_bottom(velocity)
    return wp.spatial_vector(top, bot)


@wp.func
def compute_world_inertia(
    body_q: wp.transform,
    body_inertia: wp.mat33,
) -> wp.mat33:
    # Get orientation quaternion from transform
    orientation = wp.transform_get_rotation(body_q)
    R = wp.quat_to_matrix(orientation)

    # Transform inertia tensor: I_w = R * I_body * R^T
    I_w = R @ body_inertia @ wp.transpose(R)

    return I_w


@wp.func
def compute_effective_mass(
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    J_1: wp.spatial_vector,
    J_2: wp.spatial_vector,
    m_inv_1: wp.float32,
    I_inv_b_1: wp.mat33,
    m_inv_2: wp.float32,
    I_inv_b_2: wp.mat33,
    body_1_idx: int,
    body_2_idx: int,
) -> float:
    """
    Computes the diagonal term (effective mass) J M^-1 J^T.
    Expects M_inv to be in WORLD frame.
    """
    val = 0.0
    if body_1_idx >= 0:
        # compute J M^-1 J^T
        I_inv_w_1 = compute_world_inertia(body_q_1, I_inv_b_1)
        val += wp.dot(J_1, compute_spatial_momentum(m_inv_1, I_inv_w_1, J_1))

    if body_2_idx >= 0:
        I_inv_w_2 = compute_world_inertia(body_q_2, I_inv_b_2)
        val += wp.dot(J_2, compute_spatial_momentum(m_inv_2, I_inv_w_2, J_2))

    return val


def compute_joint_constraint_offsets_batched(joint_types: wp.array):
    """
    joint_types: numpy array of shape (num_worlds, num_joints)
    """

    constraint_count_map = np.array(
        [
            5,  # PRISMATIC = 0
            5,  # REVOLUTE  = 1
            3,  # BALL      = 2
            6,  # FIXED     = 3
            0,  # FREE      = 4
            1,  # DISTANCE  = 5
            6,  # D6        = 6
            0,  # CABLE     = 7
        ],
        dtype=np.int32,
    )

    joint_types_np = joint_types.numpy()  # (num_worlds, num_joints)
    # Map joint types → constraint counts
    constraint_counts = constraint_count_map[joint_types_np]  # (num_worlds, num_joints)

    # Total constraints for each batch
    total_constraints = constraint_counts.sum(axis=1)  # (num_worlds,)

    # Compute offsets per batch
    # For each batch: offsets[i, :] = cumsum(counts[i, :]) - counts[i, 0]
    constraint_offsets = np.zeros_like(constraint_counts)  # (num_worlds, num_joints)
    constraint_offsets[:, 1:] = np.cumsum(constraint_counts[:, :-1], axis=1)

    # Convert to wp.array (must flatten or provide device explicitly)
    constraint_offsets_wp = wp.array(
        constraint_offsets,
        dtype=wp.int32,
        device=joint_types.device,
    )

    return constraint_offsets_wp, total_constraints[0]


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
    if j_type == 0:  # PRISMATIC
        count = 5
    elif j_type == 1:  # REVOLUTE
        count = 5
    elif j_type == 2:  # BALL
        count = 3
    elif j_type == 3:  # FIXED
        count = 6
    elif j_type == 7:  # CABLE
        count = 0

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
    is_enabled = joint_enabled[world_idx, joint_idx] != 0

    # Check if valid child
    if joint_child[world_idx, joint_idx] < 0:
        is_enabled = False

    start_offset = constraint_offsets[world_idx, joint_idx]

    count = 0
    if j_type == 0:  # PRISMATIC
        count = 5
    elif j_type == 1:  # REVOLUTE
        count = 5
    elif j_type == 2:  # BALL
        count = 3
    elif j_type == 3:  # FIXED
        count = 6
    elif j_type == 7:  # CABLE
        count = 0

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
