import newton
import warp as wp


@wp.struct
class JointAxisKinematics:
    """Precomputed kinematic data for a single axis of a joint constraint."""

    J_parent: wp.spatial_vector
    J_child: wp.spatial_vector
    error: wp.float32
    compliance: wp.float32


@wp.struct
class JointInteraction:
    """A data descriptor for a single joint interaction between two bodies."""

    is_active: wp.bool

    # Indices of the parent and child bodies
    parent_idx: wp.int32
    child_idx: wp.int32

    axis0: JointAxisKinematics
    axis1: JointAxisKinematics
    axis2: JointAxisKinematics
    axis3: JointAxisKinematics
    axis4: JointAxisKinematics
    axis5: JointAxisKinematics


@wp.func
def orthogonal_basis(axis: wp.vec3):
    if wp.abs(axis.x) <= wp.abs(axis.y) and wp.abs(axis.x) <= wp.abs(axis.z):
        v = wp.vec3(1.0, 0.0, 0.0)
    elif wp.abs(axis.y) <= wp.abs(axis.z):
        v = wp.vec3(0.0, 1.0, 0.0)
    else:
        v = wp.vec3(0.0, 0.0, 1.0)
    b1 = wp.normalize(wp.cross(axis, v))
    b2 = wp.cross(axis, b1)
    return b1, b2


@wp.func
def compute_joint_kinematics(
    body_q: wp.transform, joint_X: wp.transform, body_com: wp.vec3
) -> wp.vec3:
    X_wj = body_q * joint_X
    joint_pos_world = wp.transform_get_translation(X_wj)
    body_com_world = wp.transform_point(body_q, body_com)
    return joint_pos_world - body_com_world


@wp.func
def get_joint_axis_kinematics(
    interaction: JointInteraction, axis_index: wp.int32
) -> JointAxisKinematics:
    """Selects the kinematic data for a specific axis from the unrolled struct."""
    if axis_index == 0:
        return interaction.axis0
    elif axis_index == 1:
        return interaction.axis1
    elif axis_index == 2:
        return interaction.axis2
    elif axis_index == 3:
        return interaction.axis3
    elif axis_index == 4:
        return interaction.axis4
    else:
        return interaction.axis5


@wp.kernel
def joint_interaction_kernel(
    # --- Inputs (same as before) ---
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    # --- Output ---
    interactions: wp.array(dtype=JointInteraction),
):
    joint_idx = wp.tid()

    interaction = JointInteraction()
    interaction.is_active = False  # Default to inactive
    j_type = joint_type[joint_idx]

    parent_idx = joint_parent[joint_idx]
    child_idx = joint_child[joint_idx]

    if (
        joint_enabled[joint_idx] == 0
        or j_type != newton.JointType.REVOLUTE
        or child_idx < 0  # A joint must have a child body
    ):
        interactions[joint_idx] = interaction
        return

    interaction.is_active = True
    interaction.parent_idx = parent_idx
    interaction.child_idx = child_idx

    # --- BEGIN KINEMATICS ---
    body_q_c = body_q[child_idx]

    body_q_p = wp.transform_identity()
    if parent_idx >= 0:
        body_q_p = body_q[parent_idx]

    child_com = body_com[child_idx]
    parent_com = wp.vec3()  # Default to origin (0,0,0) for world
    if parent_idx >= 0:
        parent_com = body_com[parent_idx]

    r_c = compute_joint_kinematics(body_q_c, joint_X_c[joint_idx], child_com)
    r_p = compute_joint_kinematics(body_q_p, joint_X_p[joint_idx], parent_com)

    joint_pos_c = wp.transform_get_translation(body_q_c * joint_X_c[joint_idx])
    joint_pos_p = wp.transform_get_translation(body_q_p * joint_X_p[joint_idx])
    c_pos = joint_pos_c - joint_pos_p

    # Create a single temporary struct to build each axis
    axis_kin = JointAxisKinematics()

    # --- Positional Constraints (Axes 0, 1, 2) | Translation ---
    lin_compliance = 0.1  # TODO: Fix the compliances

    # Axis 0: X translation
    axis_kin.J_child = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, r_c[2], -r_c[1])
    axis_kin.J_parent = wp.spatial_vector(-1.0, 0.0, 0.0, 0.0, -r_p[2], r_p[1])
    axis_kin.error = c_pos.x
    axis_kin.compliance = lin_compliance
    interaction.axis0 = axis_kin

    # Axis 1: Y translation
    axis_kin.J_child = wp.spatial_vector(0.0, 1.0, 0.0, -r_c[2], 0.0, r_c[0])
    axis_kin.J_parent = wp.spatial_vector(0.0, -1.0, 0.0, r_p[2], 0.0, -r_p[0])
    axis_kin.error = c_pos.y
    interaction.axis1 = axis_kin

    # Axis 2: Z translation
    axis_kin.J_child = wp.spatial_vector(0.0, 0.0, 1.0, r_c[1], -r_c[0], 0.0)
    axis_kin.J_parent = wp.spatial_vector(0.0, 0.0, -1.0, -r_p[1], r_p[0], 0.0)
    axis_kin.error = c_pos.z
    interaction.axis2 = axis_kin

    # --- Rotational Constraints (Axes 3, 4) | Swing ---
    ang_compliance = 0.1

    axis_start_idx = joint_qd_start[joint_idx]
    axis = joint_axis[axis_start_idx]

    X_wp = body_q_p * joint_X_p[joint_idx]
    X_wc = body_q_c * joint_X_c[joint_idx]

    # Get the rotational part of those full transforms
    q_wp_rot = wp.transform_get_rotation(X_wp)
    q_wc_rot = wp.transform_get_rotation(X_wc)

    axis_p_w = wp.quat_rotate(q_wp_rot, axis)

    b1_local, b2_local = orthogonal_basis(axis)

    # Axis 3: First rotational constraint
    b1_c_w = wp.quat_rotate(q_wc_rot, b1_local)
    axis_kin.error = wp.dot(axis_p_w, b1_c_w)
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)
    axis_kin.J_child = wp.spatial_vector(wp.vec3(), -b1_x_axis)
    axis_kin.J_parent = wp.spatial_vector(wp.vec3(), b1_x_axis)
    axis_kin.compliance = ang_compliance
    interaction.axis3 = axis_kin

    # Axis 4: Second rotational constraint
    b2_c_w = wp.quat_rotate(q_wc_rot, b2_local)
    axis_kin.error = wp.dot(axis_p_w, b2_c_w)
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)
    axis_kin.J_child = wp.spatial_vector(wp.vec3(), -b2_x_axis)
    axis_kin.J_parent = wp.spatial_vector(wp.vec3(), b2_x_axis)
    interaction.axis4 = axis_kin

    interactions[joint_idx] = interaction
