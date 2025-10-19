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

    # Number of DOFs, that are constrained by this joint
    num_constraints: wp.int32       

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

    if axis_index >= interaction.num_constraints:
        return JointAxisKinematics()

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


@wp.func
def fix_all_translational_axes(
    interaction: JointInteraction,
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    lin_compliance: wp.float32,
):
    """Helper functions that sets all 3 translation axes of a joint to be fixed"""

    # Create a single temporary struct to build each axis
    axis_kin = JointAxisKinematics()

    # Axis 0: X translation
    axis_kin.J_child = wp.spatial_vector(0.0, r_c[2], -r_c[1], 1.0, 0.0, 0.0)
    axis_kin.J_parent = wp.spatial_vector(0.0, -r_p[2], r_p[1], -1.0, 0.0, 0.0)
    axis_kin.error = c_pos.x
    axis_kin.compliance = lin_compliance
    interaction.axis0 = axis_kin

    # Axis 1: Y translation
    axis_kin.J_child = wp.spatial_vector(-r_c[2], 0.0, r_c[0], 0.0, 1.0, 0.0)
    axis_kin.J_parent = wp.spatial_vector(r_p[2], 0.0, -r_p[0], 0.0, -1.0, 0.0)
    axis_kin.error = c_pos.y
    interaction.axis1 = axis_kin

    # Axis 2: Z translation
    axis_kin.J_child = wp.spatial_vector(r_c[1], -r_c[0], 0.0, 0.0, 0.0, 1.0)
    axis_kin.J_parent = wp.spatial_vector(-r_p[1], r_p[0], 0.0, 0.0, 0.0, -1.0)
    axis_kin.error = c_pos.z
    interaction.axis2 = axis_kin

    return interaction


@wp.func
def set_revolute_interaction_constraints(
    interaction: JointInteraction,
    joint_idx: wp.int32,
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_c_rot: wp.quat,
    q_p_rot: wp.quat,
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
) -> JointInteraction:
    """Fill JointInteraction for a revolute joint"""

    interaction.num_constraints = 5

    # --- Positional Constraints (Axes 0, 1, 2) | Translation ---
    lin_compliance = joint_linear_compliance[joint_idx]
    interaction = fix_all_translational_axes(interaction, c_pos, r_c, r_p, lin_compliance)

    # --- Rotational Constraints (Axes 3, 4) | Swing ---
    axis_kin = JointAxisKinematics()  # Create a single temporary struct to build each axis
    ang_compliance = joint_angular_compliance[joint_idx]

    axis_start_idx = joint_axis_start[joint_idx]
    axis = joint_axis[axis_start_idx]
    axis_p_w = wp.quat_rotate(q_p_rot, axis)
    b1_c, b2_c = orthogonal_basis(axis)

    # Axis 3: First rotational constraint
    b1_c_w = wp.quat_rotate(q_c_rot, b1_c)
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)
    axis_kin.J_child = wp.spatial_vector(-b1_x_axis, wp.vec3())
    axis_kin.J_parent = wp.spatial_vector(b1_x_axis, wp.vec3())
    axis_kin.error = wp.dot(axis_p_w, b1_c_w)
    axis_kin.compliance = ang_compliance
    interaction.axis3 = axis_kin

    # Axis 4: Second rotational constraint
    b2_c_w = wp.quat_rotate(q_c_rot, b2_c)
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)
    axis_kin.J_child = wp.spatial_vector(-b2_x_axis, wp.vec3())
    axis_kin.J_parent = wp.spatial_vector(b2_x_axis, wp.vec3())
    axis_kin.error = wp.dot(axis_p_w, b2_c_w)
    interaction.axis4 = axis_kin

    return interaction


@wp.func
def set_spherical_interaction_constraints(
    interaction: JointInteraction,
    joint_idx: wp.int32,
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_c_rot: wp.quat,
    q_p_rot: wp.quat,
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
) -> JointInteraction:
    """Fill JointInteraction for a spherical joint"""

    interaction.num_constraints = 3

    # --- Positional Constraints (Axes 0, 1, 2) | Translation ---
    lin_compliance = joint_linear_compliance[joint_idx]
    interaction = fix_all_translational_axes(interaction, c_pos, r_c, r_p, lin_compliance)

    return interaction

@wp.kernel
def revolute_joint_interaction_kernel(
    # --- Inputs (same as before) ---
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    # --- Output ---
    revolute_interactions: wp.array(dtype=JointInteraction),
):
    joint_idx = wp.tid()
    j_type = joint_type[joint_idx]
    parent_idx = joint_parent[joint_idx]
    child_idx = joint_child[joint_idx]

    #Early exit for joints of different type:
    if j_type != wp.sim.JOINT_REVOLUTE:
        # skip with no JointInteraction created
        return  

    # Early exit for disabled or invalid joints
    if (joint_enabled[joint_idx] == 0 or parent_idx < 0 or child_idx < 0):
        # return an inactive interaction
        interaction = JointInteraction()
        interaction.is_active = False
        revolute_interactions[joint_idx] = interaction
        return

    # Continue with eligible revolute joint
    interaction = JointInteraction()
    interaction.is_active = True
    interaction.parent_idx = parent_idx
    interaction.child_idx = child_idx

    # --- Common Kinematics (depend on body_q) ---
    body_q_c = body_q[child_idx]  # child's transformation
    body_q_p = body_q[parent_idx]  # parent's transformation

    r_c = compute_joint_kinematics(
        body_q_c, joint_X_c[joint_idx], body_com[child_idx]
    )  # child link's center of mass position vector
    r_p = compute_joint_kinematics(
        body_q_p, joint_X_p[joint_idx], body_com[parent_idx]
    )  # parent link's center of mass position vector

    joint_pos_c = wp.transform_get_translation(body_q_c * joint_X_c[joint_idx])
    joint_pos_p = wp.transform_get_translation(body_q_p * joint_X_p[joint_idx])
    c_pos = joint_pos_c - joint_pos_p  # joint position

    q_c_rot = wp.transform_get_rotation(body_q_c)  # child's rotation (quaternions)
    q_p_rot = wp.transform_get_rotation(body_q_p)  # parent's rotation (quaternions)

    # Call appropriate functions for the joint type
    interaction = set_revolute_interaction_constraints(
        interaction,
        joint_idx,
        joint_axis_start,
        joint_axis,
        c_pos,
        r_c,
        r_p,
        q_c_rot,
        q_p_rot,
        joint_linear_compliance,
        joint_angular_compliance,
    )

    # Write the fully populated interaction data to global memory
    revolute_interactions[joint_idx] = interaction


@wp.kernel
def spherical_joint_interaction_kernel(
    # --- Inputs (same as before) ---
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=wp.int32),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_linear_compliance: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
    # --- Output ---
    spherical_interactions: wp.array(dtype=JointInteraction),
):
    joint_idx = wp.tid()
    j_type = joint_type[joint_idx]
    parent_idx = joint_parent[joint_idx]
    child_idx = joint_child[joint_idx]

    #Early exit for joints of different type:
    if j_type != wp.sim.JOINT_BALL:
        # skip with no JointInteraction created
        return  

    # Early exit for disabled or invalid joints
    if (joint_enabled[joint_idx] == 0 or parent_idx < 0 or child_idx < 0):
        # return an inactive interaction
        interaction = JointInteraction()
        interaction.is_active = False
        spherical_interactions[joint_idx] = interaction
        return

    # Continue with eligible revolute joint
    interaction = JointInteraction()
    interaction.is_active = True
    interaction.parent_idx = parent_idx
    interaction.child_idx = child_idx

    # --- Common Kinematics (depend on body_q) ---
    body_q_c = body_q[child_idx]  # child's transformation
    body_q_p = body_q[parent_idx]  # parent's transformation

    r_c = compute_joint_kinematics(
        body_q_c, joint_X_c[joint_idx], body_com[child_idx]
    )  # child link's center of mass position vector
    r_p = compute_joint_kinematics(
        body_q_p, joint_X_p[joint_idx], body_com[parent_idx]
    )  # parent link's center of mass position vector

    joint_pos_c = wp.transform_get_translation(body_q_c * joint_X_c[joint_idx])
    joint_pos_p = wp.transform_get_translation(body_q_p * joint_X_p[joint_idx])
    c_pos = joint_pos_c - joint_pos_p  # joint position

    q_c_rot = wp.transform_get_rotation(body_q_c)  # child's rotation (quaternions)
    q_p_rot = wp.transform_get_rotation(body_q_p)  # parent's rotation (quaternions)

    # Call appropriate functions for the joint type
    interaction = set_spherical_interaction_constraints(
        interaction,
        joint_idx,
        joint_axis_start,
        joint_axis,
        c_pos,
        r_c,
        r_p,
        q_c_rot,
        q_p_rot,
        joint_linear_compliance,
        joint_angular_compliance,
    )

    # Write the fully populated interaction data to global memory
    spherical_interactions[joint_idx] = interaction

#TODO: there is some duplication between the two joint interaction kernels above. Consider refactoring later.