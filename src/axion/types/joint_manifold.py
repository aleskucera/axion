import warp as wp


@wp.struct
class JointConstraintTerm:
    """Stores precomputed kinematic data for a single constraint axis."""

    J_p: wp.spatial_vector
    J_c: wp.spatial_vector
    error: wp.float32
    compliance: wp.float32


@wp.struct
class JointManifold:
    """
    Stores all precomputed data for a single joint.
    """

    is_active: wp.bool

    parent_idx: wp.int32
    child_idx: wp.int32

    term0: JointConstraintTerm
    term1: JointConstraintTerm
    term2: JointConstraintTerm
    term3: JointConstraintTerm
    term4: JointConstraintTerm


@wp.func
def set_joint_term(
    manifold: JointManifold, index: wp.int32, term: JointConstraintTerm
) -> JointManifold:
    """
    Helper function to set a constraint term by index in a manifold.
    It takes a manifold by value and returns the modified version.
    """
    if index == 0:
        manifold.term0 = term
    elif index == 1:
        manifold.term1 = term
    elif index == 2:
        manifold.term2 = term
    elif index == 3:
        manifold.term3 = term
    else:  # index == 4
        manifold.term4 = term
    return manifold


@wp.func
def get_joint_term(manifold: JointManifold, index: wp.int32) -> JointConstraintTerm:
    if index == 0:
        return manifold.term0
    elif index == 1:
        return manifold.term1
    elif index == 2:
        return manifold.term2
    elif index == 3:
        return manifold.term3
    else:  # index == 4
        return manifold.term4


# Provided helper functions (unchanged)
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


@wp.kernel
def joint_manifold_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    # --- Joint Definition Inputs ---
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
    joint_manifold: wp.array(dtype=JointManifold),
):
    joint_idx = wp.tid()

    # It's good practice to check bounds against a main input array
    if joint_idx >= joint_type.shape[0]:
        return

    manifold = JointManifold()
    manifold.is_active = False  # Default to inactive
    j_type = joint_type[joint_idx]

    # Early exit for disabled, non-revolute, or fixed-base joints
    # (Including more robust checks for body indices)
    num_bodies = body_q.shape[0]
    parent_idx = joint_parent[joint_idx]
    child_idx = joint_child[joint_idx]
    if (
        joint_enabled[joint_idx] == 0
        or j_type != wp.sim.JOINT_REVOLUTE
        or parent_idx < 0
        or parent_idx >= num_bodies
        or child_idx < 0
        or child_idx >= num_bodies
    ):
        joint_manifold[joint_idx] = manifold
        return

    manifold.is_active = True
    manifold.child_idx = child_idx
    manifold.parent_idx = parent_idx

    # --- Common Kinematics (depend on body_q) ---
    body_q_c = body_q[child_idx]
    body_q_p = body_q[parent_idx]

    r_c = compute_joint_kinematics(body_q_c, joint_X_c[joint_idx], body_com[child_idx])
    r_p = compute_joint_kinematics(body_q_p, joint_X_p[joint_idx], body_com[parent_idx])

    joint_pos_c = wp.transform_get_translation(body_q_c * joint_X_c[joint_idx])
    joint_pos_p = wp.transform_get_translation(body_q_p * joint_X_p[joint_idx])
    c_pos = joint_pos_c - joint_pos_p

    q_c_rot = wp.transform_get_rotation(body_q_c)
    q_p_rot = wp.transform_get_rotation(body_q_p)

    # Create a single temporary term to be reused
    term = JointConstraintTerm()

    # --- Positional Constraints (Index 0, 1, 2) | Translation ---
    lin_compliance = joint_linear_compliance[joint_idx]

    # Term 0: X translation
    term.J_c = wp.spatial_vector(0.0, r_c[2], -r_c[1], 1.0, 0.0, 0.0)
    term.J_p = wp.spatial_vector(0.0, -r_p[2], r_p[1], -1.0, 0.0, 0.0)
    term.error = c_pos.x
    term.compliance = lin_compliance
    manifold = set_joint_term(manifold, 0, term)

    # Term 1: Y translation
    term.J_c = wp.spatial_vector(-r_c[2], 0.0, r_c[0], 0.0, 1.0, 0.0)
    term.J_p = wp.spatial_vector(r_p[2], 0.0, -r_p[0], 0.0, -1.0, 0.0)
    term.error = c_pos.y
    term.compliance = lin_compliance
    manifold = set_joint_term(manifold, 1, term)

    # Term 2: Z translation
    term.J_c = wp.spatial_vector(r_c[1], -r_c[0], 0.0, 0.0, 0.0, 1.0)
    term.J_p = wp.spatial_vector(-r_p[1], r_p[0], 0.0, 0.0, 0.0, -1.0)
    term.error = c_pos.z
    term.compliance = lin_compliance
    manifold = set_joint_term(manifold, 2, term)

    # --- Rotational Constraints (Index 3, 4) | Swing ---
    ang_compliance = joint_angular_compliance[joint_idx]
    zero_vec = wp.vec3()

    axis_start_idx = joint_axis_start[joint_idx]
    axis = joint_axis[axis_start_idx]
    axis_p_w = wp.quat_rotate(q_p_rot, axis)
    b1_c, b2_c = orthogonal_basis(axis)

    # Term 3: First rotational constraint
    b1_c_w = wp.quat_rotate(q_c_rot, b1_c)
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)
    term.J_c = wp.spatial_vector(-b1_x_axis, zero_vec)
    term.J_p = wp.spatial_vector(b1_x_axis, zero_vec)
    term.error = wp.dot(axis_p_w, b1_c_w)
    term.compliance = ang_compliance
    manifold = set_joint_term(manifold, 3, term)

    # Term 4: Second rotational constraint
    b2_c_w = wp.quat_rotate(q_c_rot, b2_c)
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)
    term.J_c = wp.spatial_vector(-b2_x_axis, zero_vec)
    term.J_p = wp.spatial_vector(b2_x_axis, zero_vec)
    term.error = wp.dot(axis_p_w, b2_c_w)
    term.compliance = ang_compliance
    manifold = set_joint_term(manifold, 4, term)

    # Finally, write the fully populated manifold to global memory
    joint_manifold[joint_idx] = manifold
