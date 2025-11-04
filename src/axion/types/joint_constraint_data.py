import numpy as np
import warp as wp
from newton import JointType
from newton import utils

from .utils import orthogonal_basis


@wp.struct
class JointConstraintData:
    is_active: wp.bool

    value: wp.float32

    parent_idx: wp.int32
    child_idx: wp.int32

    J_parent: wp.spatial_vector
    J_child: wp.spatial_vector

def compute_joint_constraint_offsets(joint_types: wp.array) -> tuple[wp.array, int]:
    constraint_count_map = np.array(
        [
            5,  # PRISMATIC = 0
            5,  # REVOLUTE  = 1
            3,  # BALL      = 2
            6,  # FIXED     = 3
            0,  # FREE      = 4
            1,  # DISTANCE  = 5
            6,  # D6        = 6
        ],
        dtype=np.int32,
    )

    joint_types_np = joint_types.numpy()
    constraint_counts_np = constraint_count_map[joint_types_np]
    total_constraints = int(np.sum(constraint_counts_np))

    constraint_offsets_np = np.zeros_like(constraint_counts_np)
    constraint_offsets_np[1:] = np.cumsum(constraint_counts_np[:-1])

    constraint_offsets_wp = wp.array(
        constraint_offsets_np, dtype=wp.int32, device=joint_types.device
    )
    return constraint_offsets_wp, total_constraints


@wp.func
def compute_joint_kinematics(
    body_q: wp.transform,
    joint_X: wp.transform,
    body_com: wp.vec3,
) -> wp.vec3:
    X_wj = body_q * joint_X
    joint_pos_world = wp.transform_get_translation(X_wj)
    body_com_world = wp.transform_point(body_q, body_com)
    return joint_pos_world - body_com_world


@wp.func
def formulate_translational_constraints_in_world(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    parent_idx: wp.int32,
    child_idx: wp.int32,
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
):
    """
    Assembles the 3 common translational constraints with respect to the world's xyz
    directly into the output array.
    Used in: REVOLUTE, BALL, FIXED joints.

    This function assumes it is called first (before `formulate_rotational_constraints_in_world`,
    for example) because no start_subindex is passed as its input argument.
    """
    c = JointConstraintData()
    c.is_active = True
    c.parent_idx = parent_idx
    c.child_idx = child_idx

    # Constraint 0: X translation
    c.J_child = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, r_c[2], -r_c[1])
    c.J_parent = wp.spatial_vector(-1.0, 0.0, 0.0, 0.0, -r_p[2], r_p[1])
    c.value = c_pos.x
    joint_constraints[start_index + 0] = c

    # Constraint 1: Y translation
    c.J_child = wp.spatial_vector(0.0, 1.0, 0.0, -r_c[2], 0.0, r_c[0])
    c.J_parent = wp.spatial_vector(0.0, -1.0, 0.0, r_p[2], 0.0, -r_p[0])
    c.value = c_pos.y
    joint_constraints[start_index + 1] = c

    # Constraint 2: Z translation
    c.J_child = wp.spatial_vector(0.0, 0.0, 1.0, r_c[1], -r_c[0], 0.0)
    c.J_parent = wp.spatial_vector(0.0, 0.0, -1.0, -r_p[1], r_p[0], 0.0)
    c.value = c_pos.z
    joint_constraints[start_index + 2] = c

@wp.func
def formulate_rotational_constraints_in_world(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    start_subindex: wp.int32,
    parent_idx: wp.int32,
    child_idx: wp.int32,
    q_wc_rot: wp.quat,
    q_wp_rot: wp.quat,
):
    """
    Assembles 3 common rotational constraints with respect to the joint->world transformations
    of both the parent and the child via the difference in their quaternion rotations.
    Used in: FIXED, PRISMATIC joints.
    """
    c = JointConstraintData()
    c.is_active = True
    c.parent_idx = parent_idx
    c.child_idx = child_idx

    # relative rotation between parent's joint and child's joint in quaternions
    q_rel = wp.quat_inverse(q_wp_rot) * q_wc_rot
    q_err = 2.0*wp.vec3(q_rel[0], q_rel[1], q_rel[2])

    # Rotational constraint X
    c.value = q_err[0]
    c.J_parent = wp.spatial_vector(0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    c.J_child = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    joint_constraints[start_index + start_subindex + 0] = c

    # Rotational constraint Y
    c.value = q_err[1]
    c.J_parent = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    c.J_child = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    joint_constraints[start_index + start_subindex + 1] = c

    # Rotational constraint Z
    c.value = q_err[2]
    c.J_parent = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    c.J_child = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    joint_constraints[start_index + start_subindex + 2] = c

@wp.func
def formulate_prismatic_constraints(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    # kinematics and other data
    parent_idx: wp.int32,
    child_idx: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_idx: wp.int32,
    joint_axis: wp.array(dtype=wp.vec3),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_wc_rot: wp.quat,
    q_wp_rot: wp.quat,
):
    """Assembles all 5 constraints for a Prismatic joint."""
    # Retrieve the joint axis
    axis_start = joint_qd_start[joint_idx]
    axis = joint_axis[axis_start]
    axis_p_w = wp.quat_rotate(q_wp_rot, axis)

    # Create axes orthogonal to the joint axis
    b1_w, b2_w = orthogonal_basis(axis_p_w)

    # Reusable JointConstraintData struct
    c = JointConstraintData()
    c.is_active = True
    c.parent_idx = parent_idx
    c.child_idx = child_idx

    # Translational constraint 1
    # Source: https://danielchappuis.ch/download/ConstraintsDerivationRigidBody3D.pdf
    c.value = wp.dot(c_pos, b1_w)     # projection of c_pos onto b1_c_w
    b1_J_w = -wp.cross((r_p + c_pos), b1_w)
    c.J_parent = wp.spatial_vector(-b1_w, b1_J_w)
    c.J_child = wp.spatial_vector(b1_w, wp.cross(r_c, b1_w))
    joint_constraints[start_index + 0] = c

    # Translational constraint 2
    c.value = wp.dot(c_pos, b2_w)     # projection of c_pos onto b2_c_w
    b2_J_w = -wp.cross((r_p + c_pos), b2_w)
    c.J_parent = wp.spatial_vector(-b2_w, b2_J_w)
    c.J_child = wp.spatial_vector(b2_w, wp.cross(r_c, b2_w))
    joint_constraints[start_index + 1] = c

    # Rotational constraints
    formulate_rotational_constraints_in_world(
        joint_constraints,
        start_index,
        2,
        parent_idx,
        child_idx,
        q_wc_rot,
        q_wp_rot
    )

@wp.func
def formulate_revolute_constraints(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    # kinematics and other data
    parent_idx: wp.int32,
    child_idx: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_idx: wp.int32,
    joint_axis: wp.array(dtype=wp.vec3),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_wc_rot: wp.quat,
    q_wp_rot: wp.quat,
):
    """Assembles all 5 constraints for a Revolute joint."""
    # First, assemble the 3 translational constraints (ball-socket part)
    formulate_translational_constraints_in_world(
        joint_constraints,
        start_index,
        parent_idx,
        child_idx,
        c_pos,
        r_c,
        r_p,
    )

    # Now, assemble the 2 rotational constraints (swing part)
    c = JointConstraintData()
    c.is_active = True
    c.parent_idx = parent_idx
    c.child_idx = child_idx

    axis_start = joint_qd_start[joint_idx]
    axis = joint_axis[axis_start]
    axis_p_w = wp.quat_rotate(q_wp_rot, axis)
    b1_local, b2_local = orthogonal_basis(axis)

    # Constraint 3: First rotational constraint
    b1_c_w = wp.quat_rotate(q_wc_rot, b1_local)
    c.value = wp.dot(axis_p_w, b1_c_w)
    b1_x_axis = wp.cross(axis_p_w, b1_c_w)
    c.J_child = wp.spatial_vector(wp.vec3(), -b1_x_axis)
    c.J_parent = wp.spatial_vector(wp.vec3(), b1_x_axis)
    joint_constraints[start_index + 3] = c

    # Constraint 4: Second rotational constraint
    b2_c_w = wp.quat_rotate(q_wc_rot, b2_local)
    c.value = wp.dot(axis_p_w, b2_c_w)
    b2_x_axis = wp.cross(axis_p_w, b2_c_w)
    c.J_child = wp.spatial_vector(wp.vec3(), -b2_x_axis)
    c.J_parent = wp.spatial_vector(wp.vec3(), b2_x_axis)
    joint_constraints[start_index + 4] = c


@wp.func
def formulate_ball_constraints(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    # kinematics and other data
    parent_idx: wp.int32,
    child_idx: wp.int32,
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
):
    """Assembles all 3 constraints for a Ball (Spherical) joint."""
    formulate_translational_constraints_in_world(
        joint_constraints,
        start_index,
        parent_idx,
        child_idx,
        c_pos,
        r_c,
        r_p,
    )

@wp.func
def formulate_fixed_constraints(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    # kinematics and other data
    parent_idx: wp.int32,
    child_idx: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_idx: wp.int32,
    joint_axis: wp.array(dtype=wp.vec3),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_wc_rot: wp.quat,
    q_wp_rot: wp.quat,
):
    """Assembles all 6 constraints for a Fixed joint."""
    # First, assemble the 3 translational constraints
    formulate_translational_constraints_in_world(
        joint_constraints,
        start_index,
        parent_idx,
        child_idx,
        c_pos,
        r_c,
        r_p,
    )
 
    # The, assemble the 3 rotational constraints
    formulate_rotational_constraints_in_world(
        joint_constraints,
        start_index,
        3,
        parent_idx,
        child_idx,
        q_wc_rot,
        q_wp_rot
    )

@wp.func
def formulate_free_constraints(
    joint_constraints: wp.array(dtype=JointConstraintData),
    start_index: wp.int32,
    # kinematics and other data
    parent_idx: wp.int32,
    child_idx: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_idx: wp.int32,
    joint_axis: wp.array(dtype=wp.vec3),
    c_pos: wp.vec3,
    r_c: wp.vec3,
    r_p: wp.vec3,
    q_wc_rot: wp.quat,
    q_wp_rot: wp.quat,
):
    """ Free joint has no constraint """
    pass

@wp.kernel
def joint_constraint_data_kernel(
    # --- Inputs ---
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
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    # Precomputed offsets
    constraint_offsets: wp.array(dtype=wp.int32),
    # --- Output ---
    joint_constraints: wp.array(dtype=JointConstraintData),
):
    joint_idx = wp.tid()
    j_type = joint_type[joint_idx]

    # Early exit for disabled or unsupported joints
    if joint_enabled[joint_idx] == 0 or (
        j_type != JointType.REVOLUTE
        and j_type != JointType.FIXED
        and j_type != JointType.BALL 
        and j_type != JointType.PRISMATIC
        ):
        return

    child_idx = joint_child[joint_idx]
    if child_idx < 0:
        return

    # --- Common Calculations ---
    parent_idx = joint_parent[joint_idx]
    start_index = constraint_offsets[joint_idx]

    # Kinematics
    body_q_c = body_q[child_idx]
    body_q_p = wp.transform_identity()
    if parent_idx >= 0:
        body_q_p = body_q[parent_idx]

    child_com = body_com[child_idx]
    parent_com = wp.vec3()
    if parent_idx >= 0:
        parent_com = body_com[parent_idx]

    r_wc = compute_joint_kinematics(body_q_c, joint_X_c[joint_idx], child_com)
    r_wp = compute_joint_kinematics(body_q_p, joint_X_p[joint_idx], parent_com)

    X_wjc = body_q_c * joint_X_c[joint_idx]
    X_wjp = body_q_p * joint_X_p[joint_idx]

    c_pos = wp.transform_get_translation(X_wjc) - wp.transform_get_translation(X_wjp)
    q_wc_rot = wp.transform_get_rotation(X_wjc)
    q_wp_rot = wp.transform_get_rotation(X_wjp)

    # --- Dispatch to the correct assembly function ---
    if j_type == JointType.REVOLUTE:
        #wp.printf("Axion's Revolute joint used \n")
        formulate_revolute_constraints(
            joint_constraints,
            start_index,
            parent_idx,
            child_idx,
            joint_qd_start,
            joint_idx,
            joint_axis,
            c_pos,
            r_wc,
            r_wp,
            q_wc_rot,
            q_wp_rot,
        )
    elif j_type == JointType.BALL:
        #wp.printf("Axion's Ball joint used \n")
        formulate_ball_constraints(
            joint_constraints,
            start_index,
            parent_idx,
            child_idx,
            c_pos,
            r_wc,
            r_wp,
        )
    elif j_type == JointType.PRISMATIC:
        #wp.printf("Axion's Prismatic joint used \n")
        formulate_prismatic_constraints(
            joint_constraints,
            start_index,
            parent_idx,
            child_idx,
            joint_qd_start,
            joint_idx,
            joint_axis,
            joint_limit_lower,
            joint_limit_upper,
            c_pos,
            r_wc,
            r_wp,
            q_wc_rot,
            q_wp_rot,
        )
    elif j_type == JointType.FIXED:
        #wp.printf("Axion's Fixed joint used \n")
        formulate_fixed_constraints(
            joint_constraints,
            start_index,
            parent_idx,
            child_idx,
            joint_qd_start,
            joint_idx,
            joint_axis,
            c_pos,
            r_wc,
            r_wp,
            q_wc_rot,
            q_wp_rot,
        )