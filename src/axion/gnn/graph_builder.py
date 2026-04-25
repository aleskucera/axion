from typing import Dict, Tuple
import torch
from torch_geometric.data import HeteroData
from newton import JointType


NODE_FEATURE_DIMS = {"object": 16, "floor": 0}
EDGE_FEATURE_DIMS = {
    ("object", "contact", "object"): 11,
    ("floor", "contact", "object"): 11,
    ("object", "fixed_joint", "object"): 18,
    ("object", "revolute_joint", "object"): 14,
    ("object", "prismatic_joint", "object"): 14,
    ("object", "ball_joint", "object"): 11,
    ("floor", "fixed_joint", "object"): 18,
    ("floor", "revolute_joint", "object"): 14,
    ("floor", "prismatic_joint", "object"): 14,
    ("floor", "ball_joint", "object"): 11,
}
OUTPUT_FEATURE_DIMS = {
    "object": 6,
}

JOINT_STR_TO_INT = {
    "prismatic_joint": JointType.PRISMATIC.value,
    "revolute_joint": JointType.REVOLUTE.value,
    "ball_joint": JointType.BALL.value,
    "fixed_joint": JointType.FIXED.value,
}


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions q1 * q2, both in [x, y, z, w] format."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    original_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)

    x, y, z, w = q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.empty((q_flat.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return R.reshape(original_shape + (3, 3))


def transform_points(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot_matrix(pose[..., 3:])
    rotated = torch.einsum("...ij,...j->...i", R, points)
    return rotated + pose[..., :3]


def build_graph(
    body_vel_prev: torch.Tensor,
    body_mass: torch.Tensor,
    ext_force: torch.Tensor,
    body_pose_prev: torch.Tensor,
    body_inertia: torch.Tensor,
    body_com: torch.Tensor,
    contact_count: torch.Tensor,
    contact_point0: torch.Tensor,
    contact_point1: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_shape0: torch.Tensor,
    contact_shape1: torch.Tensor,
    shape_material_mu: torch.Tensor,
    contact_thickness0: torch.Tensor,
    contact_thickness1: torch.Tensor,
    joint_type: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_axis: torch.Tensor,
    joint_qd_start: torch.Tensor,
    joint_enabled: torch.Tensor,
    joint_compliance: torch.Tensor,
    num_joints: int,
    num_bodies: int,
    device: torch.device,
    shape_body: torch.Tensor,
    body_vel_next: torch.Tensor | None = None,
    world_indices: torch.Tensor | None = None,
    contact_dist_threshold: float | None = None,
) -> HeteroData:
    """Build complete HeteroData graph from batched state and contact data.
    Args:
        body_vel_prev: [W, B, 6] current body velocities (used for features)
        body_mass: [W, B, 1] body masses
        ext_force: [W, B, 6] external forces
        body_pose_prev: [W, B, 7] body poses
        body_inertia: [W, B, 3, 3] body inertias
        body_com: [W, B, 3] body centers of mass
        contact_count: [W] contacts per world
        contact_point0/1: [W, C] contact points in local frame
        contact_normal: [W, C] contact normals
        contact_shape0/1: [W, C] shape indices
        contact_thickness0/1: [W, C] contact thicknesses
        joint_type: [W, J] joint type indices
        joint_parent: [W, J] parent body index (-1 for floor/world)
        joint_child: [W, J] child body index
        joint_X_p: [W, J, 7] parent frame transform (position + quaternion)
        joint_X_c: [W, J, 7] child frame transform (position + quaternion)
        joint_axis: [W, Jdof, 3] joint axis vectors (indexed by DOF, not by joint)
        joint_qd_start: [W, J] starting DOF index in joint_axis for each joint
        joint_enabled: [W, J] whether each joint is active
        joint_compliance: [W, J] per-joint compliance override
        num_joints: number of joints per world
        num_bodies: number of bodies per world
        device: torch device
        shape_body: [W, num_shapes] mapping from shape index to body index (-1 for floor)
        body_vel_next: optional [W, B, 6] next body velocities (used as targets during training)
        world_indices: optional [W] original world indices for .world attributes
                      (if None, use 0..W-1)
        contact_dist_threshold: optional float maximal distance of contact detection
    Returns:
        HeteroData graph with world batch structure
    """
    num_worlds = body_vel_prev.shape[0]
    if world_indices is None:
        world_indices = torch.arange(num_worlds, dtype=torch.long, device=device)

    graph = HeteroData()
    add_objects(
        graph,
        body_vel_prev,
        body_mass,
        ext_force,
        body_inertia,
        body_pose_prev,
        num_bodies,
        world_indices,
        body_vel_next,
    )
    add_floor(
        graph,
        world_indices,
        device,
    )
    add_contacts(
        graph,
        contact_count,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_shape0,
        contact_shape1,
        shape_body,
        shape_material_mu,
        contact_thickness0,
        contact_thickness1,
        body_pose_prev,
        body_com,
        num_bodies,
        world_indices,
        device,
        contact_dist_threshold,
    )
    add_joints(
        graph,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_qd_start,
        joint_enabled,
        joint_compliance,
        body_pose_prev,
        body_com,
        num_bodies,
        num_joints,
        world_indices,
        device,
    )
    return graph


def add_objects(
    graph: HeteroData,
    body_vel: torch.Tensor,
    body_mass: torch.Tensor,
    ext_force: torch.Tensor,
    body_inertia: torch.Tensor,
    body_pose: torch.Tensor,
    num_bodies: int,
    world_indices: torch.Tensor,
    body_vel_next: torch.Tensor | None = None,
) -> None:
    """Build object node features from body state.
    Args:
        graph: PyG Hetero data
        body_vel: [W, B, 6] current body velocities (used for features)
        body_mass: [W, B, 1] body masses
        ext_force: [W, B, 6] external forces
        body_inertia: [W, B, 3, 3] body inertias
        body_pose: [W, B, 7] body poses (position + quaternion)
        num_bodies: number of bodies per world
        world_indices: [W] id of world
        body_vel_next: optional [W, B, 6] next body velocities (targets during training)
    """
    W = body_vel.shape[0]
    B = num_bodies
    R = quat_to_rot_matrix(body_pose[..., 3:])
    rot_inertia = torch.einsum("wbij,wbjk,wblk->wbil", R, body_inertia, R).reshape(W, B, 9)
    x_object = torch.cat([body_vel, body_mass, rot_inertia], dim=2).float()

    # Add to graph
    graph["object"].x = x_object.reshape(W * B, NODE_FEATURE_DIMS["object"])
    if body_vel_next is not None:
        acceleration = body_vel_next - body_vel
        graph["object"].y = acceleration.reshape(W * B, OUTPUT_FEATURE_DIMS["object"])
    graph["object"].world = torch.repeat_interleave(world_indices, B)


def add_floor(graph: HeteroData, world_indices: torch.Tensor, device: torch.device):
    """Build floor node features.
    Args:
        graph: PyG Hetero data
        world_indices: [W] id of world
        device: torch device
    """
    W = world_indices.shape[0]

    # Add to graph
    graph["floor"].x = torch.zeros(
        (W, NODE_FEATURE_DIMS["floor"]), dtype=torch.float32, device=device
    )
    graph["floor"].world = world_indices


def add_contacts(
    graph: HeteroData,
    contact_count: torch.Tensor,
    contact_point0: torch.Tensor,
    contact_point1: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_shape0: torch.Tensor,
    contact_shape1: torch.Tensor,
    shape_body: torch.Tensor,
    shape_material_mu: torch.Tensor,
    contact_thickness0: torch.Tensor,
    contact_thickness1: torch.Tensor,
    body_pose: torch.Tensor,
    body_com: torch.Tensor,
    num_bodies: int,
    world_indices: torch.Tensor,
    device: torch.device,
    contact_dist_threshold: float | None = None,
) -> None:

    W = world_indices.shape[0]
    C = contact_shape0.shape[1]
    wi1 = torch.arange(W, device=device)[:, None]
    wi2 = torch.arange(W, device=device)[:, None, None]
    valid = torch.arange(C, device=device)[None, :] < contact_count[:, None]

    def _get_body_idx(shape_idx_arr):
        body = shape_body[wi1, shape_idx_arr].to(torch.long)
        return torch.where(
            shape_idx_arr < 0, torch.tensor(-1, device=device, dtype=torch.long), body
        )

    body_idx = torch.stack([_get_body_idx(contact_shape0), _get_body_idx(contact_shape1)], dim=2)
    body_idx_is_floor = body_idx == -1
    body_idx_is_object = body_idx >= 0

    poses = body_pose[wi2, body_idx]
    identity_pose = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device, dtype=torch.float32)
    poses = torch.where(body_idx_is_floor[..., None], identity_pose, poses)

    local_points = torch.stack([contact_point0, contact_point1], dim=2)
    R = quat_to_rot_matrix(poses[..., 3:])
    R_flat = R.reshape(-1, 3, 3)
    trans_flat = poses[..., :3].reshape(-1, 3)
    pts_flat = local_points.reshape(-1, 3)
    point_global = (torch.einsum("bij,bj->bi", R_flat, pts_flat) + trans_flat).reshape(W, C, 2, 3)

    normals = torch.stack([contact_normal, -contact_normal], dim=2)
    thickness = torch.stack([contact_thickness0, contact_thickness1], dim=2)
    point_adj = point_global - thickness[..., None] * normals

    dist = torch.einsum("wci,wci->wc", contact_normal, point_adj[:, :, 0] - point_adj[:, :, 1])
    if contact_dist_threshold is not None:
        valid = valid & (dist < contact_dist_threshold)

    com_gathered = body_com[wi2, body_idx]
    com_floor = torch.zeros_like(com_gathered)
    com_gathered = torch.where(body_idx_is_object[..., None], com_gathered, com_floor)
    com_world = (
        torch.einsum("bij,bj->bi", R_flat, com_gathered.reshape(-1, 3)) + trans_flat
    ).reshape(W, C, 2, 3)
    lever = point_adj - com_world

    mu0 = shape_material_mu[wi1, contact_shape0]
    mu1 = shape_material_mu[wi1, contact_shape1]
    mu_avg = (mu0 + mu1) / 2

    lever_0 = lever[:, :, 0, :]
    lever_1 = lever[:, :, 1, :]
    rot_jac_0 = torch.cross(lever_0, contact_normal, dim=-1)
    rot_jac_1 = torch.cross(lever_1, -contact_normal, dim=-1)
    cc_extra = torch.stack([dist, mu_avg], dim=2)
    attr_fwd = torch.cat([contact_normal, lever_0, rot_jac_0, cc_extra], dim=2)  # [W, C, 11]
    attr_bwd = torch.cat([-contact_normal, lever_1, rot_jac_1, cc_extra], dim=2)  # [W, C, 11]

    world_arr = world_indices.to(torch.long)[:, None].expand(W, C)
    w_offset = torch.arange(W, device=device, dtype=torch.long)[:, None, None] * num_bodies
    global_obj = w_offset + torch.where(body_idx_is_object, body_idx, torch.zeros_like(body_idx))

    # Object-Object contacts (bidirectional)
    both_obj = valid & body_idx_is_object[:, :, 0] & body_idx_is_object[:, :, 1]
    oo_world_1d = world_arr[both_obj]
    oo_idx = torch.cat(
        [
            torch.stack([global_obj[:, :, 0][both_obj], global_obj[:, :, 1][both_obj]], dim=1),
            torch.stack([global_obj[:, :, 1][both_obj], global_obj[:, :, 0][both_obj]], dim=1),
        ]
    )
    oo_attr = torch.cat([attr_fwd[both_obj], attr_bwd[both_obj]])
    oo_world = torch.cat([oo_world_1d, oo_world_1d])

    assign_edge_data(
        graph, ("object", "contact", "object"), oo_idx, oo_attr, world=oo_world, device=device
    )

    # Floor-Object contacts (floor on side 0 or side 1)
    floor_global = torch.arange(W, device=device, dtype=torch.long)[:, None].expand(W, C)
    floor0 = valid & body_idx_is_floor[:, :, 0] & body_idx_is_object[:, :, 1]
    floor1 = valid & body_idx_is_object[:, :, 0] & body_idx_is_floor[:, :, 1]

    fo_idx_parts, fo_attr_parts, fo_world_parts = [], [], []
    if floor0.any():
        fo_idx_parts.append(torch.stack([floor_global[floor0], global_obj[:, :, 1][floor0]], dim=1))
        fo_attr_parts.append(attr_fwd[floor0])
        fo_world_parts.append(world_arr[floor0])
    if floor1.any():
        fo_idx_parts.append(torch.stack([floor_global[floor1], global_obj[:, :, 0][floor1]], dim=1))
        fo_attr_parts.append(attr_bwd[floor1])
        fo_world_parts.append(world_arr[floor1])

    fo_idx = (
        torch.cat(fo_idx_parts)
        if fo_idx_parts
        else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    fo_attr = (
        torch.cat(fo_attr_parts)
        if fo_attr_parts
        else torch.zeros(
            (0, EDGE_FEATURE_DIMS[("floor", "contact", "object")]),
            dtype=torch.float32,
            device=device,
        )
    )
    fo_world = (
        torch.cat(fo_world_parts)
        if fo_world_parts
        else torch.zeros((0,), dtype=torch.long, device=device)
    )

    assign_edge_data(
        graph, ("floor", "contact", "object"), fo_idx, fo_attr, world=fo_world, device=device
    )


def get_joint_side_features(
    joint_name: str,
    lever: torch.Tensor,
    axis: torch.Tensor,
    quat: torch.Tensor,
) -> torch.Tensor:
    if joint_name == "fixed_joint":
        return torch.cat([lever, axis, quat], dim=-1)
    elif joint_name == "ball_joint":
        return torch.cat([lever, quat], dim=-1)
    else:  # revolute, prismatic
        return torch.cat([lever, axis], dim=-1)


def get_joint_error_features(
    joint_name: str,
    linear_err: torch.Tensor,
    angular_err: torch.Tensor,
    axis_err: torch.Tensor,
) -> torch.Tensor:
    """Constraint violation error for each joint type.
    linear_err:  pos_c_world - pos_p_world  [N, 3]
    angular_err: 2 * Im(q_p^{-1} * q_c)    [N, 3]  (mirrors get_angular_component)
    axis_err:    cross(axis_c, axis_p)       [N, 3]  (mirrors get_revolute_angular_component)
    """
    if joint_name in ("fixed_joint", "prismatic_joint"):
        return torch.cat([linear_err, angular_err], dim=-1)
    elif joint_name == "revolute_joint":
        return torch.cat([linear_err, axis_err], dim=-1)
    else:  # ball_joint
        return linear_err


def add_joints(
    graph: HeteroData,
    joint_type: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_axis: torch.Tensor,
    joint_qd_start: torch.Tensor,
    joint_enabled: torch.Tensor,
    joint_compliance: torch.Tensor,
    body_pose: torch.Tensor,
    body_com: torch.Tensor,
    num_bodies: int,
    num_joints: int,
    world_indices: torch.Tensor,
    device: torch.device,
) -> None:

    # FREE joints are Newton's articulation-tree root joints; skip them
    non_free = joint_type[0] != JointType.FREE.value  # [J], same structure across worlds
    if not non_free.all():
        joint_type = joint_type[:, non_free]
        joint_parent = joint_parent[:, non_free]
        joint_child = joint_child[:, non_free]
        joint_X_p = joint_X_p[:, non_free]
        joint_X_c = joint_X_c[:, non_free]
        joint_qd_start = joint_qd_start[:, non_free]
        joint_enabled = joint_enabled[:, non_free]
        joint_compliance = joint_compliance[:, non_free]
        num_joints = int(non_free.sum())

    W = world_indices.shape[0]
    J = num_joints

    wi = torch.arange(W, device=device)[:, None]
    parent_indices = joint_parent.to(torch.long)
    child_indices = joint_child.to(torch.long)

    identity_pose = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=device, dtype=body_pose.dtype)
    body_pose_p = body_pose[wi, parent_indices]
    body_pose_p = torch.where((parent_indices >= 0)[..., None], body_pose_p, identity_pose)
    body_pose_c = body_pose[wi, child_indices]

    # World-space body rotation matrices
    R_body_p = quat_to_rot_matrix(body_pose_p[..., 3:])  # [W, J, 3, 3]
    R_body_c = quat_to_rot_matrix(body_pose_c[..., 3:])  # [W, J, 3, 3]

    # Body COM in local frame (floor COM = 0)
    com_p_local = body_com[wi, parent_indices]
    com_p_local = torch.where(
        (parent_indices >= 0)[..., None], com_p_local, torch.zeros_like(com_p_local)
    )
    com_c_local = body_com[wi, child_indices]

    # Lever arms in world space: R_body @ (joint_pos_local - com_local)
    # Mirrors: r = pos_joint_world - pos_com_world = R_body @ (joint_pos_local - com_local)
    lever_p = torch.einsum("wjmn,wjn->wjm", R_body_p, joint_X_p[..., :3] - com_p_local)
    feat_lever_p = torch.cat([lever_p, torch.norm(lever_p, dim=-1, keepdim=True)], dim=-1)

    lever_c = torch.einsum("wjmn,wjn->wjm", R_body_c, joint_X_c[..., :3] - com_c_local)
    feat_lever_c = torch.cat([lever_c, torch.norm(lever_c, dim=-1, keepdim=True)], dim=-1)

    # Joint axis in world space: R_body @ R_joint @ axis_local
    # joint_axis is DOF-indexed; use joint_qd_start to look up the axis for each joint
    axis_local = joint_axis[wi, joint_qd_start.to(torch.long)]  # [W, J, 3]
    R_joint_p = quat_to_rot_matrix(joint_X_p[..., 3:])
    axis_p = torch.einsum("wjmn,wjnk,wjk->wjm", R_body_p, R_joint_p, axis_local)
    feat_axis_p = torch.cat([axis_p, torch.ones((W, J, 1), device=device)], dim=-1)

    R_joint_c = quat_to_rot_matrix(joint_X_c[..., 3:])
    axis_c = torch.einsum("wjmn,wjnk,wjk->wjm", R_body_c, R_joint_c, axis_local)
    feat_axis_c = torch.cat([axis_c, torch.ones((W, J, 1), device=device)], dim=-1)

    # Joint frame quaternion in world space: q_body * q_joint
    # Mirrors: X_w = body_q * joint_X_local → q_w = q_body * q_joint
    feat_quat_p = quat_mul(body_pose_p[..., 3:], joint_X_p[..., 3:])
    feat_quat_c = quat_mul(body_pose_c[..., 3:], joint_X_c[..., 3:])

    # Constraint violation errors
    # Linear: world-space anchor position difference (pos_c - pos_p)
    pos_p_world = torch.einsum("wjmn,wjn->wjm", R_body_p, joint_X_p[..., :3]) + body_pose_p[..., :3]
    pos_c_world = torch.einsum("wjmn,wjn->wjm", R_body_c, joint_X_c[..., :3]) + body_pose_c[..., :3]
    linear_err = pos_c_world - pos_p_world  # [W, J, 3]

    # Angular: 2 * Im(q_p^{-1} * q_c), mirrors get_angular_component error formula
    q_p_conj = feat_quat_p * feat_quat_p.new_tensor([-1.0, -1.0, -1.0, 1.0])
    q_rel = quat_mul(q_p_conj, feat_quat_c)
    angular_err_raw = 2.0 * q_rel[..., :3]
    angular_err = torch.where(q_rel[..., 3:] < 0, -angular_err_raw, angular_err_raw)  # [W, J, 3]

    # Revolute axis misalignment: cross(axis_c, axis_p), mirrors get_revolute_angular_component
    axis_err = torch.cross(axis_c, axis_p, dim=-1)  # [W, J, 3]

    world_arr = world_indices[:, None].expand(W, J)

    for joint_name, joint_idx in JOINT_STR_TO_INT.items():
        mask = (joint_type == joint_idx) & joint_enabled

        obj_obj_idx, obj_obj_attr, obj_obj_world = [], [], []
        floor_obj_idx, floor_obj_attr, floor_obj_world = [], [], []

        if mask.any():
            w_idx, j_idx = torch.where(mask)
            p_ids = parent_indices[w_idx, j_idx]
            c_ids = child_indices[w_idx, j_idx]

            feat_p = get_joint_side_features(
                joint_name,
                feat_lever_p[w_idx, j_idx],
                feat_axis_p[w_idx, j_idx],
                feat_quat_p[w_idx, j_idx],
            )
            feat_c = get_joint_side_features(
                joint_name,
                feat_lever_c[w_idx, j_idx],
                feat_axis_c[w_idx, j_idx],
                feat_quat_c[w_idx, j_idx],
            )
            err_all = get_joint_error_features(
                joint_name,
                linear_err[w_idx, j_idx],
                angular_err[w_idx, j_idx],
                axis_err[w_idx, j_idx],
            )

            # Object-Object joints (parent is an object)
            obj_mask = p_ids >= 0
            if obj_mask.any():
                wo, jo = w_idx[obj_mask], j_idx[obj_mask]
                po, co = p_ids[obj_mask], c_ids[obj_mask]
                fp, fc = feat_p[obj_mask], feat_c[obj_mask]
                err_oo = err_all[obj_mask]
                world_oo = world_arr[wo, jo]

                # parent -> child
                idx_pc = torch.stack([wo * num_bodies + po, wo * num_bodies + co], dim=1)
                # child -> parent (reverse)
                idx_cp = torch.stack([wo * num_bodies + co, wo * num_bodies + po], dim=1)

                obj_obj_idx.append(torch.cat([idx_pc, idx_cp]))
                obj_obj_attr.append(
                    torch.cat(
                        [
                            torch.cat([fc, err_oo], dim=-1),  # parent→child: child geometry
                            torch.cat([fp, -err_oo], dim=-1),  # child→parent: parent geometry
                        ]
                    )
                )
                obj_obj_world.append(torch.cat([world_oo, world_oo]))

            # Floor-Object joints (parent is floor)
            floor_mask = p_ids == -1
            if floor_mask.any():
                wf, jf = w_idx[floor_mask], j_idx[floor_mask]
                cf = c_ids[floor_mask]
                fp, fc = feat_p[floor_mask], feat_c[floor_mask]
                err_f = err_all[floor_mask]
                world_f = world_arr[wf, jf]

                # floor -> child object: child geometry so object can compute its constraint response
                floor_obj_idx.append(torch.stack([wf, wf * num_bodies + cf], dim=1))
                floor_obj_attr.append(torch.cat([fc, err_f], dim=-1))
                floor_obj_world.append(world_f)

        dim = EDGE_FEATURE_DIMS[("object", joint_name, "object")]

        # object -> object
        idx = (
            torch.cat(obj_obj_idx)
            if obj_obj_idx
            else torch.zeros((0, 2), dtype=torch.long, device=device)
        )
        attr = torch.cat(obj_obj_attr) if obj_obj_attr else torch.zeros((0, dim), device=device)
        world = (
            torch.cat(obj_obj_world)
            if obj_obj_world
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        assign_edge_data(
            graph, ("object", joint_name, "object"), idx, attr, world=world, device=device
        )

        # floor -> object
        idx = (
            torch.cat(floor_obj_idx)
            if floor_obj_idx
            else torch.zeros((0, 2), dtype=torch.long, device=device)
        )
        attr = torch.cat(floor_obj_attr) if floor_obj_attr else torch.zeros((0, dim), device=device)
        world = (
            torch.cat(floor_obj_world)
            if floor_obj_world
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        assign_edge_data(
            graph, ("floor", joint_name, "object"), idx, attr, world=world, device=device
        )


def assign_edge_data(
    graph: HeteroData,
    edge_type: Tuple[str, str, str],
    indices: torch.Tensor,
    attrs: torch.Tensor,
    preds: torch.Tensor | None = None,
    world: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> None:
    if len(indices) > 0:
        graph[edge_type].edge_index = indices.T
        graph[edge_type].edge_attr = attrs
        if not preds is None:
            graph[edge_type].y = preds
        if not world is None:
            graph[edge_type].world = world.to(torch.long)
    else:
        if device is None:
            device = attrs.device if isinstance(attrs, torch.Tensor) else torch.device("cpu")
        attr_dim = EDGE_FEATURE_DIMS[edge_type]
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        graph[edge_type].edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32, device=device)
        if not preds is None:
            pred_dim = OUTPUT_FEATURE_DIMS[edge_type]
            graph[edge_type].y = torch.zeros((0, pred_dim), dtype=torch.float32, device=device)
        if not world is None:
            graph[edge_type].world = torch.zeros((0,), dtype=torch.long, device=device)
