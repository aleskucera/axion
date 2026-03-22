from typing import Dict, Tuple
import torch
from torch_geometric.data import HeteroData


NODE_FEATURE_DIMS = {"object": 22, "contact_point": 0, "floor": 0}
EDGE_FEATURE_DIMS = {
    ("object", "inter_object", "contact_point"): 4,
    ("contact_point", "inter_object", "object"): 4,
    ("floor", "inter_object", "contact_point"): 4,
    ("contact_point", "contact", "contact_point"): 5,
    ("object", "fixed_joint", "object"): 12,
    ("object", "revolute_joint", "object"): 8,
    ("object", "prismatic_joint", "object"): 8,
    ("floor", "fixed_joint", "object"): 12,
    ("floor", "revolute_joint", "object"): 8,
    ("floor", "prismatic_joint", "object"): 8,
}
OUTPUT_FEATURE_DIMS = {
    "object": 6,
    # ("contact_point", "contact", "contact_point"): 1,
    # ("object", "fixed_joint", "object"): 3,
    # ("object", "revolute_joint", "object"): 2,
    # ("object", "prismatic_joint", "object"): 2,
    # ("floor", "fixed_joint", "object"): 3,
    # ("floor", "revolute_joint", "object"): 2,
    # ("floor", "prismatic_joint", "object"): 2,
}


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
    num_bodies: int,
    device: torch.device,
    shape_body: torch.Tensor,
    body_vel_next: torch.Tensor | None = None,
    world_indices: torch.Tensor | None = None,
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
        num_bodies: number of bodies per world
        device: torch device
        shape_body: [W, num_shapes] mapping from shape index to body index (-1 for floor)
        body_vel_next: optional [W, B, 6] next body velocities (used as targets during training)
        world_indices: optional [W] original world indices for .world attributes
                      (if None, use 0..W-1)
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
    )
    add_joints(
        graph,
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
    x_object = torch.cat([body_vel, body_mass, ext_force, rot_inertia], dim=2).float()

    # Add to graph
    graph["object"].x = x_object.reshape(W * B, NODE_FEATURE_DIMS["object"])
    if body_vel_next is not None:
        graph["object"].y = body_vel_next.reshape(W * B, OUTPUT_FEATURE_DIMS["object"])
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

    tol = 1e-3
    grid = torch.round(point_global / tol).to(torch.long)
    keys = torch.zeros((W, C, 2, 5), dtype=torch.long, device=device)
    keys[:, :, :, 0] = world_indices[:, None, None]
    keys[:, :, :, 1] = body_idx
    keys[:, :, :, 2:] = grid
    valid_jk = valid[:, :, None].expand(W, C, 2)
    valid_pos = torch.where(valid_jk.reshape(-1))[0]
    valid_keys = keys.reshape(-1, 5)[valid_pos]
    unique_keys, inverse = torch.unique(valid_keys, dim=0, return_inverse=True)
    cp_node_idx_flat = torch.full((W * C * 2,), -1, dtype=torch.long, device=device)
    cp_node_idx_flat[valid_pos] = inverse
    cp_node_idx = cp_node_idx_flat.reshape(W, C, 2)
    cp_worlds = unique_keys[:, 0]
    total_cp = len(unique_keys)

    normals = torch.stack([contact_normal, -contact_normal], dim=2)
    thickness = torch.stack([contact_thickness0, contact_thickness1], dim=2)
    point_adj = point_global - thickness[..., None] * normals
    com_gathered = body_com[wi2, body_idx]
    com_floor = torch.zeros_like(com_gathered)
    com_gathered = torch.where(body_idx_is_object[..., None], com_gathered, com_floor)
    com_world = (
        torch.einsum("bij,bj->bi", R_flat, com_gathered.reshape(-1, 3)) + trans_flat
    ).reshape(W, C, 2, 3)
    lever = point_adj - com_world
    lever_norm = torch.norm(lever, dim=3, keepdim=True)
    edge_attrs_oc = torch.cat([lever, lever_norm], dim=3)
    edge_attrs_co = torch.cat([-lever, lever_norm], dim=3)

    valid3 = valid[:, :, None]
    valid_obj = valid3 & body_idx_is_object
    valid_floor = valid3 & body_idx_is_floor

    world_arr3 = world_indices.to(torch.long)[:, None, None].expand(W, C, 2)
    global_floor_idx = torch.arange(W, device=device, dtype=torch.long)[:, None, None].expand(
        W, C, 2
    )
    global_obj_idx = (
        torch.arange(W, device=device, dtype=torch.long)[:, None, None] * num_bodies + body_idx
    )

    # Object - Contact point
    idx_oc = torch.stack([global_obj_idx[valid_obj], cp_node_idx[valid_obj]], dim=1)
    attr_oc = edge_attrs_oc[valid_obj]
    world_oc = world_arr3[valid_obj]

    # Contact point - Object
    idx_co = torch.stack([cp_node_idx[valid_obj], global_obj_idx[valid_obj]], dim=1)
    attr_co = edge_attrs_co[valid_obj]

    # Floor - Contact point
    idx_fc = torch.stack([global_floor_idx[valid_floor], cp_node_idx[valid_floor]], dim=1)
    attr_fc = edge_attrs_oc[valid_floor]
    world_fc = world_arr3[valid_floor]

    # Contact point - Contact point
    mu0 = shape_material_mu[wi1, contact_shape0]
    mu1 = shape_material_mu[wi1, contact_shape1]
    mu_avg = (mu0 + mu1) / 2
    dist = torch.einsum("wci,wci->wc", contact_normal, point_adj[:, :, 0] - point_adj[:, :, 1])
    cc_extra = torch.stack([dist, mu_avg], dim=2)
    cc_attr_fwd = torch.cat([contact_normal, cc_extra], dim=2)
    cc_attr_bwd = torch.cat([-contact_normal, cc_extra], dim=2)

    cp0 = cp_node_idx[:, :, 0]
    cp1 = cp_node_idx[:, :, 1]
    world_cc_1d = world_indices.to(torch.long)[:, None].expand(W, C)[valid]
    idx_cc = torch.cat(
        [
            torch.stack([cp0[valid], cp1[valid]], dim=1),
            torch.stack([cp1[valid], cp0[valid]], dim=1),
        ]
    )
    attr_cc = torch.cat([cc_attr_fwd[valid], cc_attr_bwd[valid]])
    world_cc = torch.cat([world_cc_1d, world_cc_1d])

    # Add to graph
    graph["contact_point"].x = torch.zeros(
        (total_cp, NODE_FEATURE_DIMS["contact_point"]), dtype=torch.float32, device=device
    )
    graph["contact_point"].world = cp_worlds.to(torch.long)
    assign_edge_data(
        graph,
        ("object", "inter_object", "contact_point"),
        idx_oc,
        attr_oc,
        world=world_oc,
        device=device,
    )
    assign_edge_data(
        graph,
        ("contact_point", "inter_object", "object"),
        idx_co,
        attr_co,
        world=world_oc,
        device=device,
    )
    assign_edge_data(
        graph,
        ("floor", "inter_object", "contact_point"),
        idx_fc,
        attr_fc,
        world=world_fc,
        device=device,
    )
    assign_edge_data(
        graph,
        ("contact_point", "contact", "contact_point"),
        idx_cc,
        attr_cc,
        world=world_cc,
        device=device,
    )


def add_joints(
    graph: HeteroData,
) -> None:
    pass


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
