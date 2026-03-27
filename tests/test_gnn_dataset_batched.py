"""Tests for the world-batched graph construction in AxionDatasetGNN.

Each call to ``construct_graph`` should return one HeteroData graph that
contains *all* simulation worlds as disconnected sub-graphs with correctly
offset edge indices.
"""

import numpy as np
import pytest
import torch
import h5py

from axion.gnn.dataset import AxionDatasetGNN, EDGE_FEATURE_DIMS, NODE_FEATURE_DIMS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_h5(
    num_worlds: int = 2,
    num_bodies: int = 2,
    num_contacts_per_world: int = 1,
    num_steps: int = 11,
    seed: int = 0,
):
    """Return an in-memory h5py.File with deterministic synthetic data.

    Layout mirrors what the real simulation writes:
      - shape 0 → body 0  (object-side of contact)
      - shape 1 → body 1  (object-side of contact)
      - shape 2 → body -1 (floor-side; body_idx resolved to -1 via shape_body=-1)

    Contact layout (when num_contacts_per_world >= 2):
      - contact 0: shape 0 vs shape 1  (object–object)
      - contact 1: shape 0 vs shape 2  (object–floor)
    Each contact has distinct point positions so deduplication keeps all 4 points.
    """
    rng = np.random.default_rng(seed)
    num_shapes = 3  # shape 0, 1 are on bodies; shape 2 is floor

    f = h5py.File("test_mock.h5", "w", driver="core", backing_store=False)

    # dims
    f.create_dataset("dims/num_worlds", data=num_worlds)
    f.create_dataset("dims/body_count", data=num_bodies)

    # model
    f.create_dataset(
        "model/body_mass",
        data=rng.random((num_worlds, num_bodies)).astype(np.float32) + 0.5,
    )
    # body_inertia: (num_worlds, num_bodies, 3, 3) – random symmetric positive-definite
    I_base = np.eye(3, dtype=np.float32)
    body_inertia = np.tile(I_base, (num_worlds, num_bodies, 1, 1))
    body_inertia += rng.random((num_worlds, num_bodies, 3, 3)).astype(np.float32) * 0.1
    f.create_dataset("model/body_inertia", data=body_inertia)

    f.create_dataset(
        "model/body_com",
        data=rng.random((num_worlds, num_bodies, 3)).astype(np.float32),
    )
    # shape_body: shape 0→body 0, shape 1→body 1, shape 2→floor (-1 stored as large int,
    # resolved via resolve_body_indices which checks shape0 >= 0 and returns shape_body[shape0])
    shape_body = np.array([[0, 1, -1]] * num_worlds, dtype=np.int32)
    f.create_dataset("model/shape_body", data=shape_body)
    f.create_dataset(
        "model/shape_material_mu",
        data=np.full((num_worlds, num_shapes), 0.5, dtype=np.float32),
    )

    # data
    f.create_dataset(
        "data/body_vel",
        data=rng.random((num_steps, num_worlds, num_bodies, 6)).astype(np.float32),
    )
    f.create_dataset(
        "data/body_vel_prev",
        data=rng.random((num_steps, num_worlds, num_bodies, 6)).astype(np.float32),
    )
    f.create_dataset(
        "data/ext_force",
        data=np.zeros((num_steps, num_worlds, num_bodies, 6), dtype=np.float32),
    )
    # Identity quaternion [x,y,z,w] = [0,0,0,1]
    poses = np.zeros((num_steps, num_worlds, num_bodies, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    f.create_dataset("data/body_pose", data=poses)
    f.create_dataset("data/body_pose_prev", data=poses)

    # contacts
    f.create_dataset(
        "data/contact_count",
        data=np.full((num_steps, num_worlds), num_contacts_per_world, dtype=np.int32),
    )
    # contact 0: shape 0 (body 0) vs shape 1 (body 1)  — object–object
    # contact 1: shape 0 (body 0) vs shape 2 (floor)   — object–floor
    contact_shape0 = np.zeros((num_steps, num_worlds, num_contacts_per_world), dtype=np.int32)
    contact_shape1 = np.ones((num_steps, num_worlds, num_contacts_per_world), dtype=np.int32)
    if num_contacts_per_world >= 2:
        contact_shape1[:, :, 1] = 2  # shape 2 = floor
    f.create_dataset("data/contact_shape0", data=contact_shape0)
    f.create_dataset("data/contact_shape1", data=contact_shape1)
    # Distinct positions per contact so deduplication produces one node per contact-side
    cp0 = np.zeros((num_steps, num_worlds, num_contacts_per_world, 3), dtype=np.float32)
    cp1 = np.ones((num_steps, num_worlds, num_contacts_per_world, 3), dtype=np.float32)
    if num_contacts_per_world >= 2:
        cp0[:, :, 1, :] = 0.5  # body-0 point for floor contact (distinct from contact 0)
        cp1[:, :, 1, :] = 2.0  # floor point for floor contact
    f.create_dataset("data/contact_point0", data=cp0)
    f.create_dataset("data/contact_point1", data=cp1)
    normals = np.zeros((num_steps, num_worlds, num_contacts_per_world, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    f.create_dataset("data/contact_normal", data=normals)
    f.create_dataset(
        "data/contact_thickness0",
        data=np.full((num_steps, num_worlds, num_contacts_per_world), 0.01, dtype=np.float32),
    )
    f.create_dataset(
        "data/contact_thickness1",
        data=np.full((num_steps, num_worlds, num_contacts_per_world), 0.01, dtype=np.float32),
    )

    return f


def _make_dataset() -> AxionDatasetGNN:
    """Instantiate AxionDatasetGNN without triggering InMemoryDataset.__init__."""
    return object.__new__(AxionDatasetGNN)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

NUM_WORLDS = 3
NUM_BODIES = 2
NUM_CONTACTS = 2  # contact 0: object–object, contact 1: object–floor → 4 contact points per world


@pytest.fixture(scope="module")
def batched_graph():
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        return ds.construct_graph(f, step=0)[0]


@pytest.fixture(scope="module")
def per_world_features():
    """Reference: per-world (x, y) tensors computed independently per world."""
    from scipy.spatial.transform import Rotation

    results = []
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        step = 0
        for w in range(NUM_WORLDS):
            body_vel = f["data/body_vel"][step][w]
            body_vel_prev = f["data/body_vel_prev"][step][w]
            body_mass = f["model/body_mass"][w]
            ext_force = f["data/ext_force"][step][w]
            body_pose_prev = f["data/body_pose_prev"][step][w]
            body_inertia = f["model/body_inertia"][w]

            toi_list = []
            for b in range(NUM_BODIES):
                R = Rotation.from_quat(body_pose_prev[b, 3:]).as_matrix()
                toi_list.append((R @ body_inertia[b] @ R.T).flatten())
            toi_arr = np.stack(toi_list)

            nodes = np.concatenate(
                [body_vel_prev, body_mass[:, np.newaxis], ext_force, toi_arr], axis=1
            )
            # Target is acceleration (delta-v), not absolute next velocity
            acceleration = body_vel - body_vel_prev
            results.append(
                (
                    torch.tensor(nodes, dtype=torch.float32),
                    torch.tensor(acceleration, dtype=torch.float32),
                )
            )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Node count tests
# ──────────────────────────────────────────────────────────────────────────────


def test_object_node_count(batched_graph):
    assert batched_graph["object"].num_nodes == NUM_WORLDS * NUM_BODIES


def test_floor_node_count(batched_graph):
    assert batched_graph["floor"].num_nodes == NUM_WORLDS


def test_contact_point_node_count(batched_graph):
    # Each object–object contact yields 2 contact points (one per side)
    expected_cp_per_world = NUM_CONTACTS * 2
    assert batched_graph["contact_point"].num_nodes == NUM_WORLDS * expected_cp_per_world


# ──────────────────────────────────────────────────────────────────────────────
# Feature correctness tests
# ──────────────────────────────────────────────────────────────────────────────


def test_object_features_match_per_world(batched_graph, per_world_features):
    """Slice [w*B:(w+1)*B] of the batched object.x must equal world-w individual features."""
    B = NUM_BODIES
    for w, (x_ref, _) in enumerate(per_world_features):
        x_batched = batched_graph["object"].x[w * B : (w + 1) * B]
        assert torch.allclose(x_batched, x_ref), f"object.x mismatch for world {w}"


def test_object_targets_match_per_world(batched_graph, per_world_features):
    B = NUM_BODIES
    for w, (_, y_ref) in enumerate(per_world_features):
        y_batched = batched_graph["object"].y[w * B : (w + 1) * B]
        assert torch.allclose(y_batched, y_ref), f"object.y mismatch for world {w}"


def test_object_feature_dim(batched_graph):
    assert batched_graph["object"].x.shape[1] == NODE_FEATURE_DIMS["object"]


def test_floor_feature_dim(batched_graph):
    assert batched_graph["floor"].x.shape == (NUM_WORLDS, 0)


def test_contact_point_feature_dim(batched_graph):
    n_cp = batched_graph["contact_point"].num_nodes
    assert batched_graph["contact_point"].x.shape == (n_cp, 0)


# ──────────────────────────────────────────────────────────────────────────────
# World-membership attribute tests
# ──────────────────────────────────────────────────────────────────────────────


def test_object_world_attr(batched_graph):
    expected = torch.repeat_interleave(torch.arange(NUM_WORLDS), NUM_BODIES)
    assert torch.equal(batched_graph["object"].world, expected)


def test_floor_world_attr(batched_graph):
    assert torch.equal(batched_graph["floor"].world, torch.arange(NUM_WORLDS))


def test_contact_point_world_attr(batched_graph):
    cp_per_world = NUM_CONTACTS * 2
    expected = torch.repeat_interleave(torch.arange(NUM_WORLDS), cp_per_world)
    assert torch.equal(batched_graph["contact_point"].world, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Edge-index correctness tests
# ──────────────────────────────────────────────────────────────────────────────


def _world_of(graph, node_type, indices):
    return graph[node_type].world[indices]


@pytest.mark.parametrize(
    "edge_type,src_type,dst_type",
    [
        (("object", "inter_object", "contact_point"), "object", "contact_point"),
        (("contact_point", "inter_object", "object"), "contact_point", "object"),
        (("floor", "inter_object", "contact_point"), "floor", "contact_point"),
        (("contact_point", "contact", "contact_point"), "contact_point", "contact_point"),
    ],
)
def test_no_cross_world_edges(batched_graph, edge_type, src_type, dst_type):
    """Every edge must connect nodes that belong to the same world."""
    if batched_graph[edge_type].num_edges == 0:
        pytest.skip("no edges of this type in synthetic data")

    src_idx = batched_graph[edge_type].edge_index[0]
    dst_idx = batched_graph[edge_type].edge_index[1]

    src_worlds = _world_of(batched_graph, src_type, src_idx)
    dst_worlds = _world_of(batched_graph, dst_type, dst_idx)

    assert torch.all(src_worlds == dst_worlds), (
        f"Cross-world edge found in {edge_type}. "
        f"Mismatched worlds: {(src_worlds != dst_worlds).nonzero()}"
    )


@pytest.mark.parametrize(
    "edge_type,src_type,dst_type",
    [
        (("object", "inter_object", "contact_point"), "object", "contact_point"),
        (("contact_point", "inter_object", "object"), "contact_point", "object"),
        (("floor", "inter_object", "contact_point"), "floor", "contact_point"),
        (("contact_point", "contact", "contact_point"), "contact_point", "contact_point"),
    ],
)
def test_edge_indices_in_bounds(batched_graph, edge_type, src_type, dst_type):
    if batched_graph[edge_type].num_edges == 0:
        pytest.skip("no edges of this type in synthetic data")

    n_src = batched_graph[src_type].num_nodes
    n_dst = batched_graph[dst_type].num_nodes
    src_idx = batched_graph[edge_type].edge_index[0]
    dst_idx = batched_graph[edge_type].edge_index[1]

    assert src_idx.min() >= 0 and src_idx.max() < n_src, f"src index out of bounds for {edge_type}"
    assert dst_idx.min() >= 0 and dst_idx.max() < n_dst, f"dst index out of bounds for {edge_type}"


# ──────────────────────────────────────────────────────────────────────────────
# Edge attribute dimension tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "edge_type",
    [
        ("object", "inter_object", "contact_point"),
        ("contact_point", "inter_object", "object"),
        ("floor", "inter_object", "contact_point"),
        ("contact_point", "contact", "contact_point"),
    ],
)
def test_edge_attr_dim(batched_graph, edge_type):
    expected_dim = EDGE_FEATURE_DIMS[edge_type]
    actual = batched_graph[edge_type].edge_attr
    assert (
        actual.shape[1] == expected_dim
    ), f"edge_attr dim mismatch for {edge_type}: got {actual.shape[1]}, expected {expected_dim}"


# ──────────────────────────────────────────────────────────────────────────────
# No-contact edge case
# ──────────────────────────────────────────────────────────────────────────────


def test_zero_contacts():
    """With no contacts, contact_point node count should be 0 and edge tensors empty."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=2, num_bodies=2, num_contacts_per_world=0) as f:
        graph = ds.construct_graph(f, step=0)[0]

    assert graph["contact_point"].num_nodes == 0
    assert graph[("object", "inter_object", "contact_point")].num_edges == 0
    assert graph[("contact_point", "contact", "contact_point")].num_edges == 0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step consistency
# ──────────────────────────────────────────────────────────────────────────────


def test_different_steps_have_different_features():
    """Steps 0 and 10 should have different object velocities (random synthetic data)."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        g0 = ds.construct_graph(f, step=0)[0]
        g10 = ds.construct_graph(f, step=10)[0]

    assert not torch.equal(g0["object"].x, g10["object"].x)


# ──────────────────────────────────────────────────────────────────────────────
# World filtering tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_h5_with_bad_world(bad_world: int, bad_value: float, num_worlds: int = 3):
    """Return a mock h5 where one world has a corrupted body_vel."""
    f = _make_mock_h5(
        num_worlds=num_worlds, num_bodies=NUM_BODIES, num_contacts_per_world=NUM_CONTACTS
    )
    vel = f["data/body_vel"][:]
    vel[:, bad_world, :, :] = bad_value
    del f["data/body_vel"]
    f.create_dataset("data/body_vel", data=vel)
    return f


def test_nan_world_is_filtered():
    """A world with NaN velocities must be excluded; remaining worlds are kept."""
    ds = _make_dataset()
    bad_world = 1
    with _make_mock_h5_with_bad_world(bad_world, float("nan")) as f:
        graphs = ds.construct_graph(f, step=0)

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph["object"].num_nodes == (NUM_WORLDS - 1) * NUM_BODIES
    assert bad_world not in graph["object"].world.tolist()


def test_large_value_world_is_filtered():
    """A world with an extreme velocity must be excluded."""
    ds = _make_dataset()
    bad_world = 0
    with _make_mock_h5_with_bad_world(bad_world, 1e9) as f:
        graphs = ds.construct_graph(f, step=0)

    assert len(graphs) == 1
    graph = graphs[0]
    assert bad_world not in graph["object"].world.tolist()


def test_all_worlds_bad_returns_empty():
    """When every world is invalid, construct_graph must return an empty list."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=2, num_bodies=NUM_BODIES, num_contacts_per_world=0) as f:
        vel = f["data/body_vel"][:]
        vel[:] = float("nan")
        del f["data/body_vel"]
        f.create_dataset("data/body_vel", data=vel)
        graphs = ds.construct_graph(f, step=0)

    assert graphs == []


def test_world_attr_after_filtering():
    """After filtering world 1, .world on object nodes should be [0,0,2,2] (original indices)."""
    ds = _make_dataset()
    bad_world = 1
    with _make_mock_h5_with_bad_world(bad_world, float("nan"), num_worlds=3) as f:
        graph = ds.construct_graph(f, step=0)[0]

    expected = torch.tensor(
        [w for w in range(3) if w != bad_world for _ in range(NUM_BODIES)], dtype=torch.long
    )
    assert torch.equal(graph["object"].world, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Edge attribute correctness tests
# ──────────────────────────────────────────────────────────────────────────────


def test_lever_arm_values():
    """Verify lever arm calculations are correct."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=1, num_bodies=2, num_contacts_per_world=1) as f:
        graph = ds.construct_graph(f, step=0)[0]

        if graph[("object", "inter_object", "contact_point")].num_edges == 0:
            pytest.skip("no object-contact edges in test data")

        # Get contact data for world 0
        step = 0
        body_pose_prev = f["data/body_pose_prev"][step][0]  # [B, 7]
        body_com = f["model/body_com"][0]  # [B, 3]
        contact_point0 = f["data/contact_point0"][step][0, 0]  # [3]
        contact_normal = f["data/contact_normal"][step][0, 0]  # [3]
        contact_thickness0 = f["data/contact_thickness0"][step][0, 0]  # scalar

        # Compute expected lever arm using torch (consistent with conversion)
        import torch
        from axion.gnn.graph_builder import quat_to_rot_matrix

        body_pose_torch = torch.tensor(body_pose_prev[[0]], dtype=torch.float32)
        com_torch = torch.tensor(body_com[[0]], dtype=torch.float32)
        cp0_torch = torch.tensor(contact_point0, dtype=torch.float32)
        normal_torch = torch.tensor(contact_normal, dtype=torch.float32)

        R = quat_to_rot_matrix(body_pose_torch[..., 3:])
        R_np = R[0].numpy()

        point_global_a = R_np @ cp0_torch.numpy() + body_pose_torch[0, :3].numpy()
        com_global_a = R_np @ com_torch[0].numpy() + body_pose_torch[0, :3].numpy()
        # For contact side 0, normal is contact_normal (not negated)
        # Adjustment: point_adj = point_global - thickness * normal
        point_adj_a = point_global_a - contact_thickness0 * contact_normal
        expected_lever = point_adj_a - com_global_a

        # Extract from graph edge attributes
        edge_attr = graph[("object", "inter_object", "contact_point")].edge_attr[0]  # [4]
        lever_from_graph = edge_attr[:3].numpy()

        assert np.allclose(
            lever_from_graph, expected_lever, atol=1e-6
        ), f"Lever arm mismatch: got {lever_from_graph}, expected {expected_lever}"


def test_lever_norm_is_positive():
    """Lever norms should always be non-negative."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

    for edge_type in [
        ("object", "inter_object", "contact_point"),
        ("contact_point", "inter_object", "object"),
        ("floor", "inter_object", "contact_point"),
    ]:
        if graph[edge_type].num_edges == 0:
            continue
        # Lever norm is last column of 4-dim edge attr
        lever_norms = graph[edge_type].edge_attr[:, 3]
        assert (lever_norms >= 0).all(), f"Negative lever norm found in {edge_type}"


def test_contact_normal_values():
    """Contact normals in contact-point edges should match input normals."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=1, num_bodies=2, num_contacts_per_world=1) as f:
        graph = ds.construct_graph(f, step=0)[0]

        if graph[("contact_point", "contact", "contact_point")].num_edges == 0:
            pytest.skip("no contact-contact edges in test data")

        # Get contact data
        contact_normal = f["data/contact_normal"][0][0, 0]  # [3]

        edge_attr = graph[("contact_point", "contact", "contact_point")].edge_attr
        # Forward edge (cp0 -> cp1) has normal pointing from cp0 to cp1 = contact_normal
        # Reverse edge (cp1 -> cp0) has normal pointing from cp1 to cp0 = -contact_normal
        normal_fwd = edge_attr[0, :3].numpy()  # Forward edge
        normal_bwd = edge_attr[1, :3].numpy()  # Reverse edge

        assert np.allclose(normal_fwd, contact_normal, atol=1e-5)
        assert np.allclose(normal_bwd, -contact_normal, atol=1e-5)


def test_friction_coefficient_in_edges():
    """Contact-contact edges should have friction coefficient (default 0.5)."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

        if graph[("contact_point", "contact", "contact_point")].num_edges == 0:
            pytest.skip("no contact-contact edges in test data")

        edge_attr = graph[("contact_point", "contact", "contact_point")].edge_attr
        # Friction is last column of 5-dim edge attr
        friction = edge_attr[:, 4]
        assert (friction > 0).all(), "Friction coefficients should be positive"
        assert (friction <= 1.0).all(), "Friction coefficients should be <= 1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Body indexing tests
# ──────────────────────────────────────────────────────────────────────────────


def test_body_indices_in_object_contact_edges():
    """Object indices in edges should be valid body indices in correct world."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

    edge_type = ("object", "inter_object", "contact_point")
    if graph[edge_type].num_edges == 0:
        pytest.skip("no object-contact edges")

    src_idx = graph[edge_type].edge_index[0]
    src_worlds = graph["object"].world[src_idx]
    cp_worlds = graph["contact_point"].world[graph[edge_type].edge_index[1]]

    # Source and destination should be in same world
    assert torch.all(src_worlds == cp_worlds), "Cross-world edge in object-contact edge"


def test_floor_contacts_have_body_index_minus_one():
    """Contacts with floor should have been processed with body_idx=-1."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=1, num_bodies=2, num_contacts_per_world=2) as f:
        # Contact 1 is object-floor: shape 0 vs shape 2
        graph = ds.construct_graph(f, step=0)[0]

    # Floor should appear in ("floor", "inter_object", "contact_point")
    if graph[("floor", "inter_object", "contact_point")].num_edges > 0:
        floor_edges = graph[("floor", "inter_object", "contact_point")]
        # Verify floor node indices are valid
        floor_idx = floor_edges.edge_index[0]
        assert (floor_idx < graph["floor"].num_nodes).all()


# ──────────────────────────────────────────────────────────────────────────────
# Contact batching variants
# ──────────────────────────────────────────────────────────────────────────────


def test_single_contact_per_world():
    """Single contact per world should create 2 contact points."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=2, num_bodies=2, num_contacts_per_world=1) as f:
        graph = ds.construct_graph(f, step=0)[0]

    # 2 worlds × 1 contact × 2 sides = 4 contact points
    assert graph["contact_point"].num_nodes == 4


def test_zero_contacts_per_world():
    """Zero contacts should have empty contact nodes."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=2, num_bodies=2, num_contacts_per_world=0) as f:
        graph = ds.construct_graph(f, step=0)[0]

    assert graph["contact_point"].num_nodes == 0
    assert graph[("object", "inter_object", "contact_point")].num_edges == 0


def test_large_contact_count():
    """Verify batching works with many contacts per world."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=1, num_bodies=3, num_contacts_per_world=5) as f:
        graph = ds.construct_graph(f, step=0)[0]

    # Mock data generates: contact 0 (2 unique points), contact 1 (2 unique points),
    # contacts 2-4 are duplicates of contact 0's positions → 4 total unique contact points
    assert graph["contact_point"].num_nodes == 4


# ──────────────────────────────────────────────────────────────────────────────
# Numerical stability tests
# ──────────────────────────────────────────────────────────────────────────────


def test_no_nan_in_features():
    """No NaN values should appear in any features."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

    # Check all node features
    for node_type in ["object", "floor", "contact_point"]:
        x = graph[node_type].x
        assert not torch.isnan(x).any(), f"NaN found in {node_type}.x"

    # Check all edge features
    for edge_type in graph.edge_types:
        edge_attr = graph[edge_type].edge_attr
        if edge_attr.numel() > 0:
            assert not torch.isnan(edge_attr).any(), f"NaN found in {edge_type}.edge_attr"


def test_no_inf_in_features():
    """No infinity values should appear in any features."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

    # Check all node features
    for node_type in ["object", "floor", "contact_point"]:
        x = graph[node_type].x
        assert not torch.isinf(x).any(), f"Inf found in {node_type}.x"

    # Check all edge features
    for edge_type in graph.edge_types:
        edge_attr = graph[edge_type].edge_attr
        if edge_attr.numel() > 0:
            assert not torch.isinf(edge_attr).any(), f"Inf found in {edge_type}.edge_attr"


def test_lever_norms_reasonable():
    """Lever norms should be reasonable (not huge)."""
    ds = _make_dataset()
    with _make_mock_h5(NUM_WORLDS, NUM_BODIES, NUM_CONTACTS) as f:
        graph = ds.construct_graph(f, step=0)[0]

    for edge_type in [
        ("object", "inter_object", "contact_point"),
        ("contact_point", "inter_object", "object"),
        ("floor", "inter_object", "contact_point"),
    ]:
        if graph[edge_type].num_edges == 0:
            continue
        # Lever norm is last column, should be < 100 (reasonable for object sizes)
        lever_norms = graph[edge_type].edge_attr[:, 3]
        assert (lever_norms < 100).all(), f"Unreasonably large lever norm in {edge_type}"


# ──────────────────────────────────────────────────────────────────────────────
# Contact-contact edge symmetry tests
# ──────────────────────────────────────────────────────────────────────────────


def test_contact_contact_edges_bidirectional():
    """Contact-contact edges should be bidirectional (cp0->cp1 and cp1->cp0)."""
    ds = _make_dataset()
    with _make_mock_h5(num_worlds=1, num_bodies=2, num_contacts_per_world=1) as f:
        graph = ds.construct_graph(f, step=0)[0]

    edge_type = ("contact_point", "contact", "contact_point")
    if graph[edge_type].num_edges == 0:
        pytest.skip("no contact-contact edges")

    edges = graph[edge_type].edge_index
    attrs = graph[edge_type].edge_attr

    # For each contact, there should be 2 edges (forward and backward)
    # Check they have opposite normals
    if edges.shape[1] >= 2:
        normal_fwd = attrs[0, :3]
        normal_bwd = attrs[1, :3]
        # Normals should be opposite: bwd = -fwd
        assert torch.allclose(
            normal_bwd, -normal_fwd, atol=1e-5
        ), "Contact-contact edge normals should be opposite for bidirectional edges"
