import numpy as np
import warp as wp

# =================================================================================
# 1. BASELINE: ATOMIC SCATTER (UNIFIED & SEPARATED)
# =================================================================================


@wp.kernel
def kernel_unified_scatter(
    lambda_: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    u: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = lambda_[c_idx]

    # Body 0
    b0 = constraint_to_body[c_idx, 0]
    if b0 >= 0:
        impulse0 = J[c_idx, 0] * lam
        wp.atomic_add(u, b0, M_inv[b0] * impulse0)

    # Body 1
    b1 = constraint_to_body[c_idx, 1]
    if b1 >= 0:
        impulse1 = J[c_idx, 1] * lam
        wp.atomic_add(u, b1, M_inv[b1] * impulse1)


@wp.kernel
def kernel_scatter_joints(
    lambda_: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    u: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = lambda_[c_idx]
    b0 = constraint_to_body[c_idx, 0]
    b1 = constraint_to_body[c_idx, 1]

    # Simple atomic add
    wp.atomic_add(u, b0, M_inv[b0] * (J[c_idx, 0] * lam))
    wp.atomic_add(u, b1, M_inv[b1] * (J[c_idx, 1] * lam))


@wp.kernel
def kernel_scatter_contacts(
    lambda_: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    u: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = lambda_[c_idx]
    b0 = constraint_to_body[c_idx]
    wp.atomic_add(u, b0, M_inv[b0] * (J[c_idx] * lam))


@wp.kernel
def kernel_sum_vectors(
    a: wp.array(dtype=wp.spatial_vector),
    b: wp.array(dtype=wp.spatial_vector),
    out: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    out[i] = a[i] + b[i]


# =================================================================================
# 2. NAIVE TILED (Indirect Load)
# =================================================================================


@wp.kernel
def kernel_precompute_joint_W_naive(
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W_flat: wp.array(dtype=wp.spatial_vector),
):
    flat_idx = wp.tid()
    if flat_idx == 0:
        return

    idx = flat_idx - 1
    c_idx = idx // 2
    side = idx % 2

    b_idx = constraint_to_body[c_idx, side]
    W_flat[flat_idx] = M_inv[b_idx] * J_flat[flat_idx]


@wp.func
def compute_joint_lambda_idx(J_idx: int):
    return wp.rshift(J_idx + 1, 1)


def create_naive_tiled_joint_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_joints_naive(
        lambda_padded: wp.array(dtype=wp.float32),
        W_flat: wp.array(dtype=wp.spatial_vector),
        body_to_constraints: wp.array(dtype=wp.int32),
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        tile_offset = tile_idx * tile_size

        # Indirect Load: Must read indices first
        J_indices = wp.tile_load(
            body_to_constraints, shape=(tile_size,), offset=(tile_offset,), storage="shared"
        )
        lam_indices = wp.tile_map(compute_joint_lambda_idx, J_indices)

        # Scattered Global Memory Access (Slow)
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        u_tile = wp.tile_map(wp.mul, W_tile, lam_tile)
        u_matrix = wp.tile_reshape(u_tile, shape=(bodies_in_tile, constraints_per_body))
        u_sum = wp.tile_reduce(wp.add, u_matrix, axis=1)

        wp.tile_store(u, u_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_joints_naive


# =================================================================================
# 3. OPTIMIZED TILED (Baked / Flat 1D)
# =================================================================================


@wp.kernel
def kernel_bake_joint_data_flat(
    # Inputs
    J_flat: wp.array(dtype=wp.spatial_vector),
    body_to_constraints: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # Outputs (1D Flattened / Coalesced)
    W_binned_flat: wp.array(dtype=wp.spatial_vector),
    lam_indices_binned_flat: wp.array(dtype=wp.int32),
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]

    # Calculate base offset in the 1D array
    base_offset = body_idx * stride

    for i in range(stride):
        flat_idx = body_to_constraints[body_idx, i]
        write_idx = base_offset + i

        if flat_idx > 0:
            # 1. Bake W = M_inv * J
            W_binned_flat[write_idx] = m_inv_local * J_flat[flat_idx]
            # 2. Pre-calc Lambda Index
            lam_indices_binned_flat[write_idx] = wp.rshift(flat_idx + 1, 1)
        else:
            # Padding
            W_binned_flat[write_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            lam_indices_binned_flat[write_idx] = 0


def create_optimized_tiled_joint_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_joints_optimized(
        lambda_padded: wp.array(dtype=wp.float32),
        W_binned_flat: wp.array(dtype=wp.spatial_vector),  # 1D Input
        lam_indices_binned_flat: wp.array(dtype=wp.int32),  # 1D Input
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        data_offset = tile_idx * tile_size

        # 1. Load Indices (DIRECT 1D LOAD - Coalesced)
        lam_idx_tile = wp.tile_load(
            lam_indices_binned_flat, shape=(tile_size,), offset=(data_offset,)
        )

        # 2. Load Vectors (DIRECT 1D LOAD - Coalesced)
        W_tile = wp.tile_load(W_binned_flat, shape=(tile_size,), offset=(data_offset,))

        # 3. Gather Lambda (Still indirect, but strictly scalar float load)
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_idx_tile, shape=(tile_size,))

        # 4. Compute
        u_tile = wp.tile_map(wp.mul, W_tile, lam_tile)
        u_matrix = wp.tile_reshape(u_tile, shape=(bodies_in_tile, constraints_per_body))
        u_sum = wp.tile_reduce(wp.add, u_matrix, axis=1)

        wp.tile_store(u, u_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_joints_optimized


# =================================================================================
# 4. CONTACTS (Standard Tiled)
# =================================================================================


@wp.kernel
def kernel_precompute_contact_W(
    J: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    b_idx = constraint_to_body[c_idx]
    W[c_idx] = M_inv[b_idx] * J[c_idx]


def create_tiled_contact_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_contacts(
        lambda_: wp.array(dtype=wp.float32),
        W: wp.array(dtype=wp.spatial_vector),
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        offset = tile_idx * tile_size

        lam_tile = wp.tile_load(lambda_, shape=(tile_size,), offset=(offset,))
        W_tile = wp.tile_load(W, shape=(tile_size,), offset=(offset,))

        u_tile = wp.tile_map(wp.mul, W_tile, lam_tile)
        u_matrix = wp.tile_reshape(u_tile, shape=(bodies_in_tile, constraints_per_body))
        u_sum = wp.tile_reduce(wp.add, u_matrix, axis=1)

        wp.tile_store(u, u_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_contacts


# =================================================================================
# 5. HELPERS
# =================================================================================


def generate_safe_joint_map(num_joints, num_bodies, max_joints_per_body):
    rng = np.random.default_rng(42)
    map_j = np.zeros((num_joints, 2), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)
    available = list(range(num_bodies))

    for i in range(num_joints):
        if len(available) < 2:
            raise RuntimeError("Joint Gen Failed")
        b1, b2 = rng.choice(available, size=2, replace=False)
        map_j[i] = [b1, b2]
        counts[b1] += 1
        counts[b2] += 1
        if counts[b1] >= max_joints_per_body:
            available.remove(b1)
        if counts[b2] >= max_joints_per_body and b2 in available:
            available.remove(b2)
    return map_j


def build_joint_graph(num_bodies, max_constraints, J_np, map_np):
    num_cons = len(J_np)
    J_flat = np.zeros((1 + num_cons * 2, 6), dtype=np.float32)
    J_flat[1:] = J_np.reshape(-1, 6)

    body_to_flat = np.zeros((num_bodies, max_constraints), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)

    for c_i in range(num_cons):
        idx_A, idx_B = 1 + c_i * 2, 1 + c_i * 2 + 1
        b0, b1 = map_np[c_i]
        if counts[b0] < max_constraints:
            body_to_flat[b0, counts[b0]] = idx_A
            counts[b0] += 1
        if counts[b1] < max_constraints:
            body_to_flat[b1, counts[b1]] = idx_B
            counts[b1] += 1

    return J_flat, body_to_flat
