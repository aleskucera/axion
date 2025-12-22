import numpy as np
import warp as wp

# =================================================================================
# 1. PRECOMPUTE ("Baking")
# =================================================================================


@wp.kernel
def kernel_bake_joints_flat(
    J_flat: wp.array(dtype=wp.spatial_vector),
    body_to_constraints: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # Outputs (1D Flat for Coalesced Access)
    W_j_flat: wp.array(dtype=wp.spatial_vector),  # M^-1 * J^T
    x_idx_flat: wp.array(dtype=wp.int32),  # Index into x_joints
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]
    base_offset = body_idx * stride

    for i in range(stride):
        # body_to_constraints stores (index + 1) to use 0 as padding
        flat_idx = body_to_constraints[body_idx, i]
        write_idx = base_offset + i

        if flat_idx > 0:
            # Bake W = M^-1 * J
            W_j_flat[write_idx] = m_inv_local * J_flat[flat_idx]

            # Store Index for 'x'
            # J_flat layout is [pad, c0_b0, c0_b1, c1_b0, c1_b1...]
            # Indices are 1, 2, 3, 4...
            # (1-1)>>1 = 0, (2-1)>>1 = 0. (3-1)>>1 = 1...
            x_idx_flat[write_idx] = wp.rshift(flat_idx - 1, 1)
        else:
            # Padding
            W_j_flat[write_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            x_idx_flat[write_idx] = -1


@wp.kernel
def kernel_bake_contacts_flat(
    J_c: wp.array(dtype=wp.spatial_vector),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # Output
    W_c: wp.array(dtype=wp.spatial_vector),
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]
    base_offset = body_idx * stride

    for i in range(stride):
        idx = base_offset + i
        W_c[idx] = m_inv_local * J_c[idx]


# =================================================================================
# 2. PHASE 1: BODY GATHER (Optimized Tiled Kernel)
#    Computes v = M^-1 * J^T * x
# =================================================================================


def create_body_gather_kernel(block_bodies, joints_per_body, contacts_per_body):

    tile_size_j = block_bodies * joints_per_body
    tile_size_c = block_bodies * contacts_per_body

    @wp.kernel
    def kernel_body_gather_tiled(
        # Joint Inputs (Graph)
        x_joints: wp.array(dtype=wp.float32),
        W_j_flat: wp.array(dtype=wp.spatial_vector),
        x_j_indices: wp.array(dtype=wp.int32),
        # Contact Inputs (Dense)
        x_contacts: wp.array(dtype=wp.float32),
        W_c_flat: wp.array(dtype=wp.spatial_vector),
        # Output
        v_body: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()

        # --- A. JOINTS (Graph) ---
        offset_j = tile_idx * tile_size_j

        # Coalesced loads
        inds_j = wp.tile_load(x_j_indices, shape=(tile_size_j,), offset=(offset_j,))
        W_j = wp.tile_load(W_j_flat, shape=(tile_size_j,), offset=(offset_j,))

        # Gather x (Indirect)
        vals_j = wp.tile_load_indexed(x_joints, indices=inds_j, shape=(tile_size_j,))

        impulse_j = wp.tile_map(wp.mul, W_j, vals_j)

        # --- B. CONTACTS (Dense) ---
        offset_c = tile_idx * tile_size_c

        # Coalesced loads
        W_c = wp.tile_load(W_c_flat, shape=(tile_size_c,), offset=(offset_c,))
        vals_c = wp.tile_load(x_contacts, shape=(tile_size_c,), offset=(offset_c,))

        impulse_c = wp.tile_map(wp.mul, W_c, vals_c)

        # --- C. REDUCE & SUM ---
        mat_j = wp.tile_reshape(impulse_j, shape=(block_bodies, joints_per_body))
        mat_c = wp.tile_reshape(impulse_c, shape=(block_bodies, contacts_per_body))

        sum_j = wp.tile_reduce(wp.add, mat_j, axis=1)
        sum_c = wp.tile_reduce(wp.add, mat_c, axis=1)

        # Final Body Velocity = Joint Impulse + Contact Impulse
        v_final = wp.tile_map(wp.add, sum_j, sum_c)

        wp.tile_store(v_body, v_final, offset=(tile_idx * block_bodies,))

    return kernel_body_gather_tiled


# =================================================================================
# 3. PHASE 2: CONSTRAINT APPLY
#    Computes z = J * v + C * x
# =================================================================================


@wp.kernel
def kernel_apply_joints(
    v_body: wp.array(dtype=wp.spatial_vector),
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    C: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
):
    c_idx = wp.tid()

    flat_idx_0 = 1 + c_idx * 2
    flat_idx_1 = 1 + c_idx * 2 + 1

    b0 = constraint_to_body[c_idx, 0]
    b1 = constraint_to_body[c_idx, 1]

    res = 0.0

    if b0 >= 0:
        res += wp.dot(J_flat[flat_idx_0], v_body[b0])
    if b1 >= 0:
        res += wp.dot(J_flat[flat_idx_1], v_body[b1])

    res += C[c_idx] * x[c_idx]

    z[c_idx] = res


def create_apply_contacts_kernel(stride):
    @wp.kernel
    def kernel_apply_contacts_stride(
        v_body: wp.array(dtype=wp.spatial_vector),
        J_c: wp.array(dtype=wp.spatial_vector),
        C: wp.array(dtype=wp.float32),
        x: wp.array(dtype=wp.float32),
        z: wp.array(dtype=wp.float32),
    ):
        c_idx = wp.tid()

        # Identify body owner
        b_idx = c_idx // stride

        val = wp.dot(J_c[c_idx], v_body[b_idx])
        val += C[c_idx] * x[c_idx]

        z[c_idx] = val

    return kernel_apply_contacts_stride


# =================================================================================
# 4. BASELINE (SCATTER / ATOMIC) - FOR VERIFICATION
# =================================================================================


@wp.kernel
def kernel_baseline_scatter_joints(
    x: wp.array(dtype=wp.float32),
    J_flat: wp.array(dtype=wp.spatial_vector),
    map_j: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    v: wp.array(dtype=wp.spatial_vector),  # Accumulator
):
    c_idx = wp.tid()
    val = x[c_idx]

    b0 = map_j[c_idx, 0]
    if b0 >= 0:
        idx0 = 1 + c_idx * 2
        imp0 = J_flat[idx0] * val
        wp.atomic_add(v, b0, M_inv[b0] * imp0)

    b1 = map_j[c_idx, 1]
    if b1 >= 0:
        idx1 = 1 + c_idx * 2 + 1
        imp1 = J_flat[idx1] * val
        wp.atomic_add(v, b1, M_inv[b1] * imp1)


@wp.kernel
def kernel_baseline_scatter_contacts(
    x: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    v: wp.array(dtype=wp.spatial_vector),
    stride: int,
):
    c_idx = wp.tid()
    val = x[c_idx]

    b_idx = c_idx // stride

    imp = J[c_idx] * val
    wp.atomic_add(v, b_idx, M_inv[b_idx] * imp)


# =================================================================================
# 5. GRAPH HELPERS
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
