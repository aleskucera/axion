import numpy as np
import warp as wp

wp.init()


# =================================================================================
# 1. BASELINE: UNIFIED SCATTER KERNEL
# =================================================================================
@wp.kernel
def kernel_unified_scatter(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = dlambda[c_idx]

    # Body 0
    b0 = constraint_to_body[c_idx, 0]
    if b0 >= 0:
        impulse0 = J[c_idx, 0] * lam
        wp.atomic_add(du, b0, M_inv[b0] * impulse0)

    # Body 1
    b1 = constraint_to_body[c_idx, 1]
    if b1 >= 0:
        impulse1 = J[c_idx, 1] * lam
        wp.atomic_add(du, b1, M_inv[b1] * impulse1)


# =================================================================================
# 2. SEPARATED SCATTER KERNELS
# =================================================================================
@wp.kernel
def kernel_scatter_joints(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector, ndim=2),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = dlambda[c_idx]

    b0 = constraint_to_body[c_idx, 0]
    b1 = constraint_to_body[c_idx, 1]

    wp.atomic_add(du, b0, M_inv[b0] * (J[c_idx, 0] * lam))
    wp.atomic_add(du, b1, M_inv[b1] * (J[c_idx, 1] * lam))


@wp.kernel
def kernel_scatter_contacts(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    lam = dlambda[c_idx]
    b0 = constraint_to_body[c_idx]

    wp.atomic_add(du, b0, M_inv[b0] * (J[c_idx] * lam))


# =================================================================================
# 3. OPTIMIZED TILED KERNELS
# =================================================================================


# --- A. Precomputation Kernels ---
@wp.kernel
def kernel_precompute_joint_W(
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    W_flat: wp.array(dtype=wp.spatial_vector),
):
    flat_idx = wp.tid()
    if flat_idx == 0:
        return  # Padding

    idx = flat_idx - 1
    c_idx = idx // 2
    side = idx % 2

    b_idx = constraint_to_body[c_idx, side]
    W_flat[flat_idx] = M_inv[b_idx] * J_flat[flat_idx]


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


# --- B. Solver Kernels ---


@wp.func
def compute_joint_lambda_idx(J_idx: int):
    return wp.rshift(J_idx + 1, 1)


def create_tiled_joint_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_joints(
        lambda_padded: wp.array(dtype=wp.float32),
        W_flat: wp.array(dtype=wp.spatial_vector),
        body_to_constraints: wp.array(dtype=wp.int32),
        du: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        tile_offset = tile_idx * tile_size

        J_indices = wp.tile_load(
            body_to_constraints, shape=(tile_size,), offset=(tile_offset,), storage="shared"
        )

        lam_indices = wp.tile_map(compute_joint_lambda_idx, J_indices)

        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        dv_tile = wp.tile_map(wp.mul, W_tile, lam_tile)

        dv_matrix = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))
        dv_sum = wp.tile_reduce(wp.add, dv_matrix, axis=1)

        wp.tile_store(du, dv_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_joints


def create_tiled_contact_solver(bodies_in_tile, constraints_per_body):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def solve_contacts(
        lambda_: wp.array(dtype=wp.float32),
        W: wp.array(dtype=wp.spatial_vector),
        du: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        offset = tile_idx * tile_size

        lam_tile = wp.tile_load(lambda_, shape=(tile_size,), offset=(offset,))
        W_tile = wp.tile_load(W, shape=(tile_size,), offset=(offset,))

        dv_tile = wp.tile_map(wp.mul, W_tile, lam_tile)

        dv_matrix = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))
        dv_sum = wp.tile_reduce(wp.add, dv_matrix, axis=1)

        wp.tile_store(du, dv_sum, offset=(tile_idx * bodies_in_tile,))

    return solve_contacts


@wp.kernel
def kernel_sum_vectors(
    a: wp.array(dtype=wp.spatial_vector),
    b: wp.array(dtype=wp.spatial_vector),
    out: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    out[i] = a[i] + b[i]


# =================================================================================
# 4. HELPERS
# =================================================================================


def build_joint_graph(num_bodies, max_constraints, J_np, map_np):
    num_cons = len(J_np)
    J_flat = np.zeros((1 + num_cons * 2, 6), dtype=np.float32)
    J_flat[1:] = J_np.reshape(-1, 6)

    body_to_flat = np.zeros((num_bodies, max_constraints), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)

    for c_i in range(num_cons):
        idx_A, idx_B = 1 + c_i * 2, 1 + c_i * 2 + 1
        b0, b1 = map_np[c_i]

        # If we hit the limit, we MUST warn or fail for this test,
        # because the Reference kernel will calculate it but Tiled will skip it.
        if counts[b0] < max_constraints:
            body_to_flat[b0, counts[b0]] = idx_A
            counts[b0] += 1

        if counts[b1] < max_constraints:
            body_to_flat[b1, counts[b1]] = idx_B
            counts[b1] += 1

    return J_flat, body_to_flat.flatten()


def generate_safe_joint_map(num_joints, num_bodies, max_joints_per_body):
    """Generates random joints ensuring no body exceeds the max_joints_per_body limit."""
    rng = np.random.default_rng(42)
    map_j = np.zeros((num_joints, 2), dtype=np.int32)
    counts = np.zeros(num_bodies, dtype=np.int32)

    # List of bodies that still have "slots" available
    available_bodies = list(range(num_bodies))

    for i in range(num_joints):
        if len(available_bodies) < 2:
            raise RuntimeError("Not enough capacity in bodies to satisfy num_joints constraint!")

        # Pick 2 distinct bodies
        b1, b2 = rng.choice(available_bodies, size=2, replace=False)

        map_j[i] = [b1, b2]

        counts[b1] += 1
        counts[b2] += 1

        # If full, remove from pool
        if counts[b1] >= max_joints_per_body:
            available_bodies.remove(b1)
        if counts[b2] >= max_joints_per_body and b2 in available_bodies:
            available_bodies.remove(b2)

    return map_j


def main():
    # Settings
    num_bodies = 1024
    num_joints = 2048

    # Tiling limits
    joint_constraints_per_body = 8
    contact_constraints_per_body = 16
    bodies_in_tile = 8

    num_contacts = num_bodies * contact_constraints_per_body

    print(
        f"Generating Simulation: {num_bodies} Bodies, {num_joints} Joints, {num_contacts} Contacts"
    )

    # ---------------------------
    # DATA GENERATION
    # ---------------------------
    M_inv_np = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T

    # Joints: Use SAFE generator to prevent divergence
    try:
        map_j_np = generate_safe_joint_map(num_joints, num_bodies, joint_constraints_per_body)
    except RuntimeError as e:
        print(f"Error generating graph: {e}")
        return

    lam_j_np = np.random.randn(num_joints).astype(np.float32)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)

    # Contacts
    lam_c_np = np.random.randn(num_contacts).astype(np.float32)
    J_c_np = np.random.randn(num_contacts, 6).astype(np.float32)
    map_c_np = np.repeat(np.arange(num_bodies, dtype=np.int32), contact_constraints_per_body)

    # Unified Data Construction
    lam_all = np.concatenate([lam_j_np, lam_c_np])

    # Pad contacts to 2 bodies for unified format
    J_c_padded = np.zeros((num_contacts, 2, 6), dtype=np.float32)
    J_c_padded[:, 0, :] = J_c_np
    J_all = np.concatenate([J_j_np, J_c_padded])

    map_c_padded = np.full((num_contacts, 2), -1, dtype=np.int32)
    map_c_padded[:, 0] = map_c_np
    map_all = np.concatenate([map_j_np, map_c_padded])

    # ---------------------------
    # WARP ALLOCATION
    # ---------------------------
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)

    lam_u = wp.array(lam_all, dtype=wp.float32)
    J_u = wp.array(J_all, dtype=wp.spatial_vector)
    map_u = wp.array(map_all, dtype=wp.int32)

    lam_j = wp.array(lam_j_np, dtype=wp.float32)
    J_j = wp.array(J_j_np, dtype=wp.spatial_vector)
    map_j = wp.array(map_j_np, dtype=wp.int32)

    lam_c = wp.array(lam_c_np, dtype=wp.float32)
    J_c = wp.array(J_c_np, dtype=wp.spatial_vector)
    map_c = wp.array(map_c_np, dtype=wp.int32)

    du_unified = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_separated = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_tiled_j = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_tiled_c = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # ---------------------------
    # EXECUTION
    # ---------------------------

    # 1. Unified
    wp.launch(
        kernel_unified_scatter,
        dim=len(lam_all),
        inputs=[lam_u, J_u, map_u, M_inv],
        outputs=[du_unified],
    )

    # 2. Separated
    wp.launch(
        kernel_scatter_joints,
        dim=num_joints,
        inputs=[lam_j, J_j, map_j, M_inv],
        outputs=[du_separated],
    )
    wp.launch(
        kernel_scatter_contacts,
        dim=num_contacts,
        inputs=[lam_c, J_c, map_c, M_inv],
        outputs=[du_separated],
    )

    # 3. Tiled

    # Joint Prep
    J_j_flat_np, map_j_flat_np = build_joint_graph(
        num_bodies, joint_constraints_per_body, J_j_np, map_j_np
    )
    lam_j_pad_np = np.zeros(num_joints + 1, dtype=np.float32)
    lam_j_pad_np[1:] = lam_j_np

    lam_j_pad = wp.array(lam_j_pad_np, dtype=wp.float32)
    J_j_flat = wp.array(J_j_flat_np, dtype=wp.spatial_vector)
    map_j_flat = wp.array(map_j_flat_np, dtype=wp.int32)
    W_j_flat = wp.zeros_like(J_j_flat)

    # Joint Solve
    wp.launch(
        kernel_precompute_joint_W,
        dim=len(J_j_flat),
        inputs=[J_j_flat, map_j, M_inv],
        outputs=[W_j_flat],
    )
    k_joint = create_tiled_joint_solver(bodies_in_tile, joint_constraints_per_body)
    wp.launch_tiled(
        k_joint,
        dim=[num_bodies // bodies_in_tile],
        inputs=[lam_j_pad, W_j_flat, map_j_flat],
        outputs=[du_tiled_j],
        block_dim=bodies_in_tile * joint_constraints_per_body,
    )

    # Contact Prep & Solve
    W_c = wp.zeros_like(J_c)
    wp.launch(
        kernel_precompute_contact_W, dim=num_contacts, inputs=[J_c, map_c, M_inv], outputs=[W_c]
    )
    k_contact = create_tiled_contact_solver(bodies_in_tile, contact_constraints_per_body)
    wp.launch_tiled(
        k_contact,
        dim=[num_bodies // bodies_in_tile],
        inputs=[lam_c, W_c],
        outputs=[du_tiled_c],
        block_dim=bodies_in_tile * contact_constraints_per_body,
    )

    # Combine
    wp.launch(
        kernel_sum_vectors, dim=num_bodies, inputs=[du_tiled_j, du_tiled_c], outputs=[du_tiled]
    )

    wp.synchronize()

    # ---------------------------
    # COMPARISON
    # ---------------------------
    res_u = du_unified.numpy()
    res_s = du_separated.numpy()
    res_t = du_tiled.numpy()

    err_s = np.max(np.abs(res_u - res_s))
    err_t = np.max(np.abs(res_u - res_t))

    print(f"\nResults:")
    print(f"Unified vs Separated Error: {err_s:.6f}")
    print(f"Unified vs Tiled Error:     {err_t:.6f}")

    if err_t < 1e-4:
        print("SUCCESS: Tiled optimization is physically identical.")
    else:
        print("FAILURE: Divergence detected.")


if __name__ == "__main__":
    main()
