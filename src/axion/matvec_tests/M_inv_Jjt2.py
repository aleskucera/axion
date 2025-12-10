import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 1. REFERENCE: SCATTER KERNEL (Unchanged)
# =================================================================================


@wp.kernel
def kernel_M_inv_Jjt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),  # (num_constraints)
    J: wp.array(dtype=wp.spatial_vector, ndim=2),  # (num_constraints, 2)
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),  # (num_constraints, 2)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    du: wp.array(dtype=wp.spatial_vector),  # (num_bodies) - Output
):
    c_idx = wp.tid()

    b0_idx = constraint_to_body[c_idx, 0]
    b1_idx = constraint_to_body[c_idx, 1]

    # Load lambda once
    lam = dlambda[c_idx]

    # Apply to Body 0
    if b0_idx >= 0:
        delta_impulse0 = J[c_idx, 0] * lam
        m0_inv = M_inv[b0_idx]
        wp.atomic_add(du, b0_idx, m0_inv * delta_impulse0)

    # Apply to Body 1
    if b1_idx >= 0:
        delta_impulse1 = J[c_idx, 1] * lam
        m1_inv = M_inv[b1_idx]
        wp.atomic_add(du, b1_idx, m1_inv * delta_impulse1)


# =================================================================================
# 2. NEW: PRE-COMPUTATION KERNEL (Proposition 2)
# =================================================================================


@wp.kernel
def kernel_precompute_W(
    # Inputs
    J_flat: wp.array(dtype=wp.spatial_vector),  # (2*N + 1)
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),  # (N, 2)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    # Output
    W_flat: wp.array(dtype=wp.spatial_vector),  # (2*N + 1) - Weighted Jacobian
):
    # Iterate linearly over the flat array (skipping index 0)
    flat_idx = wp.tid()

    # 0 is padding
    if flat_idx == 0:
        W_flat[0] = wp.spatial_vector()
        return

    # Map flat index back to constraint and side
    # Index 1 -> C0 Side 0
    # Index 2 -> C0 Side 1
    # Index 3 -> C1 Side 0

    # Logical index (0-based)
    idx = flat_idx - 1

    c_idx = idx // 2
    side = idx % 2  # 0 or 1

    # Find the body index
    body_idx = constraint_to_body[c_idx, side]

    if body_idx >= 0:
        # W = M_inv * J
        j_vec = J_flat[flat_idx]
        m_inv = M_inv[body_idx]
        w_vec = m_inv * j_vec
        W_flat[flat_idx] = w_vec
    else:
        # Static body or invalid
        W_flat[flat_idx] = wp.spatial_vector()


# =================================================================================
# 3. OPTIMIZED TILED KERNEL (Propositions 1 & 2 Combined)
# =================================================================================


@wp.func
def compute_lambda_idx_aligned(J_idx: int):
    # Proposition 1: Robust Bitwise Math
    # J_flat indices: [0(Null), 1(C0_A), 2(C0_B), 3(C1_A), 4(C1_B)]
    # Lambda_padded : [0(Null), 1(C0),   1(C0),   2(C1),   2(C1)  ]

    # Formula: (J + 1) >> 1
    # 0 -> 1 >> 1 = 0
    # 1 -> 2 >> 1 = 1
    # 2 -> 3 >> 1 = 1
    # 3 -> 4 >> 1 = 2
    return wp.rshift(J_idx + 1, 1)


def create_solver_tiled_optimized(bodies_in_tile, constraints_per_body):

    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_solve_tiled_opt(
        # -- Padded Data (Prop 1) --
        lambda_padded: wp.array(dtype=wp.float32),  # (num_constraints + 1)
        # -- Precomputed Data (Prop 2) --
        W_flat: wp.array(dtype=wp.spatial_vector),  # M_inv * J stored linearly
        # -- Topology --
        body_to_constraints: wp.array(dtype=wp.int32),
        # -- Output --
        du: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        body_offset = tile_idx * bodies_in_tile
        map_offset = tile_idx * tile_size

        # 1. Load Topology (Indirection)
        # ------------------------------
        J_indices = wp.tile_load(
            body_to_constraints,
            shape=(tile_size,),
            offset=(map_offset,),
            storage="shared",
        )

        # 2. Compute Lambda Indices (Math - Prop 1)
        # -----------------------------------------
        # No memory load here! Just math.
        lam_indices = wp.tile_map(compute_lambda_idx_aligned, J_indices)

        # 3. Load Data
        # ------------
        # Load Lambda (Coalesced-ish gather)
        lam_tile = wp.tile_load_indexed(lambda_padded, indices=lam_indices, shape=(tile_size,))

        # Load W (Weighted Jacobian)
        # Note: We load W instead of loading J AND M_inv separately.
        W_tile = wp.tile_load_indexed(W_flat, indices=J_indices, shape=(tile_size,))

        # 4. Math (Simplified - Prop 2)
        # -----------------------------
        # du = W * lambda
        # (Mass matrix multiplication is already baked into W)
        dv_tile = wp.tile_map(wp.mul, W_tile, lam_tile)

        # 5. Reduction
        # ------------
        # Reshape to (bodies, constraints)
        dv_by_body = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))

        # Sum constraints per body
        total_dv = wp.tile_reduce(wp.add, dv_by_body, axis=1)

        # 6. Store
        # --------
        wp.tile_store(du, total_dv, offset=body_offset)

    return kernel_solve_tiled_opt


# =================================================================================
# 4. HELPER: DATA CONVERSION
# =================================================================================


def build_tiled_graph_data(num_bodies, max_constraints_per_body, J_np, constr_map_np):
    num_constraints = len(J_np)
    num_flat_entries = 1 + (num_constraints * 2)

    # Flatten J
    J_flat_np = np.zeros((num_flat_entries, 6), dtype=np.float32)
    J_flat_np[1:] = J_np.reshape(-1, 6)

    # Build Body -> Flat Indices Map
    body_to_flat_indices = np.zeros((num_bodies, max_constraints_per_body), dtype=np.int32)
    body_counts = np.zeros(num_bodies, dtype=np.int32)

    for c_i in range(num_constraints):
        # Calculate indices once
        idx_A = 1 + c_i * 2 + 0
        idx_B = 1 + c_i * 2 + 1

        b0 = constr_map_np[c_i, 0]
        if b0 >= 0 and body_counts[b0] < max_constraints_per_body:
            body_to_flat_indices[b0, body_counts[b0]] = idx_A
            body_counts[b0] += 1

        b1 = constr_map_np[c_i, 1]
        if b1 >= 0 and body_counts[b1] < max_constraints_per_body:
            body_to_flat_indices[b1, body_counts[b1]] = idx_B
            body_counts[b1] += 1

    return J_flat_np, body_to_flat_indices.flatten()


def generate_constr_to_body(num_constraints, num_bodies, max_constraints_per_body, seed=None):
    rng = np.random.default_rng(seed)
    counts = np.zeros(num_bodies, dtype=np.int32)
    arr = np.zeros((num_constraints, 2), dtype=np.int32)

    for i in range(num_constraints):
        available = np.where(counts < max_constraints_per_body)[0]
        if len(available) < 2:
            raise ValueError("Topology generation failed.")
        v1 = rng.choice(available)
        avail2 = available[available != v1]
        if len(avail2) == 0:
            raise ValueError("Topology generation failed.")
        v2 = rng.choice(avail2)
        arr[i, 0] = v1
        arr[i, 1] = v2
        counts[v1] += 1
        counts[v2] += 1
    return arr


def generate_M_inv(num_bodies: int):
    M_inv = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv[i] = M_inv[i] @ M_inv[i].T
    return M_inv


# =================================================================================
# 5. MAIN EXECUTION
# =================================================================================


def main():
    # Settings
    num_bodies = 1024
    num_joint_constraints = 4096
    constraints_per_body = 16
    bodies_in_tile = 8

    # -----------------------
    # A. Generate Random Data
    # -----------------------
    M_inv_np = generate_M_inv(num_bodies)
    dlambda_np = np.random.randn(num_joint_constraints).astype(np.float32)
    J_np = np.random.randn(num_joint_constraints, 2, 6).astype(np.float32)
    constr_map_np = generate_constr_to_body(num_joint_constraints, num_bodies, constraints_per_body)

    # -----------------------
    # B. Run Scatter Kernel (The Reference)
    # -----------------------
    # Standard arrays for Scatter
    dlambda_wp = wp.array(dlambda_np, dtype=wp.float32)
    J_wp = wp.array(J_np, dtype=wp.spatial_vector)
    constr_map_wp = wp.array(constr_map_np, dtype=wp.int32)
    M_inv_wp = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    du_scatter = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    wp.launch(
        kernel_M_inv_Jjt_matvec_scatter,
        dim=num_joint_constraints,
        inputs=[dlambda_wp, J_wp, constr_map_wp, M_inv_wp],
        outputs=[du_scatter],
    )
    wp.synchronize()

    # -----------------------
    # C. Run Tiled Kernel (Optimized)
    # -----------------------

    # 1. Prep Data (Padding and Flattening)
    J_flat_np, body_to_J_np = build_tiled_graph_data(
        num_bodies, constraints_per_body, J_np, constr_map_np
    )

    # [Prop 1] Pad Lambda Array
    # Create size N+1. Index 0 is 0.0. Copy data to 1:.
    dlambda_padded_np = np.zeros(num_joint_constraints + 1, dtype=np.float32)
    dlambda_padded_np[1:] = dlambda_np

    dlambda_padded_wp = wp.array(dlambda_padded_np, dtype=wp.float32)
    J_flat_wp = wp.array(J_flat_np, dtype=wp.spatial_vector)
    body_to_J_wp = wp.array(body_to_J_np, dtype=wp.int32)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # [Prop 2] Create W container
    W_flat_wp = wp.zeros_like(J_flat_wp)

    # 2. Run Pre-computation
    # This runs ONCE per substep (not per solver iteration)
    num_flat_entries = len(J_flat_np)
    wp.launch(
        kernel_precompute_W,
        dim=num_flat_entries,
        inputs=[J_flat_wp, constr_map_wp, M_inv_wp],
        outputs=[W_flat_wp],
    )

    # 3. Instantiate and Run Solver
    # This loop is what runs 10-20 times in a real physics engine
    # Notice we don't pass M_inv or J anymore, just W.
    tiled_kernel = create_solver_tiled_optimized(bodies_in_tile, constraints_per_body)
    num_tiles = num_bodies // bodies_in_tile

    wp.launch_tiled(
        kernel=tiled_kernel,
        dim=[num_tiles],
        inputs=[dlambda_padded_wp, W_flat_wp, body_to_J_wp],
        outputs=[du_tiled],
        block_dim=bodies_in_tile * constraints_per_body,
    )
    wp.synchronize()

    # -----------------------
    # D. Compare
    # -----------------------
    res_scatter = du_scatter.numpy()
    res_tiled = du_tiled.numpy()

    diff = np.abs(res_scatter - res_tiled)
    max_err = np.max(diff)

    print(f"Comparison Results:")
    print(f"-------------------")
    print(f"Max Absolute Error: {max_err:.8f}")

    if max_err < 1e-4:
        print("SUCCESS: Optimized Tiled implementation matches Reference.")
    else:
        print("WARNING: Mismatch detected.")
        for i in range(min(5, num_bodies)):
            print(f"B{i}: Ref {res_scatter[i,0]:.4f} | Opt {res_tiled[i,0]:.4f}")


if __name__ == "__main__":
    main()
