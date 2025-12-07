import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 1. REFERENCE: SCATTER KERNEL (For Correctness Check)
# =================================================================================


@wp.kernel
def kernel_M_inv_Jt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    b_idx = constraint_to_body[c_idx]

    # Standard: du += M_inv * (J * lambda)
    delta_impulse = J[c_idx] * dlambda[c_idx]
    m_inv = M_inv[b_idx]
    wp.atomic_add(du, b_idx, m_inv * delta_impulse)


# =================================================================================
# 2. STAGE A: PRE-COMPUTATION KERNEL
# =================================================================================


@wp.kernel
def kernel_precompute_W(
    J: wp.array(dtype=wp.spatial_vector),  # (num_constraints)
    constraint_to_body: wp.array(dtype=wp.int32),  # (num_constraints)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    W: wp.array(dtype=wp.spatial_vector),  # (num_constraints) [OUTPUT]
):
    c_idx = wp.tid()

    # 1. Identify Body
    b_idx = constraint_to_body[c_idx]

    # 2. Load Data
    j_vec = J[c_idx]
    m_inv = M_inv[b_idx]

    # 3. Fuse M_inv and J
    # W = M^-1 * J
    # This vector represents the velocity change of the body
    # for a unit impulse of 1.0 on this constraint.
    w_vec = m_inv * j_vec

    W[c_idx] = w_vec


# =================================================================================
# 3. STAGE B: OPTIMIZED TILED SOLVER
# =================================================================================


def create_solve_tiled_contact_precomputed(
    num_bodies: int,
    constraints_per_body: int,
    bodies_in_tile: int,
):
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_solve_tiled_optimized(
        lambda_: wp.array(dtype=wp.float32),  # (num_constraints)
        W: wp.array(dtype=wp.spatial_vector),  # (num_constraints) [PRECOMPUTED]
        u: wp.array(dtype=wp.spatial_vector),  # (num_bodies) [OUTPUT]
    ):
        tile_idx = wp.tid()
        constraint_start = tile_idx * tile_size

        # 1. Load Precomputed Data
        # ------------------------
        # Instead of loading J (6 floats) AND M_inv (36 floats),
        # we only load W (6 floats).
        # Huge bandwidth saving.
        W_tile = wp.tile_load(W, shape=(tile_size,), offset=(constraint_start,))
        lambda_tile = wp.tile_load(lambda_, shape=(tile_size,), offset=(constraint_start,))

        # 2. Compute Velocity Delta Directly
        # --------------------------------
        # dv = W * lambda
        # (No matrix multiplication here, just vector scaling)
        dv_tile = wp.tile_map(wp.mul, W_tile, lambda_tile)

        # 3. Tile Reduction
        # -----------------
        # Sum up all velocity changes for bodies in this tile
        dv_by_body = wp.tile_reshape(dv_tile, shape=(bodies_in_tile, constraints_per_body))
        total_dv_per_body = wp.tile_reduce(wp.add, dv_by_body, axis=1)

        # 4. Store Result
        # ---------------
        # No M_inv multiplication needed at the end!
        wp.tile_store(u, total_dv_per_body, offset=(tile_idx * bodies_in_tile,))

    return kernel_solve_tiled_optimized


# =================================================================================
# 4. MAIN & VERIFICATION
# =================================================================================


def main():
    # Settings
    num_bodies = 1200
    constraints_per_body = 16
    bodies_in_tile = 8

    tile_size = bodies_in_tile * constraints_per_body
    num_constraints = num_bodies * constraints_per_body
    num_tiles = num_constraints // tile_size

    print(f"Simulation: {num_bodies} bodies, {num_constraints} contacts.")
    print(f"Optimization: Precomputing W = M_inv * J")

    # -----------------------------
    # 1. Construct test data
    # -----------------------------
    dlambda_np = np.random.randn(num_constraints).astype(np.float32)
    J_np = np.random.randn(num_constraints, 6).astype(np.float32)
    M_inv_np = np.random.randn(num_bodies, 6, 6).astype(np.float32)

    # Ensure M_inv is symmetric positive definite (for realism)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T

    # Map: [0,0... 1,1... ]
    constraint_to_body_np = np.repeat(np.arange(num_bodies, dtype=np.int32), constraints_per_body)

    # -----------------------------
    # 2. Upload to Warp
    # -----------------------------
    dlambda = wp.array(dlambda_np, dtype=wp.float32)
    J = wp.array(J_np, dtype=wp.spatial_vector)
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    constraint_to_body = wp.array(constraint_to_body_np, dtype=wp.int32)

    # Buffers
    W = wp.zeros_like(J)  # Container for precomputed weighted jacobian
    du_scatter = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # -----------------------------
    # 3. Run Reference (Scatter)
    # -----------------------------
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=num_constraints,
        inputs=[dlambda, J, constraint_to_body, M_inv],
        outputs=[du_scatter],
    )

    # -----------------------------
    # 4. Run Optimized (Precompute + Tiled)
    # -----------------------------

    # Phase A: Precompute W (Runs once per substep)
    wp.launch(
        kernel_precompute_W, dim=num_constraints, inputs=[J, constraint_to_body, M_inv], outputs=[W]
    )

    # Phase B: Solve (Runs many times per substep)
    kernel_solve = create_solve_tiled_contact_precomputed(
        num_bodies, constraints_per_body, bodies_in_tile
    )

    wp.launch_tiled(
        kernel=kernel_solve,
        dim=num_tiles,
        inputs=[dlambda, W],
        outputs=[du_tiled],
        block_dim=tile_size,
    )

    wp.synchronize()

    # -----------------------------
    # 5. Compare
    # -----------------------------
    out_scatter = du_scatter.numpy()
    out_tiled = du_tiled.numpy()

    max_err = np.max(np.abs(out_scatter - out_tiled))

    print("\n=== Result Comparison ===")
    print(f"Max abs error: {max_err:.8f}")
    if max_err < 1e-4:
        print("SUCCESS: Optimization matches reference exactly.")
    else:
        print("FAILURE: Mismatch detected.")


if __name__ == "__main__":
    main()
