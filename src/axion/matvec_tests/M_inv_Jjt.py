import numpy as np
import warp as wp

wp.init()

# =================================================================================
# 1. REFERENCE: SCATTER KERNEL (What you already have)
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
# 2. TILED KERNEL
# =================================================================================


@wp.func
def compute_lambda_idx(J_idx: int):
    # 1. Handle the safety clamp (prevent -1)
    #    If j_idx is 0, we use 1. (1-1 = 0).
    safe_val = wp.max(J_idx, 1) - 1

    # 2. Perform the shift: safe_val >> 1
    return wp.rshift(safe_val, 1)


def create_M_inv_Jjt_matvec_tiled(bodies_in_tile, constraints_per_body):

    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_M_inv_Jjt_matvec_tiled(
        # -- Compact Arrays (Original Data) --
        lambda_: wp.array(dtype=wp.float32),  # (num_constraints + 1)
        # -- Flattened/Expanded Arrays --
        # J_flat: Contains [0.0, J_0_A, J_0_B, J_1_A, J_1_B...]
        J: wp.array(dtype=wp.spatial_vector),  # (2 * num_constraints + 1)
        # -- Indirection Maps --
        # Maps Body -> List of indices in J_flat
        body_to_constraints: wp.array(dtype=wp.int32),  # (num_bodies * constraints_per_body)
        # -- Physics Data --
        M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
        du: wp.array(dtype=wp.spatial_vector),  # (num_bodies) - Output
    ):
        tile_idx = wp.tid()

        # 1. Identify where we are in the body list
        body_offset = tile_idx * bodies_in_tile

        # 2. Identify where we are in the "Body -> Constraint Map"
        #    This map is size (num_bodies * constraints_per_body)
        map_offset = tile_idx * tile_size

        # -----------------------------------------------------------------
        # STEP A: Load the map for this tile
        # This tells us: "For the bodies in this tile, which J entries do they need?"
        # -----------------------------------------------------------------
        # Shape of result: (tile_size,)
        J_indices = wp.tile_load(
            body_to_constraints,
            shape=(tile_size,),
            offset=(map_offset,),
            storage="shared",  # Usually faster for indices we reuse
        )

        lambda_indices = wp.tile_map(compute_lambda_idx, J_indices)
        lambda_tile = wp.tile_load_indexed(lambda_, indices=lambda_indices, shape=(tile_size,))

        # -----------------------------------------------------------------
        # STEP D: Load the Flattened Jacobian Vectors
        # -----------------------------------------------------------------
        J_tile = wp.tile_load_indexed(J, indices=J_indices, shape=(tile_size,))

        # -----------------------------------------------------------------
        # STEP E: Standard Math (Identical to Contact Kernel)
        # -----------------------------------------------------------------

        # Calculate impulse = J * lambda
        impulse_tile = wp.tile_map(wp.mul, J_tile, lambda_tile)

        # Reshape to sum per body
        impulse_by_body = wp.tile_reshape(
            impulse_tile, shape=(bodies_in_tile, constraints_per_body)
        )

        # Sum all constraints acting on this body
        total_impulse_per_body = wp.tile_reduce(wp.add, impulse_by_body, axis=1)

        # Load Mass Inverse
        m_inv_tile = wp.tile_load(M_inv, shape=(bodies_in_tile,), offset=(body_offset,))

        # Calculate Delta Velocity = M_inv * TotalImpulse
        u_tile = wp.tile_map(wp.mul, m_inv_tile, total_impulse_per_body)

        # Store
        wp.tile_store(du, u_tile, offset=body_offset)

    return kernel_M_inv_Jjt_matvec_tiled


# =================================================================================
# 3. HELPER: DATA CONVERSION (Graph -> Tiled Format)
# =================================================================================


def build_tiled_graph_data(
    num_bodies,
    max_constraints_per_body,
    J_np,  # (N_c, 2, 6)
    constr_map_np,  # (N_c, 2)
):
    """
    Converts standard joint data into the flattened/indirection format.
    Includes a "Null Constraint" at index 0 for padding safety.
    """
    num_constraints = len(J_np)

    # 1. Flatten J (and add a zero-entry at index 0 for padding)
    #    Real data starts at index 1.
    #    Layout: [Zero, J0_A, J0_B, J1_A, J1_B, ...]

    # Each constraint produces 2 entries in the flat array
    num_flat_entries = 1 + (num_constraints * 2)

    J_flat_np = np.zeros((num_flat_entries, 6), dtype=np.float32)
    # Fill J_flat (skipping index 0)
    # reshape J_np from (N, 2, 6) to (N*2, 6)
    J_flat_np[1:] = J_np.reshape(-1, 6)

    # 3. Build Body -> Flat Indices Map
    #    Shape: (NumBodies, ConstraintsPerBody)
    body_to_flat_indices = np.zeros((num_bodies, max_constraints_per_body), dtype=np.int32)

    # We use a counter to fill slots for each body
    body_counts = np.zeros(num_bodies, dtype=np.int32)

    for c_i in range(num_constraints):
        # Body A
        b0 = constr_map_np[c_i, 0]
        if b0 >= 0 and body_counts[b0] < max_constraints_per_body:
            slot = body_counts[b0]
            # Calculate flat index for (Constraint c_i, Side 0)
            # Index = 1 + (c_i * 2) + 0
            flat_idx = 1 + c_i * 2
            body_to_flat_indices[b0, slot] = flat_idx
            body_counts[b0] += 1

        # Body B
        b1 = constr_map_np[c_i, 1]
        if b1 >= 0 and body_counts[b1] < max_constraints_per_body:
            slot = body_counts[b1]
            # Calculate flat index for (Constraint c_i, Side 1)
            # Index = 1 + (c_i * 2) + 1
            flat_idx = 1 + c_i * 2 + 1
            body_to_flat_indices[b1, slot] = flat_idx
            body_counts[b1] += 1

    # Note: Unused slots in body_to_flat_indices remain 0.
    # Index 0 points to J_flat[0] (Zero) and lambda_map[0] -> lambda[0].
    # As long as J_flat[0] is zero, this adds 0 impulse. Correct.

    return J_flat_np, body_to_flat_indices.flatten()


def generate_constr_to_body(num_constraints, num_bodies, max_constraints_per_body, seed=None):
    rng = np.random.default_rng(seed)

    # Track how many times each value has been used
    counts = np.zeros(num_bodies, dtype=np.int32)

    arr = np.zeros((num_constraints, 2), dtype=np.int32)

    for i in range(num_constraints):
        # Values still available (count < M)
        available = np.where(counts < max_constraints_per_body)[0]

        if len(available) < 2:
            raise ValueError("Not enough available distinct values to form a pair.")

        # Choose first value
        v1 = rng.choice(available)

        # Choose second value that is != v1 and still available
        avail2 = available[available != v1]
        if len(avail2) == 0:
            raise ValueError("No valid second value available for row {}.".format(i))

        v2 = rng.choice(avail2)

        arr[i, 0] = v1
        arr[i, 1] = v2

        counts[v1] += 1
        counts[v2] += 1

    return arr


def generate_M_inv(num_bodies: int):
    M_inv = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    # Make M_inv symmetric/positive for realism (optional, just for math)
    for i in range(num_bodies):
        M_inv[i] = M_inv[i] @ M_inv[i].T
    return M_inv


# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================


def main():
    # Settings
    num_bodies = 1024
    num_joint_constraints = 4096  # Linear chain mostly

    # For tiling, we need to define a max capacity.
    # Since we generate random pairs, let's pick a safe number or sparse it out.
    # For a chain, 2 is enough. For random, let's say 8.
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
    # B. Run Scatter Kernel
    # -----------------------

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
    # C. Prepare & Run Tiled Kernel
    # -----------------------

    # 1. Convert Data to Option 1 Format
    J_flat_np, body_to_J_np = build_tiled_graph_data(
        num_bodies, constraints_per_body, J_np, constr_map_np
    )

    J_flat_wp = wp.array(J_flat_np, dtype=wp.spatial_vector)
    body_to_J_wp = wp.array(body_to_J_np, dtype=wp.int32)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # 2. Instantiate Kernel
    tiled_kernel = create_M_inv_Jjt_matvec_tiled(bodies_in_tile, constraints_per_body)

    num_tiles = num_bodies // bodies_in_tile
    assert num_bodies % bodies_in_tile == 0, "Align bodies for simple test"

    wp.launch_tiled(
        kernel=tiled_kernel,
        dim=[num_tiles],
        inputs=[dlambda_wp, J_flat_wp, body_to_J_wp, M_inv_wp],
        outputs=[du_tiled],
        block_dim=bodies_in_tile * constraints_per_body,
    )

    wp.synchronize()

    # -----------------------
    # D. Compare
    # -----------------------

    res_scatter = du_scatter.numpy()
    res_tiled = du_tiled.numpy()

    # Note: Floating point atomic adds order is non-deterministic.
    # Tiled reduction is deterministic. Small epsilon diffs are expected.
    diff = np.abs(res_scatter - res_tiled)
    max_err = np.max(diff)

    print(f"Comparison Results:")
    print(f"-------------------")
    print(f"Data shapes: {num_bodies} bodies, {num_joint_constraints} joints.")
    print(f"Max Absolute Error: {max_err:.8f}")

    if max_err < 1e-4:
        print("SUCCESS: Tiled implementation matches Scatter implementation.")
    else:
        print("WARNING: Mismatch detected.")
        # Debug helper:
        for i in range(min(10, num_bodies)):
            print(f"Body {i}: Scatter {res_scatter[i,0]:.4f} | Tiled {res_tiled[i,0]:.4f}")


if __name__ == "__main__":
    main()
