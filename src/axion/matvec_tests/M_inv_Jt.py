import numpy as np
import warp as wp

wp.init()


@wp.kernel
def kernel_M_inv_Jt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),  # (num_constraints)
    J: wp.array(dtype=wp.spatial_vector, ndim=2),  # (num_constraints, 2)
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),  # (num_constraints, 2)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    du: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
):
    c_idx = wp.tid()

    b0_idx = constraint_to_body[c_idx, 0]
    b1_idx = constraint_to_body[c_idx, 1]

    if b0_idx >= 0:
        delta_impulse0 = J[c_idx, 0] * dlambda[c_idx]
        m0_inv = M_inv[b0_idx]
        wp.atomic_add(du, b0_idx, m0_inv * delta_impulse0)
    if b1_idx >= 0:
        delta_impulse1 = J[c_idx, 1] * dlambda[c_idx]
        m1_inv = M_inv[b1_idx]
        wp.atomic_add(du, b1_idx, m1_inv * delta_impulse1)


@wp.kernel
def kernel_M_inv_Jjt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),  # (num_constraints)
    J: wp.array(dtype=wp.spatial_vector, ndim=2),  # (num_constraints, 2)
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),  # (num_constraints, 2)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    du: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
):
    c_idx = wp.tid()

    b0_idx = constraint_to_body[c_idx, 0]
    b1_idx = constraint_to_body[c_idx, 1]

    delta_impulse0 = J[c_idx, 0] * dlambda[c_idx]
    delta_impulse1 = J[c_idx, 1] * dlambda[c_idx]

    m0_inv = M_inv[b0_idx]
    m1_inv = M_inv[b1_idx]

    wp.atomic_add(du, b0_idx, m0_inv * delta_impulse0)
    wp.atomic_add(du, b1_idx, m1_inv * delta_impulse1)


@wp.kernel
def kernel_M_inv_Jct_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),  # (num_constraints)
    J: wp.array(dtype=wp.spatial_vector),  # (num_constraints)
    constraint_to_body: wp.array(dtype=wp.int32),  # (num_constraints)
    M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
    du: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
):
    c_idx = wp.tid()

    b_idx = constraint_to_body[c_idx]
    delta_impulse = J[c_idx] * dlambda[c_idx]

    m_inv = M_inv[b_idx]

    wp.atomic_add(du, b_idx, m_inv * delta_impulse)


def create_M_inv_Jct_matvec_indexed_tiled(
    num_bodies: int,
    constraints_per_body: int,
    bodies_in_tile: int,
):
    assert (
        num_bodies % bodies_in_tile == 0
    ), "Number of bodies must be divisible by bodies in one tile."
    tile_size = bodies_in_tile * constraints_per_body
    num_constraints = num_bodies * constraints_per_body

    @wp.kernel
    def kernel_M_inv_Jct_matvec_indexed_tiled(
        lambda_: wp.array(dtype=wp.float32),  # (num_constraints)
        J: wp.array(dtype=wp.spatial_vector),  # (num_constraints)
        body_to_constraints: wp.array(dtype=wp.int32),  # (num_constraints)
        M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
        u: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
    ):
        tile_idx = wp.tid()
        body_offset = tile_idx * bodies_in_tile
        body_constr_offset = tile_idx * tile_size

        constr_indices = wp.tile_load(
            body_to_constraints,
            tile_size,
            offset=body_constr_offset,
            storage="shared",
        )
        jacobian_tile = wp.tile_load_indexed(J, indices=constr_indices, shape=(tile_size,))
        lambda_tile = wp.tile_load_indexed(lambda_, indices=constr_indices, shape=(tile_size,))
        m_inv_tile = wp.tile_load(M_inv, bodies_in_tile, offset=body_offset, storage="shared")

        impulse_tile = wp.tile_map(wp.mul, jacobian_tile, lambda_tile)
        impulse_by_body = wp.tile_reshape(
            impulse_tile, shape=(bodies_in_tile, constraints_per_body)
        )
        total_impulse_per_body = wp.tile_reduce(wp.add, impulse_by_body, axis=1)

        u_tile = wp.tile_map(wp.mul, m_inv_tile, total_impulse_per_body)
        wp.tile_store(u, u_tile, offset=tile_idx * bodies_in_tile)

    return kernel_M_inv_Jct_matvec_indexed_tiled


def create_M_inv_Jct_matvec_tiled(
    num_bodies: int,
    constraints_per_body: int,
    bodies_in_tile: int,
):
    assert (
        num_bodies % bodies_in_tile == 0
    ), "Number of bodies must be divisible by bodies in one tile."
    tile_size = bodies_in_tile * constraints_per_body
    num_constraints = num_bodies * constraints_per_body

    @wp.kernel
    def kernel_M_inv_Jct_matvec_tiled(
        lambda_: wp.array(dtype=wp.float32),  # (num_constraints)
        J: wp.array(dtype=wp.spatial_vector),  # (num_constraints)
        M_inv: wp.array(dtype=wp.spatial_matrix),  # (num_bodies)
        u: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
    ):
        tile_idx = wp.tid()
        constraint_start = tile_idx * tile_size

        jacobian_tile = wp.tile_load(J, tile_size, offset=constraint_start)
        lambda_tile = wp.tile_load(lambda_, tile_size, offset=constraint_start)

        impulse_tile = wp.tile_map(wp.mul, jacobian_tile, lambda_tile)
        impulse_by_body = wp.tile_reshape(
            impulse_tile, shape=(bodies_in_tile, constraints_per_body)
        )
        total_impulse_per_body = wp.tile_reduce(wp.add, impulse_by_body, axis=1)

        body_offset = tile_idx * bodies_in_tile
        m_inv_tile = wp.tile_load(M_inv, bodies_in_tile, offset=body_offset, storage="shared")
        u_tile = wp.tile_map(wp.mul, m_inv_tile, total_impulse_per_body)
        wp.tile_store(u, u_tile, offset=tile_idx * bodies_in_tile)

    return kernel_M_inv_Jct_matvec_tiled


def main():
    num_bodies = 1200
    joint_constraints_per_body = 2
    contact_constraints_per_body = 16
    num_joint_constraints = num_bodies * joint_constraints_per_body
    num_contact_constraints = num_bodies * contact_constraints_per_body
    num_total_constraints = num_joint_constraints + num_contact_constraints

    bodies_in_tile = 8
    tile_size = bodies_in_tile * contact_constraints_per_body
    num_tiles = num_contact_constraints // tile_size

    # -----------------------------
    # 1. Construct test data
    # -----------------------------

    M_inv_np = np.random.randn(num_bodies, 6, 6).astype(np.float32)

    # ---- Jacobians and lambdas ----
    dlambda_j_np = np.random.randn(num_joint_constraints).astype(np.float32)
    J_j_np = np.random.randn(num_joint_constraints, 2, 6).astype(np.float32)

    dlambda_c_np = np.random.randn(num_contact_constraints).astype(np.float32)
    J_c_np = np.random.randn(num_contact_constraints, 6).astype(np.float32)

    dlambda_all_np = np.concatenate([dlambda_j_np, dlambda_c_np])
    # Create a padded Jacobian for contacts: Shape (NumContacts, 2, 6)
    # Slot 0 gets the contact jacobian, Slot 1 gets Zeros
    J_c_padded = np.zeros((num_contact_constraints, 2, 6), dtype=np.float32)
    J_c_padded[:, 0, :] = J_c_np
    J_all_np = np.concatenate([J_j_np, J_c_padded], axis=0)

    # ---- Constraint to body maps ----
    joint_constraint_to_body_np = np.random.choice(num_bodies, size=(num_joint_constraints, 2))

    # Contact constraint to body map must match the tiled layout: [0,0,...,1,1,...,2,2,...]
    contact_constraint_to_body_np = np.repeat(
        np.arange(num_bodies, dtype=np.int32),
        contact_constraints_per_body,
    )
    contact_body_to_constraints_np = np.arange(num_contact_constraints, dtype=np.int32)

    # Create a padded map for contacts: [body_idx, -1]
    contact_map_padded = np.full((num_contact_constraints, 2), -1, dtype=np.int32)
    contact_map_padded[:, 0] = contact_constraint_to_body_np

    # Concatenated map [Joint Map] + [Contact Map]
    constraint_to_body_all_np = np.concatenate(
        [joint_constraint_to_body_np, contact_map_padded], axis=0
    )

    # -----------------------------
    # 2. Upload to Warp
    # -----------------------------
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)

    dlambda_j = wp.array(dlambda_j_np, dtype=wp.float32)
    J_j = wp.array(J_j_np, dtype=wp.spatial_vector)

    dlambda_c = wp.array(dlambda_c_np, dtype=wp.float32)
    J_c = wp.array(J_c_np, dtype=wp.spatial_vector)

    dlambda_all = wp.array(dlambda_all_np, dtype=wp.float32)
    J_all = wp.array(J_all_np, dtype=wp.spatial_vector)

    joint_constraint_to_body = wp.array(joint_constraint_to_body_np, dtype=wp.int32)
    contact_constraint_to_body = wp.array(contact_constraint_to_body_np, dtype=wp.int32)
    contact_body_to_constraints = wp.array(contact_body_to_constraints_np, dtype=wp.int32)
    constraint_to_body_all = wp.array(constraint_to_body_all_np, dtype=wp.int32)

    du_j_scatter = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_c_scatter = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_j_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_c_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_unified = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    stream_j = wp.Stream()
    stream_c = wp.Stream()

    event_j_scatter = wp.Event()
    event_j_tiled = wp.Event()
    # -----------------------------
    # 3. Launch both kernels
    # -----------------------------

    # Scatter
    with wp.ScopedStream(stream_j):
        wp.launch(
            kernel_M_inv_Jjt_matvec_scatter,
            dim=num_joint_constraints,
            inputs=[dlambda_j, J_j, joint_constraint_to_body, M_inv],
            outputs=[du_j_scatter],
        )
        wp.record_event(event_j_scatter)

    with wp.ScopedStream(stream_c):
        wp.launch(
            kernel_M_inv_Jct_matvec_scatter,
            dim=num_contact_constraints,
            inputs=[dlambda_c, J_c, contact_constraint_to_body, M_inv],
            outputs=[du_c_scatter],
        )
        wp.wait_event(event_j_scatter)
        du_scatter = du_j_scatter + du_c_scatter

    # Tiled
    kernel_Minv_Jct_matvec_tiled = create_M_inv_Jct_matvec_tiled(
        num_bodies=num_bodies,
        constraints_per_body=contact_constraints_per_body,
        bodies_in_tile=bodies_in_tile,
    )

    kernel_Minv_Jct_matvec_indexed_tiled = create_M_inv_Jct_matvec_indexed_tiled(
        num_bodies=num_bodies,
        constraints_per_body=contact_constraints_per_body,
        bodies_in_tile=bodies_in_tile,
    )

    with wp.ScopedStream(stream_j):
        wp.launch(
            kernel_M_inv_Jjt_matvec_scatter,
            dim=num_joint_constraints,
            inputs=[dlambda_j, J_j, joint_constraint_to_body, M_inv],
            outputs=[du_j_tiled],
        )
        wp.record_event(event_j_tiled)

    with wp.ScopedStream(stream_c):
        # wp.launch_tiled(
        #     kernel=kernel_Minv_Jct_matvec_tiled,
        #     dim=num_tiles,
        #     inputs=[dlambda_c, J_c, M_inv],
        #     outputs=[du_c_tiled],
        #     block_dim=tile_size,
        # )
        wp.launch_tiled(
            kernel=kernel_Minv_Jct_matvec_indexed_tiled,
            dim=num_tiles,
            inputs=[dlambda_c, J_c, contact_body_to_constraints, M_inv],
            outputs=[du_c_tiled],
            block_dim=tile_size,
        )

        wp.wait_event(event_j_tiled)
        du_tiled = du_j_tiled + du_c_tiled

    # Unified (old) kernel
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=num_total_constraints,
        inputs=[dlambda_all, J_all, constraint_to_body_all, M_inv],
        outputs=[du_unified],
    )

    # Ensure it's finished before comparing
    wp.synchronize()

    # -----------------------------
    # 4. Compare outputs
    # -----------------------------
    du_scatter_host = du_scatter.numpy()
    du_tiled_host = du_tiled.numpy()
    du_unified_host = du_unified.numpy()

    # Compare Scatter vs Tiled
    max_err_tiled = np.max(np.abs(du_scatter_host - du_tiled_host))

    # Compare Scatter vs Unified (Old)
    max_err_unified = np.max(np.abs(du_scatter_host - du_unified_host))

    print("\n=== Result Comparison ===")
    print(f"Max abs error (Scatter vs Tiled): {max_err_tiled:.6f}")
    print(f"Max abs error (Scatter vs Unified): {max_err_unified:.6f}")

    is_consistent = np.allclose(du_scatter_host, du_unified_host, atol=1e-4)
    print(f"Unified matches Scatter: {is_consistent}")


if __name__ == "__main__":
    main()
