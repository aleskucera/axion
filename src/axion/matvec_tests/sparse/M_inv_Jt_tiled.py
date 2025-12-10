import numpy as np
import warp as wp

wp.init()


@wp.kernel
def kernel_M_inv_Jt_matvec_scatter(
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


def create_M_inv_Jt_matvec_tiled(
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
    def kernel_M_inv_Jt_matvec_tiled(
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
        m_inv_tile = wp.tile_load(M_inv, bodies_in_tile, offset=body_offset)
        u_tile = wp.tile_map(wp.mul, m_inv_tile, total_impulse_per_body)
        wp.tile_store(u, u_tile, offset=tile_idx * bodies_in_tile)

    return kernel_M_inv_Jt_matvec_tiled


def main():
    num_bodies = 1200
    constraints_per_body = 16
    bodies_in_tile = 8
    tile_size = bodies_in_tile * constraints_per_body
    num_constraints = num_bodies * constraints_per_body
    num_tiles = num_constraints // tile_size

    # -----------------------------
    # 1. Construct test data
    # -----------------------------
    dlambda_np = np.random.randn(num_constraints).astype(np.float32)
    J_np = np.random.randn(num_constraints, 6).astype(np.float32)
    M_inv_np = np.random.randn(num_bodies, 6, 6).astype(np.float32)

    # Constraint to body map must match the tiled layout: [0,0,...,1,1,...,2,2,...]
    constraint_to_body = np.repeat(np.arange(num_bodies, dtype=np.int32), constraints_per_body)

    # -----------------------------
    # 2. Upload to Warp
    # -----------------------------
    dlambda = wp.array(dlambda_np, dtype=wp.float32)
    J = wp.array(J_np, dtype=wp.spatial_vector)
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    constraint_to_body = wp.array(constraint_to_body, dtype=wp.int32)

    du_scatter = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_tiled = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # -----------------------------
    # 3. Launch both kernels
    # -----------------------------
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=num_constraints,
        inputs=[dlambda, J, constraint_to_body, M_inv],
        outputs=[du_scatter],
    )

    k_tiled = create_M_inv_Jt_matvec_tiled(
        num_bodies=num_bodies,
        constraints_per_body=constraints_per_body,
        bodies_in_tile=bodies_in_tile,
    )

    wp.launch_tiled(
        kernel=k_tiled,
        dim=num_tiles,
        inputs=[dlambda, J, M_inv],
        outputs=[du_tiled],
        block_dim=tile_size,
    )

    # -----------------------------
    # 4. Compare outputs
    # -----------------------------
    out_scatter_host = du_scatter.numpy()
    out_tiled_host = du_tiled.numpy()

    max_abs_err = np.max(np.abs(out_scatter_host - out_tiled_host))
    same = np.allclose(out_scatter_host, out_tiled_host, atol=1e-4)

    print("\n=== Result Comparison ===")
    print("Max abs error :", max_abs_err)
    print("Outputs match :", same)


if __name__ == "__main__":
    main()
