"""
Test the possible difference between the scattered and tiled operations.

We have a matrix J of shape (num_constraints, num_bodies). Important thing is that
the matrix has only one non-zero value per row. We also can make it like that each
body has exactly constraints_per_body values. So we permute the rows so the
J is block diagonal matrix
"""
import numpy as np
import warp as wp

NUM_BODIES = 1600
CONSTRAINTS_PER_BODY = 16  # max 6 contacts per body
TILE_SIZE = 16 * 6
BODIES_IN_TILE = 1  # (tile_size // (constraints_per_body * 6))
NUM_CONSTRAINTS = NUM_BODIES * CONSTRAINTS_PER_BODY

wp.init()


@wp.kernel
def kernel_J_matvec_scatter(
    x: wp.array(dtype=wp.spatial_vector),  # (num_bodies)
    J_values: wp.array(dtype=wp.spatial_vector),  # (num_constraints)
    constraint_body_idx: wp.array(dtype=wp.int32),  # (num_constraints)
    out: wp.array(dtype=wp.float32),  # (num_constraints)
):
    constraint_idx = wp.tid()

    body_idx = constraint_body_idx[constraint_idx]
    out[constraint_idx] = wp.dot(J_values[constraint_idx], x[body_idx])


@wp.kernel
def kernel_J_matvec_tiled(
    x: wp.array(dtype=wp.float32),  # (num_bodies * 6)
    J_values: wp.array(dtype=wp.float32),  # (num_constraints * 6)
    out: wp.array(dtype=wp.float32),  # (num_constraints)
):
    tid, block_idx = wp.tid()

    body_idx = tid

    J_tile = wp.tile_load(
        J_values,
        shape=(TILE_SIZE),
        offset=(body_idx * TILE_SIZE),
    )

    x_idx = body_idx * 6 + block_idx % 6
    x_val = x[x_idx]
    # wp.printf("TID: %d, block_idx: %d, x_idx: %d\n", tid, block_idx, x_idx)
    mul_tile = J_tile * x_val
    out_tile = wp.tile_reduce(
        wp.add, wp.tile_reshape(mul_tile, shape=(CONSTRAINTS_PER_BODY, 6)), axis=1
    )
    wp.tile_store(out, out_tile, offset=body_idx * CONSTRAINTS_PER_BODY)


def main():
    # -----------------------------
    # 1. Construct test data
    # -----------------------------

    # x is arbitrary vector of body values
    x_host = np.random.rand(NUM_BODIES * 6).astype(np.float32)

    # J_values: each block of constraints-per-body rows belongs to one body
    J_values_host = np.random.rand(NUM_CONSTRAINTS * 6).astype(np.float32)
    print(J_values_host)

    # constraint_body_idx must match the tiled layout: [0,0,...,1,1,...,2,2,...]
    constraint_body_idx_host = np.repeat(
        np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY
    )

    # -----------------------------
    # 2. Upload to Warp
    # -----------------------------
    x = wp.array(x_host, dtype=wp.float32)
    x_v = wp.array(x_host.reshape(NUM_BODIES, 6), dtype=wp.spatial_vector)
    J_values = wp.array(J_values_host, dtype=wp.float32)
    J_values_v = wp.array(J_values_host.reshape(NUM_CONSTRAINTS, 6), dtype=wp.spatial_vector)
    constraint_body_idx = wp.array(constraint_body_idx_host, dtype=wp.int32)

    out_scatter = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)
    out_tiled = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)

    # -----------------------------
    # 3. Launch both kernels
    # -----------------------------
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x_v, J_values_v, constraint_body_idx],
        outputs=[out_scatter],
    )

    wp.launch_tiled(
        kernel=kernel_J_matvec_tiled,
        dim=(NUM_BODIES,),
        inputs=[x, J_values],
        outputs=[out_tiled],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    # -----------------------------
    # 4. Compare outputs
    # -----------------------------
    out_scatter_host = out_scatter.numpy().flatten()
    out_tiled_host = out_tiled.numpy()
    print(out_scatter_host)
    print(out_tiled_host)

    max_abs_err = np.max(np.abs(out_scatter_host - out_tiled_host))
    same = np.allclose(out_scatter_host, out_tiled_host, atol=1e-6)

    print("\n=== Result Comparison ===")
    print("Max abs error :", max_abs_err)
    print("Outputs match :", same)


if __name__ == "__main__":
    main()
