"""
Test the possible difference between the scattered and tiled operations.

We have a matrix J of shape (num_constraints, num_bodies). Important thing is that
the matrix has only one non-zero value per row. We also can make it like that each
body has exactly constraints_per_body values. So we permute the rows so the
J is block diagonal matrix
"""
import numpy as np
import warp as wp

NUM_BODIES = 3200
CONSTRAINTS_PER_BODY = 16
TILE_SIZE = 128
BODIES_IN_TILE = 8  # (constriants_per_body // tile_size)
NUM_CONSTRAINTS = NUM_BODIES * CONSTRAINTS_PER_BODY

wp.init()


@wp.kernel
def kernel_J_matvec_scatter(
    x: wp.array(dtype=wp.float32),  # (num_bodies)
    J_values: wp.array(dtype=wp.float32),  # (num_constraints)
    constraint_body_idx: wp.array(dtype=wp.int32),  # (num_constraints)
    out: wp.array(dtype=wp.float32),  # (num_constraints)
):
    constraint_idx = wp.tid()

    body_idx = constraint_body_idx[constraint_idx]
    out[constraint_idx] = J_values[constraint_idx] * x[body_idx]


# @wp.kernel
# def kernel_J_matvec_tiled(
#     x: wp.array(dtype=wp.float32),  # (num_bodies)
#     J_values: wp.array(dtype=wp.float32),  # (num_constraints)
#     constraint_body_idx: wp.array(dtype=wp.int32),  # (num_constraints)
#     out: wp.array(dtype=wp.float32),  # (num_constraints)
# ):
#     body_idx = wp.tid()
#
#     constraint_offset = body_idx * CONSTRAINTS_PER_BODY
#
#     J_body_tile = wp.tile_load(
#         J_values,
#         CONSTRAINTS_PER_BODY,
#         offset=constraint_offset,
#     )
#     out_tile = J_body_tile * x[body_idx]
#     wp.tile_store(out, out_tile, offset=constraint_offset)


@wp.kernel
def kernel_J_matvec_tiled(
    x: wp.array(dtype=wp.float32),  # (num_bodies)
    J_values: wp.array(dtype=wp.float32),  # (num_constraints)
    constraint_body_idx: wp.array(dtype=wp.int32),  # (num_constraints)
    out: wp.array(dtype=wp.float32),  # (num_constraints)
):
    tid, block_idx = wp.tid()

    J_tile = wp.tile_load(
        J_values,
        TILE_SIZE,
        offset=tid * TILE_SIZE,
    )
    x_val = x[tid * BODIES_IN_TILE + block_idx // CONSTRAINTS_PER_BODY]
    out_tile = J_tile * x_val
    wp.tile_store(out, out_tile, offset=tid * TILE_SIZE)


def main():
    # -----------------------------
    # 1. Construct test data
    # -----------------------------

    # x is arbitrary vector of body values
    x_host = np.random.rand(NUM_BODIES).astype(np.float32)

    # J_values: each block of constraints-per-body rows belongs to one body
    J_values_host = np.random.rand(NUM_CONSTRAINTS).astype(np.float32)

    # constraint_body_idx must match the tiled layout: [0,0,...,1,1,...,2,2,...]
    constraint_body_idx_host = np.repeat(
        np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY
    )

    # -----------------------------
    # 2. Upload to Warp
    # -----------------------------
    x = wp.array(x_host, dtype=wp.float32)
    J_values = wp.array(J_values_host, dtype=wp.float32)
    constraint_body_idx = wp.array(constraint_body_idx_host, dtype=wp.int32)

    out_scatter = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)
    out_tiled = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)

    # -----------------------------
    # 3. Launch both kernels
    # -----------------------------
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x, J_values, constraint_body_idx, out_scatter],
    )

    # wp.launch_tiled(
    #     kernel_J_matvec_tiled,
    #     dim=NUM_BODIES,
    #     inputs=[x, J_values, constraint_body_idx, out_tiled],
    #     block_dim=CONSTRAINTS_PER_BODY,
    # )
    tiled_launch_dim = NUM_BODIES * CONSTRAINTS_PER_BODY // TILE_SIZE
    wp.launch_tiled(
        kernel=kernel_J_matvec_tiled,
        dim=tiled_launch_dim,
        inputs=[x, J_values, constraint_body_idx, out_tiled],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    # -----------------------------
    # 4. Compare outputs
    # -----------------------------
    out_scatter_host = out_scatter.numpy()
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
