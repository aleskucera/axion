import numpy as np
import warp as wp
from warp.sparse import bsr_from_triplets

wp.init()

NUM_WORLDS = 200
NUM_JOINT_CONSTRAINTS_PER_WORLD = 3
NUM_BODIES_PER_WORLD = 4
MAX_GROUND_CONTACTS_PER_BODY = 16
MAX_CONTACTS_BETWEEN_BODIES_PER_WORLD = 8

NUM_JOINT_CONSTRAINTS = NUM_WORLDS * NUM_JOINT_CONSTRAINTS_PER_WORLD
NUM_BODIES = NUM_WORLDS * NUM_BODIES_PER_WORLD


class JacobianOperator:
    def __init__(
        self,
        device: wp.context.Device,
        num_worlds: int,
        num_bodies: int,
        num_joint_constraints: int,
        max_ground_contacts_per_body: int,
        max_contacts_between_bodies_per_world: int,
    ):
        self.device = device

        self.num_worlds = num_worlds
        self.num_bodies = num_bodies
        self.num_joint_constraints = num_joint_constraints
        self.max_ground_contacts_per_body = max_ground_contacts_per_body
        self.max_contacts_between_bodies_per_world = max_contacts_between_bodies_per_world

        self.J_j = None  # (num_joint_constraints, 6 * num_bodies)
        self.J_gc = None  # (3 * max_ground_contacts_per_body * num_bodies, 6 * num_bodies)
        self.J_bc = None  # (max_contacts_between_bodies_per_world * num_worlds, 6 * num_bodies)

        self._init_sparse_matrices()

        self.shape = (
            self.num_joint_constraints
            + 3 * self.max_ground_contacts_per_body * self.num_bodies
            + 3 * self.max_contacts_between_bodies_per_world * self.num_worlds,
            6 * self.num_bodies,
        )

    def _init_sparse_matrices(self):
        # Dense matrix shape (num_joint_constraints, 6 * num_bodies)
        J_j_block_shape = (1, 6)
        J_j_rows_list = []
        J_j_columns_list = []
        J_j_values_list = []
        for i in range(self.num_joint_constraints):
            J_j_rows_list += [i, i]
            rand_body_indices = np.random.choice(self.num_bodies, 2, replace=False)
            J_j_columns_list += rand_body_indices.tolist()
            rand_values = np.random.rand(2, J_j_block_shape[0], J_j_block_shape[1])
            J_j_values_list += rand_values.tolist()

        J_j_rows = wp.array(J_j_rows_list, dtype=wp.int32)
        J_j_columns = wp.array(J_j_columns_list, dtype=wp.int32)
        J_j_values = wp.array(J_j_values_list, dtype=wp.float32)
        self.J_j = bsr_from_triplets(
            rows_of_blocks=self.num_joint_constraints,  # Number of rows of blocks
            cols_of_blocks=self.num_bodies,  # Number of columns of blocks
            rows=J_j_rows,  # Row indices
            columns=J_j_columns,  # Column indices
            values=J_j_values,  # Block values
            prune_numerical_zeros=False,
        )

        # Dense matrix shape (3 * max_ground_contacts_per_body * num_bodies, 6 * num_bodies)
        J_gc_block_shape = (3 * self.max_ground_contacts_per_body, 6)
        J_gc_rows = wp.from_numpy(np.arange(self.num_bodies), dtype=wp.int32)
        J_gc_columns = wp.from_numpy(np.arange(self.num_bodies), dtype=wp.int32)
        J_gc_values = wp.array(
            np.random.rand(self.num_bodies, J_gc_block_shape[0], J_gc_block_shape[1]),
            dtype=wp.float32,
        )
        # self.J_gc = wp.sparse.bsr_diag(diag=J_gc_values)
        self.J_gc = bsr_from_triplets(
            rows_of_blocks=self.num_bodies,  # Number of rows of blocks
            cols_of_blocks=self.num_bodies,  # Number of columns of blocks
            rows=J_gc_rows,  # Row indices
            columns=J_gc_columns,  # Column indices
            values=J_gc_values,  # Block values
            prune_numerical_zeros=False,
        )

        # Dense matrix shape (max_contacts_between_bodies_per_world * num_worlds, 6 * num_bodies)
        J_bc_block_shape = (
            3 * self.max_contacts_between_bodies_per_world,
            self.num_bodies * 6 // self.num_worlds,
        )
        J_bc_rows = wp.from_numpy(np.arange(self.num_worlds), dtype=wp.int32)
        J_bc_columns = wp.from_numpy(np.arange(self.num_worlds), dtype=wp.int32)
        J_bc_values = wp.array(
            np.random.rand(
                self.num_worlds,
                J_bc_block_shape[0],
                J_bc_block_shape[1],
            ),
            dtype=wp.float32,
        )
        self.J_bc = bsr_from_triplets(
            rows_of_blocks=self.num_worlds,  # Number of rows of blocks
            cols_of_blocks=self.num_worlds,  # Number of columns of blocks
            rows=J_bc_rows,  # Row indices
            columns=J_bc_columns,  # Column indices
            values=J_bc_values,  # Block values
            prune_numerical_zeros=False,
        )

    @property
    def j_slice(self):
        return slice(0, self.num_joint_constraints)

    @property
    def gc_slice(self):
        return slice(
            self.num_joint_constraints,
            self.num_joint_constraints + 3 * self.max_ground_contacts_per_body * self.num_bodies,
        )

    @property
    def bc_slice(self):
        return slice(
            self.num_joint_constraints + 3 * self.max_ground_contacts_per_body * self.num_bodies,
            self.num_joint_constraints
            + 3 * self.max_ground_contacts_per_body * self.num_bodies
            + 3 * self.max_contacts_between_bodies_per_world * self.num_worlds,
        )

    def matvec(self, x, y, z, alpha, beta):
        assert x.shape[0] == self.shape[1], "Wrong x shape"
        assert y.shape[0] == self.shape[0], "Wrong y shape"

        if z.ptr != y.ptr and beta != 0.0:
            wp.copy(src=y, dest=z)

        wp.sparse.bsr_mv(self.J_j, x, z[self.j_slice], alpha, beta)
        wp.sparse.bsr_mv(self.J_gc, x, z[self.gc_slice], alpha, beta)
        wp.sparse.bsr_mv(self.J_bc, x, z[self.bc_slice], alpha, beta)


def bsr_to_dense(bsr):
    """
    Convert Warp BSR matrix to a dense NumPy matrix.
    Works for any block size and any sparsity pattern.
    """
    # Block and matrix sizes
    br, bc = bsr.block_shape
    n_block_rows = bsr.nrow
    n_block_cols = bsr.ncol

    M = n_block_rows * br
    N = n_block_cols * bc

    dense = np.zeros((M, N), dtype=np.float32)

    row_ptr = bsr.offsets.numpy()
    col_idx = bsr.columns.numpy()
    vals = bsr.values.numpy()  # shape: (nnzb, br, bc)

    # Build dense from BSR data
    for block_row in range(n_block_rows):
        start = row_ptr[block_row]
        end = row_ptr[block_row + 1]
        for bi in range(start, end):
            block_col = col_idx[bi]

            r0 = block_row * br
            c0 = block_col * bc

            dense[r0 : r0 + br, c0 : c0 + bc] = vals[bi]

    return dense


def test_operator_dense_equivalence():
    device = wp.get_device()

    op = JacobianOperator(
        device=device,
        num_worlds=NUM_WORLDS,
        num_bodies=NUM_BODIES,
        num_joint_constraints=NUM_JOINT_CONSTRAINTS,
        max_ground_contacts_per_body=MAX_GROUND_CONTACTS_PER_BODY,
        max_contacts_between_bodies_per_world=MAX_CONTACTS_BETWEEN_BODIES_PER_WORLD,
    )

    # random vector x
    x_np = np.random.rand(op.num_bodies * 6).astype(np.float32)
    x_wp = wp.array(x_np, dtype=wp.float32, device=device)

    # Warp output buffer
    y_wp = wp.zeros(op.shape[0], dtype=wp.float32, device=device)
    z_wp = wp.zeros(op.shape[0], dtype=wp.float32, device=device)

    # Warp matvec
    alpha = 1.0
    beta = 0.0
    op.matvec(x_wp, y_wp, z_wp, alpha, beta)
    z_wp_np = z_wp.numpy()

    # Convert all three BSR matrices to dense
    J_j = bsr_to_dense(op.J_j)
    J_gc = bsr_to_dense(op.J_gc)
    J_bc = bsr_to_dense(op.J_bc)

    # Combine them in the same order as your slices
    J_dense = np.vstack([J_j, J_gc, J_bc])

    # NumPy reference
    z_np = J_dense @ x_np

    # Compare
    diff = np.linalg.norm(z_wp_np - z_np)
    print("L2 difference:", diff)
    print("Max abs difference:", np.max(np.abs(z_wp_np - z_np)))

    if diff < 1e-4:
        print("✔ Matching matvec")
    else:
        print("✘ Mismatch")


def matrix_creation_showcase():
    rows = wp.array([0, 1, 2], dtype=wp.int32)  # Row indices
    cols = wp.array([0, 1, 2], dtype=wp.int32)  # Column indices
    vals = wp.array(
        [
            [[0], [1], [2], [3], [4], [5]],
            [[0], [1], [2], [3], [4], [5]],
            [[0], [1], [2], [3], [4], [5]],
        ],
        dtype=wp.float32,
    )  # Block values

    # Create BSR matrix
    A = bsr_from_triplets(
        rows_of_blocks=3,  # Number of rows of blocks
        cols_of_blocks=3,  # Number of columns of blocks
        rows=rows,  # Row indices
        columns=cols,  # Column indices
        values=vals,  # Block values
        prune_numerical_zeros=False,
    )

    print(A.values.numpy())


if __name__ == "__main__":
    test_operator_dense_equivalence()
