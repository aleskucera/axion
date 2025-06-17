import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.sparse as wps


def bsr_to_dense_numpy(bsr_matrix: wps.BsrMatrix):
    """
    Converts a Warp BSR/CSR matrix to a dense NumPy array for visualization.

    Args:
        bsr_matrix: The Warp BSR/CSR matrix object.

    Returns:
        A dense NumPy array representing the full matrix.
    """
    # Ensure all Warp operations are complete before accessing data
    wp.synchronize()

    # Get the matrix propertie/home/kuceral4/school/diff-pbds
    total_rows, total_cols = bsr_matrix.shape
    block_rows, block_cols = bsr_matrix.block_shape
    num_block_rows = bsr_matrix.nrow

    # Copy the BSR data from the device (e.g., GPU) to the host (CPU) as NumPy arrays
    offsets_np = bsr_matrix.offsets.numpy()
    columns_np = bsr_matrix.columns.numpy()
    values_np = bsr_matrix.values.numpy()

    # Create an empty (zero-filled) dense matrix to fill
    # Use the scalar type of the blocks for the dense matrix
    dense_matrix = np.zeros(shape=(total_rows, total_cols))

    # Iterate through each block row
    for r in range(num_block_rows):
        # Find the start and end index for the non-zero blocks in this row
        block_idx_start = offsets_np[r]
        block_idx_end = offsets_np[r + 1]

        # Iterate through each non-zero block in this row
        for i in range(block_idx_start, block_idx_end):
            # Get the block's column index and its value
            c = columns_np[i]
            block_value = values_np[i]

            # Calculate the top-left coordinate in the dense matrix to place the block
            start_row = r * block_rows
            start_col = c * block_cols

            # Use NumPy slicing to place the block into the dense matrix
            dense_matrix[
                start_row : start_row + block_rows, start_col : start_col + block_cols
            ] = block_value

    return dense_matrix


# --- Setup from the previous tutorial ---
wp.init()

rows_of_blocks = 4
cols_of_blocks = 4
block_type = wp.mat(shape=(2, 2), dtype=wp.float32)

triplet_rows = wp.array([0, 1, 3], dtype=wp.int32)
triplet_cols = wp.array([1, 2, 3], dtype=wp.int32)
triplet_values = wp.array(
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 0.0], [0.0, 9.0]],
    ],
    dtype=block_type,
)

bsr_matrix = wps.bsr_from_triplets(
    rows_of_blocks=rows_of_blocks,
    cols_of_blocks=cols_of_blocks,
    rows=triplet_rows,
    columns=triplet_cols,
    values=triplet_values,
)
# --- End of setup ---

# Convert our BSR matrix to its dense form
dense_bsr_visual = bsr_to_dense_numpy(bsr_matrix)

print("--- Dense BSR Matrix (Full Form) ---")
print(dense_bsr_visual)

# For a better visual, we can use matplotlib to plot the matrix
plt.figure(figsize=(6, 6))
plt.imshow(dense_bsr_visual, cmap="viridis")
plt.title("Visualization of the 8x8 BSR Matrix")
plt.colorbar(label="Value")
plt.grid(True, which="both", color="white", linestyle="--", linewidth=0.5)
plt.xticks(np.arange(-0.5, 8, 2), np.arange(0, 9, 2))
plt.yticks(np.arange(-0.5, 8, 2), np.arange(0, 9, 2))
plt.show()
