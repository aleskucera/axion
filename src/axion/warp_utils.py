import numpy as np
import warp as wp


# Kernels for 1D arrays
@wp.kernel
def add_value_to_array_item_float32_1D_kernel(
    array: wp.array(dtype=wp.float32), idx: int, value: wp.float32
):
    array[idx] += value


@wp.kernel
def add_value_to_array_item_vec3_1D_kernel(
    array: wp.array(dtype=wp.vec3), idx: int, value: wp.vec3
):
    array[idx] += value


@wp.kernel
def add_value_to_array_item_spatial_vector_1D_kernel(
    array: wp.array(dtype=wp.spatial_vector), idx: int, value: wp.spatial_vector
):
    array[idx] += value


@wp.kernel
def add_value_to_array_item_transform_1D_kernel(
    array: wp.array(dtype=wp.transform), idx: int, value: wp.transform
):
    array[idx] += value


# Kernels for 2D arrays
@wp.kernel
def add_value_to_array_item_float32_2D_kernel(
    array: wp.array(dtype=wp.float32, ndim=2), idx1: int, idx2: int, value: wp.float32
):
    array[idx1, idx2] += value


@wp.kernel
def add_value_to_array_item_vec3_2D_kernel(
    array: wp.array(dtype=wp.vec3, ndim=2), idx1: int, idx2: int, value: wp.vec3
):
    array[idx1, idx2] += value


@wp.kernel
def add_value_to_array_item_spatial_vector_2D_kernel(
    array: wp.array(dtype=wp.spatial_vector, ndim=2),
    idx1: int,
    idx2: int,
    value: wp.spatial_vector,
):
    array[idx1, idx2] += value


@wp.kernel
def add_value_to_array_item_transform_2D_kernel(
    array: wp.array(dtype=wp.transform, ndim=2),
    idx1: int,
    idx2: int,
    value: wp.transform,
):
    array[idx1, idx2] += value


def add_value_to_array_item(array: wp.array, indices: tuple, value):
    # Validate inputs
    if not isinstance(array, wp.array):
        raise TypeError("array must be a Warp array")
    if not isinstance(indices, tuple):
        raise TypeError("indices must be a tuple")
    if len(indices) not in (1, 2):
        raise ValueError(
            f"Expected 1 or 2 indices for 1D or 2D array, got {len(indices)}"
        )
    if len(indices) != len(array.shape):
        raise ValueError(
            f"Expected {len(array.shape)} indices for array with shape {array.shape}, got {len(indices)}"
        )

    # Bounds checking
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= array.shape[i]:
            raise IndexError(
                f"Index {idx} out of bounds for dimension {i} with size {array.shape[i]}"
            )

    # Select kernel and validate value type
    if len(indices) == 1:
        idx = indices[0]
        if array.dtype == wp.float32:
            if not isinstance(value, (float, wp.float32)):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_float32_1D_kernel
            value = wp.float32(value)
            inputs = [array, idx, value]
        elif array.dtype == wp.vec3:
            if not isinstance(value, wp.vec3):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_vec3_1D_kernel
            inputs = [array, idx, value]
        elif array.dtype == wp.spatial_vector:
            if not isinstance(value, wp.spatial_vector):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_spatial_vector_1D_kernel
            inputs = [array, idx, value]
        elif array.dtype == wp.transform:
            if not isinstance(value, wp.transform):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_transform_1D_kernel
            inputs = [array, idx, value]
        else:
            raise NotImplementedError(f"Unsupported array dtype: {array.dtype}")
    else:  # len(indices) == 2
        idx1, idx2 = indices
        if array.dtype == wp.float32:
            if not isinstance(value, (float, wp.float32)):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_float32_2D_kernel
            value = wp.float32(value)
            inputs = [array, idx1, idx2, value]
        elif array.dtype == wp.vec3:
            if not isinstance(value, wp.vec3):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_vec3_2D_kernel
            inputs = [array, idx1, idx2, value]
        elif array.dtype == wp.spatial_vector:
            if not isinstance(value, wp.spatial_vector):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_spatial_vector_2D_kernel
            inputs = [array, idx1, idx2, value]
        elif array.dtype == wp.transform:
            if not isinstance(value, wp.transform):
                raise TypeError(
                    f"Value type {type(value)} does not match array dtype {array.dtype}"
                )
            kernel = add_value_to_array_item_transform_2D_kernel
            inputs = [array, idx1, idx2, value]
        else:
            raise NotImplementedError(f"Unsupported array dtype: {array.dtype}")

    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=1,
        inputs=inputs,
    )


def demonstrate_add_value_to_array_item():
    """
    Demonstrates the usage of add_value_to_array_item with 1D and 2D arrays of different types.
    """
    wp.init()

    # Example 1: 1D float32 array
    print("Example 1: 1D float32 array")
    array_1d_float = wp.array(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=wp.float32
    )
    print("Before:", array_1d_float.numpy())
    add_value_to_array_item(array_1d_float, (1,), 5.0)
    print("After adding 5.0 at index (1,):", array_1d_float.numpy())

    # Example 2: 2D float32 array
    print("\nExample 2: 2D float32 array")
    array_2d_float = wp.array(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype=wp.float32
    )
    print("Before:", array_2d_float.numpy())
    add_value_to_array_item(array_2d_float, (1, 0), 10.0)
    print("After adding 10.0 at index (1,0):", array_2d_float.numpy())

    # Example 3: 1D vec3 array
    print("\nExample 3: 1D vec3 array")
    array_vec3 = wp.array(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), dtype=wp.vec3
    )
    print("Before:", array_vec3.numpy())
    add_value_to_array_item(array_vec3, (0,), wp.vec3(0.0, 1.0, 0.0))
    print("After adding vec3(0.0, 1.0, 0.0) at index (0,):", array_vec3.numpy())

    # Example 4: 2D vec3 array
    print("\nExample 4: 2D vec3 array")
    array_2d_vec3 = wp.array(
        np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
    )
    print("Before:", array_2d_vec3.numpy())
    add_value_to_array_item(array_2d_vec3, (0, 1), wp.vec3(1.0, 0.0, 0.0))
    print("After adding vec3(1.0, 0.0, 0.0) at index (0,1):", array_2d_vec3.numpy())

    # Example 5: 1D spatial_vector array
    print("\nExample 5: 1D spatial_vector array")
    array_spatial = wp.array(
        np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        dtype=wp.spatial_vector,
    )
    print("Before:", array_spatial.numpy())
    add_value_to_array_item(
        array_spatial, (0,), wp.spatial_vector(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    )
    print(
        "After adding spatial_vector(0.0, 1.0, 0.0, 1.0, 0.0, 0.0) at index (0,):",
        array_spatial.numpy(),
    )

    # Example 6: 2D spatial_vector array
    print("\nExample 6: 2D spatial_vector array")
    array_2d_spatial = wp.array(
        np.array(
            [
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        dtype=wp.spatial_vector,
    )
    print("Before:", array_2d_spatial.numpy())
    add_value_to_array_item(
        array_2d_spatial, (1, 0), wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    print(
        "After adding spatial_vector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0) at index (1,0):",
        array_2d_spatial.numpy(),
    )

    # Example 7: Attempt to add to a 1D transform array (should raise error)
    print("\nExample 7: 1D transform array")
    array_transform = wp.array(
        [wp.transform([0.0, 0.0, 0.0], wp.quat_identity())], dtype=wp.transform
    )
    print("Before:", array_transform.numpy())
    add_value_to_array_item(
        array_transform, (0,), wp.transform([1.0, 0.0, 0.0], wp.quat_identity())
    )
    print(
        "After adding transform([1.0, 0.0, 0.0], quat_identity()) at index (0,):",
        array_transform.numpy(),
    )

    # Example 8: Attempt to add to a 2D transform array (should raise error)
    print("\nExample 8: 2D transform array (should raise error)")
    array_2d_transform = wp.array(
        np.array(
            [
                [
                    wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
                    wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
                ]
            ],
            dtype=object,
        ),
        dtype=wp.transform,
    )
    print("Before:", array_2d_transform.numpy())
    add_value_to_array_item(
        array_2d_transform,
        (0, 1),
        wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
    )
    print(
        "After adding transform([1.0, 0.0, 0.0], quat_identity()) at index (0,1):",
        array_2d_transform.numpy(),
    )


if __name__ == "__main__":
    demonstrate_add_value_to_array_item()
