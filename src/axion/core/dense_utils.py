import warp as wp
from axion.types import SpatialInertia

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_J_dense(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    # Output array
    J_dense: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, constraint_idx = wp.tid()

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]

    if body_1 >= 0:
        body_idx = body_1 * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, world_idx, constraint_idx, body_idx + st_i, J_1[st_i])

    if body_2 >= 0:
        body_idx = body_2 * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, world_idx, constraint_idx, body_idx + st_i, J_2[st_i])


@wp.kernel
def update_Minv_dense_kernel(
    M_inv: wp.array(dtype=SpatialInertia, ndim=2),
    M_inv_dense: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, body_idx = wp.tid()

    if body_idx >= M_inv.shape[1]:
        return

    # Angular part, write the tensor of inertia inverse
    for i in range(wp.static(3)):
        for j in range(wp.static(3)):
            st_i = wp.static(i)
            st_j = wp.static(j)
            h_row = body_idx * 6 + st_i
            h_col = body_idx * 6 + st_j
            body_I_inv = M_inv[world_idx, body_idx].inertia
            M_inv_dense[world_idx, h_row, h_col] = body_I_inv[st_i, st_j]

    # Linear part, write the mass inverse
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        h_row = body_idx * 6 + 3 + st_i
        h_col = body_idx * 6 + 3 + st_i
        M_inv_dense[world_idx, h_row, h_col] = M_inv[world_idx, body_idx].m


@wp.kernel
def update_C_dense_kernel(
    C_values: wp.array(dtype=wp.float32, ndim=2),
    C_dense: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, constraint_idx = wp.tid()
    if constraint_idx >= C_values.shape[1]:
        return

    # Fill the diagonal of the constraint matrix C_dense
    C_value = C_values[world_idx, constraint_idx]
    C_dense[world_idx, constraint_idx, constraint_idx] = C_value


def update_dense_matrices(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    # Clear matrices
    data.M_inv_dense.zero_()
    data.J_dense.zero_()
    data.C_dense.zero_()

    # Update H^-1 (inverse mass matrix)
    wp.launch(
        kernel=update_Minv_dense_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[data.world_M_inv],
        outputs=[data.M_inv_dense],
        device=device,
    )

    # Update J (constraint Jacobian)
    wp.launch(
        kernel=update_J_dense,
        dim=(dims.N_w, dims.N_c),
        inputs=[
            data.constraint_body_idx.full,
            data.J_values.full,
        ],
        outputs=[data.J_dense],
        device=device,
    )

    # Update C (compliance matrix)
    wp.launch(
        kernel=update_C_dense_kernel,
        dim=(dims.N_w, dims.N_c),
        inputs=[data.C_values.full],
        outputs=[data.C_dense],
        device=device,
    )


def get_system_matrix_numpy(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    Minv_np = data.M_inv_dense.numpy()
    J_np = data.J_dense.numpy()
    C_np = data.C_dense.numpy()

    A = J_np @ Minv_np @ J_np.transpose(0, 2, 1) + C_np

    return A
