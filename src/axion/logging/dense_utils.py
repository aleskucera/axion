import warp as wp
from axion.types import SpatialInertia

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def update_J_dense(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Output array
    J_dense: wp.array(dtype=wp.float32, ndim=2),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    if body_a >= 0:
        body_idx = body_a * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, constraint_idx, body_idx + st_i, J_ia[st_i])

    if body_b >= 0:
        body_idx = body_b * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, constraint_idx, body_idx + st_i, J_ib[st_i])


@wp.kernel
def update_Hinv_dense_kernel(
    gen_inv_mass: wp.array(dtype=SpatialInertia),
    H_dense: wp.array(dtype=wp.float32, ndim=2),
):
    body_idx = wp.tid()

    if body_idx >= gen_inv_mass.shape[0]:
        return

    # Angular part, write the tensor of inertia inverse
    for i in range(wp.static(3)):
        for j in range(wp.static(3)):
            st_i = wp.static(i)
            st_j = wp.static(j)
            h_row = body_idx * 6 + st_i
            h_col = body_idx * 6 + st_j
            body_I_inv = gen_inv_mass.inertia[body_idx]
            H_dense[h_row, h_col] = body_I_inv[st_i, st_j]

    # Linear part, write the mass inverse
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        h_row = body_idx * 6 + 3 + st_i
        h_col = body_idx * 6 + 3 + st_i
        H_dense[h_row, h_col] = gen_inv_mass.m[body_idx]


@wp.kernel
def update_C_dense_kernel(
    C_values: wp.array(dtype=wp.float32),
    C_dense: wp.array(dtype=wp.float32, ndim=2),
):
    constraint_idx = wp.tid()
    if constraint_idx >= C_values.shape[0]:
        return

    # Fill the diagonal of the constraint matrix C_dense
    C_value = C_values[constraint_idx]
    C_dense[constraint_idx, constraint_idx] = C_value


def update_dense_matrices(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    # Clear matrices
    data.Hinv_dense.zero_()
    data.J_dense.zero_()
    data.C_dense.zero_()

    # Update H^-1 (inverse mass matrix)
    wp.launch(
        kernel=update_Hinv_dense_kernel,
        dim=dims.N_b,
        inputs=[data.gen_inv_mass],
        outputs=[data.Hinv_dense],
        device=device,
    )

    # Update J (constraint Jacobian)
    wp.launch(
        kernel=update_J_dense,
        dim=dims.con_dim,
        inputs=[
            data.constraint_body_idx,
            data.J_values,
        ],
        outputs=[data.J_dense],
        device=device,
    )

    # Update C (compliance matrix)
    wp.launch(
        kernel=update_C_dense_kernel,
        dim=dims.con_dim,
        inputs=[data.C_values],
        outputs=[data.C_dense],
        device=device,
    )


def get_system_matrix_numpy(
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    Hinv_np = (data.Hinv_dense.numpy(),)
    J_np = (data.J_dense.numpy(),)
    C_np = (data.C_dense.numpy(),)
    g_np = (data.g.numpy(),)
    h_np = (data.h.numpy(),)

    A = J_np @ Hinv_np @ J_np.T + C_np
    b = J_np @ Hinv_np @ g_np - h_np

    return A, b
