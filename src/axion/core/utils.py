import warp as wp
from axion.constraints import get_constraint_body_index
from axion.types import *  # Needed for the generic operations
from axion.types import GeneralizedMass


@wp.func
def _compute_JHinvG_i(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    body_idx: int,
    J: wp.spatial_vector,
    g: wp.array(dtype=wp.spatial_vector),
):
    if body_idx < 0:
        return 0.0

    top = body_inertia_inv[body_idx] @ wp.spatial_top(g[body_idx])
    bot = body_mass_inv[body_idx] * wp.spatial_bottom(g[body_idx])

    Hinv_g = wp.spatial_vector(top, bot)

    return wp.dot(J, Hinv_g)


@wp.kernel
def update_system_rhs_kernel2(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: wp.int32,
    J_n_offset: wp.int32,
    J_f_offset: wp.int32,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    g: wp.array(dtype=wp.spatial_vector),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    body_a, body_b = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        tid,
    )
    J_ia = J_values[tid, 0]
    J_ib = J_values[tid, 1]

    # Calculate (J_i * H^-1 * g)
    JHinvG = _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_a, J_ia, g)
    JHinvG += _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_b, J_ib, g)

    b[tid] = JHinvG - h[tid]


@wp.kernel
def update_system_rhs_kernel_old(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    g: wp.array(dtype=wp.spatial_vector),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    body_a = constraint_body_idx[tid, 0]
    body_b = constraint_body_idx[tid, 1]

    J_ia = J_values[tid, 0]
    J_ib = J_values[tid, 1]

    # Calculate (J_i * H^-1 * g)
    JHinvG = _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_a, J_ia, g)
    JHinvG += _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_b, J_ib, g)

    b[tid] = JHinvG - h[tid]


@wp.kernel
def update_system_rhs_kernel(
    Hinv: wp.array(dtype=GeneralizedMass),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    g: wp.array(dtype=wp.spatial_vector),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    # Calculate (J_i * H^-1 * g)
    a_contrib = wp.dot(J_ia, Hinv[body_a] * g[body_a])
    b_contrib = wp.dot(J_ib, Hinv[body_b] * g[body_b])
    JHinvg = a_contrib + b_contrib

    # b = J * H^-1 * g - h
    b[constraint_idx] = JHinvg - h[constraint_idx]
