import warp as wp
from axion.types import *  # Needed for the generic operations
from axion.types import GeneralizedMass


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
