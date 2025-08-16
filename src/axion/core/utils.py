import warp as wp
from axion.constraints import get_constraint_body_index


@wp.func
def _compute_JHinvG_i(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    body_idx: int,
    J: wp.spatial_vector,
    g: wp.array(dtype=wp.float32),
):
    if body_idx < 0:
        return 0.0

    # H_inv @ g for the angular part
    g_ang_body = wp.vec3(g[body_idx * 6 + 0], g[body_idx * 6 + 1], g[body_idx * 6 + 2])
    Hinv_g_ang = body_inertia_inv[body_idx] @ g_ang_body

    # H_inv @ g for the linear part
    g_lin_body = wp.vec3(g[body_idx * 6 + 3], g[body_idx * 6 + 4], g[body_idx * 6 + 5])
    Hinv_g_lin = body_mass_inv[body_idx] * g_lin_body

    Hinv_g = wp.spatial_vector(Hinv_g_ang, Hinv_g_lin)

    return wp.dot(J, Hinv_g)


@wp.kernel
def update_system_rhs_kernel(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: wp.int32,
    J_n_offset: wp.int32,
    J_f_offset: wp.int32,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),  # Shape [N_c, 2]
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    tid = wp.tid()  # one thread per contact

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
