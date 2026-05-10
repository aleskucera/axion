import warp as wp


@wp.kernel
def project_contact_forces_kernel(
    constr_force_n: wp.array(dtype=wp.float32, ndim=2),
    constr_force_f: wp.array(dtype=wp.float32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
):
    """Project contact forces onto the feasible set.

    Enforces:
      - Signorini condition: normal force >= 0
      - Coulomb friction cone: ||f_t|| <= mu * f_n
    """
    world_idx, contact_idx = wp.tid()
    friction_idx0 = 2 * contact_idx
    friction_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        constr_force_n[world_idx, contact_idx] = 0.0
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        constr_force_n[world_idx, contact_idx] = 0.0
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    mu = (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1]) * 0.5

    # Signorini: clamp normal force to non-negative
    force_n = wp.max(constr_force_n[world_idx, contact_idx], 0.0)
    constr_force_n[world_idx, contact_idx] = force_n

    if force_n <= 0.0:
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    force_f = wp.vec2(
        constr_force_f[world_idx, friction_idx0],
        constr_force_f[world_idx, friction_idx1],
    )

    # Coulomb cone: project friction onto cone boundary if outside
    f_norm = wp.length(force_f)
    if f_norm > mu * force_n:
        scale = mu * force_n / f_norm
        constr_force_f[world_idx, friction_idx0] = constr_force_f[world_idx, friction_idx0] * scale
        constr_force_f[world_idx, friction_idx1] = constr_force_f[world_idx, friction_idx1] * scale
