"""Friction mode freezing for the adjoint backward pass.

After the forward Newton solve converges, the FB complementarity-derived friction
compliance (C_f = w/dt) may be poorly conditioned, leading to large constraint
residuals that invalidate the IFT assumption (R=0).

This module replaces the FB-derived friction properties with linearized versions
that are consistent with the converged contact mode (sliding or sticking),
ensuring the adjoint system is well-conditioned.

See docs/adjoint_warm_start_issue.md for full background.
"""

import warp as wp


@wp.kernel
def freeze_friction_mode_kernel(
    # Friction constraint data (indexed by friction pair: 2 rows per contact)
    constr_active_mask_f: wp.array(dtype=wp.float32, ndim=2),
    C_values_f: wp.array(dtype=wp.float32, ndim=2),
    res_c_f: wp.array(dtype=wp.float32, ndim=2),
    # Converged forces
    constr_force_f: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n: wp.array(dtype=wp.float32, ndim=2),
    # Material
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Config
    friction_compliance: wp.float32,
):
    """Replace FB friction compliance with a fixed compliance for the adjoint.

    For each active friction contact:
      - Zero the residual (so the IFT assumption R=0 holds)
      - Replace C_f with a fixed compliance based on the contact mode:
        * Sliding (|λ_f| ≈ μ·λ_n): small compliance (force is determined)
        * Sticking (|λ_f| < μ·λ_n): small compliance (velocity is zero)
        * Inactive (λ_n ≈ 0): deactivate the constraint
    """
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        # Deactivate out-of-range friction constraints
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # Get converged forces
    lambda_t1 = constr_force_f[world_idx, 2 * contact_idx + 0]
    lambda_t2 = constr_force_f[world_idx, 2 * contact_idx + 1]
    lambda_n = constr_force_n[world_idx, contact_idx]

    # Friction coefficient
    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]
    mu = 0.5 * (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1])

    lambda_f_norm = wp.sqrt(lambda_t1 * lambda_t1 + lambda_t2 * lambda_t2)
    coulomb_limit = mu * lambda_n

    # If normal force is negligible, deactivate friction entirely
    if lambda_n < 1e-6:
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # For both sliding and sticking: use fixed compliance and zero residual.
    # The J values (tangential direction Jacobians) are kept from compute_linear_system.
    c_adj = friction_compliance

    C_values_f[world_idx, 2 * contact_idx + 0] = c_adj
    C_values_f[world_idx, 2 * contact_idx + 1] = c_adj

    # Zero the residual so IFT holds
    res_c_f[world_idx, 2 * contact_idx + 0] = 0.0
    res_c_f[world_idx, 2 * contact_idx + 1] = 0.0
