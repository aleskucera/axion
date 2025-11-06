import warp as wp
from axion.types import JointConstraintData


@wp.kernel
def joint_constraint_kernel(
    # --- Iterative Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector),
    body_lambda_j: wp.array(dtype=wp.float32),
    joint_constraint_data: wp.array(dtype=JointConstraintData),
    # --- Parameters ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector),
    h_j: wp.array(dtype=wp.float32),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_j_values: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    c = joint_constraint_data[constraint_idx]
    lambda_j = body_lambda_j[constraint_idx]

    if not c.is_active:
        h_j[constraint_idx] = lambda_j
        J_hat_j_values[constraint_idx, 0] = wp.spatial_vector()
        J_hat_j_values[constraint_idx, 1] = wp.spatial_vector()
        C_j_values[constraint_idx] = 1.0
        return

    u_c = body_u[c.child_idx]
    u_p = wp.spatial_vector()
    if c.parent_idx >= 0:
        u_p = body_u[c.parent_idx]

    v_j = wp.dot(c.J_child, u_c) + wp.dot(c.J_parent, u_p)

    J_hat_child = c.J_child
    J_hat_parent = c.J_parent if c.parent_idx >= 0 else wp.spatial_vector()

    if c.parent_idx >= 0:
        wp.atomic_add(h_d, c.parent_idx, -J_hat_parent * lambda_j)
    wp.atomic_add(h_d, c.child_idx, -J_hat_child * lambda_j)

    h_j[constraint_idx] = v_j + upsilon / dt * c.value + compliance * lambda_j

    J_hat_j_values[constraint_idx, 0] = J_hat_parent
    J_hat_j_values[constraint_idx, 1] = J_hat_child

    C_j_values[constraint_idx] = compliance


@wp.kernel
def batch_joint_residual_kernel(
    # --- Iterative Inputs ---
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    joint_constraint_data: wp.array(dtype=JointConstraintData),
    # --- Parameters ---
    dt: wp.float32,
    upsilon: wp.float32,
    compliance: wp.float32,
    # --- Outputs ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
):
    batch_idx, constraint_idx = wp.tid()

    c = joint_constraint_data[constraint_idx]
    lambda_j = body_lambda_j[batch_idx, constraint_idx]

    if not c.is_active:
        h_j[batch_idx, constraint_idx] = 0.0
        return

    u_c = body_u[batch_idx, c.child_idx]
    u_p = wp.spatial_vector()
    if c.parent_idx >= 0:
        u_p = body_u[batch_idx, c.parent_idx]

    v_j = wp.dot(c.J_child, u_c) + wp.dot(c.J_parent, u_p)

    J_hat_child = c.J_child
    J_hat_parent = c.J_parent if c.parent_idx >= 0 else wp.spatial_vector()

    if c.parent_idx >= 0:
        wp.atomic_add(h_d, batch_idx, c.parent_idx, -J_hat_parent * lambda_j)
    wp.atomic_add(h_d, batch_idx, c.child_idx, -J_hat_child * lambda_j)

    h_j[batch_idx, constraint_idx] = v_j + upsilon / dt * c.value + compliance * lambda_j
