import warp as wp
from axion.types import JointConstraintData


@wp.kernel
def joint_constraint_kernel(
    # --- Iterative Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_j: wp.array(dtype=wp.float32),
    joint_constraint_data: wp.array(dtype=JointConstraintData),
    # --- Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Outputs ---
    g: wp.array(dtype=wp.spatial_vector),
    h_j: wp.array(dtype=wp.float32),
    J_j_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_j_values: wp.array(dtype=wp.float32),
):
    # Each thread processes one constraint axis for one joint
    constraint_idx = wp.tid()

    c = joint_constraint_data[constraint_idx]

    if not c.is_active:
        return

    body_qd_c = body_qd[c.child_idx]
    body_qd_p = wp.spatial_vector()
    if c.parent_idx >= 0:
        body_qd_p = body_qd[c.parent_idx]

    grad_c = wp.dot(c.J_child, body_qd_c) + wp.dot(c.J_parent, body_qd_p)
    bias = joint_stabilization_factor / dt * c.value

    lambda_current = lambda_j[constraint_idx]

    h_j[constraint_idx] = grad_c + bias
    C_j_values[constraint_idx] = c.compliance

    wp.atomic_add(g, c.child_idx, -c.J_child * lambda_current)
    J_j_values[constraint_idx, 1] = c.J_child
    if c.parent_idx >= 0:
        wp.atomic_add(g, c.parent_idx, -c.J_parent * lambda_current)
        J_j_values[constraint_idx, 0] = c.J_parent
    else:
        J_j_values[constraint_idx, 0] = wp.spatial_vector()


@wp.kernel
def linesearch_joint_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda_j: wp.array(dtype=wp.float32),
    # --- Iterative Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_j: wp.array(dtype=wp.float32),
    joint_constraint_data: wp.array(dtype=JointConstraintData),
    # --- Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Outputs ---
    g_alpha: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_alpha_j: wp.array(dtype=wp.float32, ndim=2),
):
    # Each thread processes one constraint axis for one joint
    alpha_idx, constraint_idx = wp.tid()

    alpha = alphas[alpha_idx]
    c = joint_constraint_data[constraint_idx]

    if not c.is_active:
        h_alpha_j[alpha_idx, constraint_idx] = 0.0
        return

    body_qd_c = body_qd[c.child_idx] + alpha * delta_body_qd[c.child_idx]
    body_qd_p = wp.spatial_vector()
    if c.parent_idx >= 0:
        body_qd_p = body_qd[c.parent_idx] + alpha * delta_body_qd[c.parent_idx]

    grad_c = wp.dot(c.J_child, body_qd_c) + wp.dot(c.J_parent, body_qd_p)
    bias = joint_stabilization_factor / dt * c.value

    h_alpha_j[alpha_idx, constraint_idx] = grad_c + bias

    lambda_current = lambda_j[constraint_idx] + alpha * delta_lambda_j[constraint_idx]

    g_alpha[alpha_idx, c.child_idx] += -c.J_child * lambda_current
    if c.parent_idx >= 0:
        g_alpha[alpha_idx, c.parent_idx] += -c.J_parent * lambda_current
