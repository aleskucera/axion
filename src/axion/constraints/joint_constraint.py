import warp as wp
from axion.types import get_joint_axis_kinematics
from axion.types import JointInteraction


@wp.kernel
def joint_constraint_kernel(
    # --- Iterative Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_j: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=JointInteraction),
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
    constraint_axis_idx, joint_idx = wp.tid()

    interaction = interactions[joint_idx]

    if not interaction.is_active:
        return

    axis_data = get_joint_axis_kinematics(interaction, constraint_axis_idx)

    child_idx = interaction.child_idx
    parent_idx = interaction.parent_idx

    body_qd_c = body_qd[child_idx]
    body_qd_p = body_qd[parent_idx]

    grad_c = wp.dot(axis_data.J_child, body_qd_c) + wp.dot(
        axis_data.J_parent, body_qd_p
    )
    bias = joint_stabilization_factor / dt * axis_data.error

    global_constraint_idx = joint_idx * 5 + constraint_axis_idx

    h_j[global_constraint_idx] = grad_c + bias
    J_j_values[global_constraint_idx, 0] = axis_data.J_parent
    J_j_values[global_constraint_idx, 1] = axis_data.J_child
    C_j_values[global_constraint_idx] = axis_data.compliance

    lambda_current = lambda_j[global_constraint_idx]
    wp.atomic_add(g, child_idx, -axis_data.J_child * lambda_current)
    wp.atomic_add(g, parent_idx, -axis_data.J_parent * lambda_current)


@wp.kernel
def linesearch_joint_residuals_kernel(
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda_j: wp.array(dtype=wp.float32),
    # --- Iterative Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_j: wp.array(dtype=wp.float32),
    interactions: wp.array(dtype=JointInteraction),
    # --- Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Outputs ---
    g_alpha: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_alpha_j: wp.array(dtype=wp.float32, ndim=2),
):
    # Each thread processes one constraint axis for one joint
    alpha_idx, constraint_axis_idx, joint_idx = wp.tid()
    global_constraint_idx = joint_idx * 5 + constraint_axis_idx

    interaction = interactions[joint_idx]
    alpha = alphas[alpha_idx]

    if not interaction.is_active:
        h_alpha_j[alpha_idx, global_constraint_idx] = 0.0
        return

    axis_data = get_joint_axis_kinematics(interaction, constraint_axis_idx)

    child_idx = interaction.child_idx
    parent_idx = interaction.parent_idx

    body_qd_c = body_qd[child_idx] + alpha * delta_body_qd[child_idx]
    body_qd_p = body_qd[parent_idx] + alpha * delta_body_qd[parent_idx]

    grad_c = wp.dot(axis_data.J_child, body_qd_c) + wp.dot(
        axis_data.J_parent, body_qd_p
    )
    bias = joint_stabilization_factor / dt * axis_data.error

    h_alpha_j[alpha_idx, global_constraint_idx] = grad_c + bias

    lambda_current = (
        lambda_j[global_constraint_idx] + alpha * delta_lambda_j[global_constraint_idx]
    )

    g_alpha[alpha_idx, child_idx] += -axis_data.J_child * lambda_current
    g_alpha[alpha_idx, parent_idx] += -axis_data.J_parent * lambda_current
