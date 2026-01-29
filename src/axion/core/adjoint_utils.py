import warp as wp
from axion.math import compute_spatial_momentum
from axion.math import compute_world_inertia
from axion.math.kinematic_mapping import Gt_matvec  # Assuming your import


# -----------------------------------------------------------------------------
# 1. Body Initialization Kernel
#    Computes: w_u = M^-1 * (grad_u + h * G^T * grad_q)
# -----------------------------------------------------------------------------
@wp.kernel
def compute_body_adjoint_init_kernel(
    grad_q: wp.array(dtype=wp.transform, ndim=2),
    grad_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),  # Local Scalar
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),  # Local Matrix
    dt: float,
    # Output
    w_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()

    # 1. Fetch Gradients
    g_u = grad_u[world_idx, body_idx]
    g_q = grad_q[world_idx, body_idx]
    q_curr = body_q[world_idx, body_idx]

    # 2. Compute Driving Force: f_drive = grad_u + dt * G^T * grad_q
    f_drive = Gt_matvec(g_u, dt, g_q, q_curr)

    # 3. Prepare Inverse Inertia in World Frame
    m_inv = body_m_inv[world_idx, body_idx]
    I_loc_inv = body_I_inv[world_idx, body_idx]
    I_w_inv = compute_world_inertia(q_curr, I_loc_inv)

    # 4. Apply Inverse Mass to Force (M^-1 * f) using your momentum function
    # Effectively: accel = compute_spatial_momentum(m_inv, I_w_inv, f_drive)
    w_u[world_idx, body_idx] = compute_spatial_momentum(m_inv, I_w_inv, f_drive)


# -----------------------------------------------------------------------------
# 2. RHS Projection Kernel (Gather Bodies -> Constraints)
#    Computes: b = J * w_u
# -----------------------------------------------------------------------------
@wp.kernel
def compute_adjoint_rhs_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    w_u_body: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Output
    adjoint_rhs: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    if constraint_active_mask[world_idx, constraint_idx] == 0.0:
        adjoint_rhs[world_idx, constraint_idx] = 0.0
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    val = float(0.0)

    # Project: J * w_u
    if body_1 >= 0:
        J_1 = J_values[world_idx, constraint_idx, 0]
        w_1 = w_u_body[world_idx, body_1]
        val += wp.dot(J_1, w_1)

    if body_2 >= 0:
        J_2 = J_values[world_idx, constraint_idx, 1]
        w_2 = w_u_body[world_idx, body_2]
        val += wp.dot(J_2, w_2)

    adjoint_rhs[world_idx, constraint_idx] = val


# -----------------------------------------------------------------------------
# 3. Constraint Feedback Kernel (Scatter Constraints -> Bodies)
#    Computes: w_u -= M^-1 * (J^T * w_lambda)
# -----------------------------------------------------------------------------
@wp.kernel
def subtract_constraint_feedback_kernel(
    w_lambda: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Output (In-Place Update)
    w_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    if constraint_active_mask[world_idx, constraint_idx] == 0.0:
        return

    lam = w_lambda[world_idx, constraint_idx]

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]

    if body_1 >= 0:
        # 1. Compute Constraint Force: f_c = J^T * lambda
        f_c_1 = lam * J_1

        # 2. Transform Inertia
        q_1 = body_q[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_loc_inv_1 = body_I_inv[world_idx, body_1]
        I_w_inv_1 = compute_world_inertia(q_1, I_loc_inv_1)

        # 3. Apply Inverse Inertia: accel = compute_spatial_momentum(m_inv, I_w_inv, force)
        accel_1 = compute_spatial_momentum(m_inv_1, I_w_inv_1, f_c_1)

        # 4. Atomic Subtract
        wp.atomic_add(w_u, world_idx, body_1, -accel_1)

    if body_2 >= 0:
        f_c_2 = lam * J_2

        q_2 = body_q[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_loc_inv_2 = body_I_inv[world_idx, body_2]
        I_w_inv_2 = compute_world_inertia(q_2, I_loc_inv_2)

        accel_2 = compute_spatial_momentum(m_inv_2, I_w_inv_2, f_c_2)

        wp.atomic_add(w_u, world_idx, body_2, -accel_2)
