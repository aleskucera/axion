import time

import numpy as np
import warp as wp

from .utils import get_random_idx_to_res_buffer
from .utils import scaled_fisher_burmeister


@wp.func
def _compute_complementarity_argument(
    grad_c_n_a: wp.spatial_vector,
    grad_c_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    c_n: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    """
    Computes the argument 'b' for the Fisher-Burmeister function: FB(a, b) = 0.

    This value represents the desired velocity-level behavior at the contact point,
    incorporating relative velocity, Baumgarte stabilization to correct position
    errors, and restitution to handle bouncing.

    Args:
        grad_c_n_a: The Jacobian of the contact normal w.r.t. body A's velocity.
        grad_c_n_b: The Jacobian of the contact normal w.r.t. body B's velocity.
        body_qd_a: The current spatial velocity of body A.
        body_qd_b: The current spatial velocity of body B.
        body_qd_prev_a: The spatial velocity of body A at the previous timestep.
        body_qd_prev_b: The spatial velocity of body B at the previous timestep.
        c_n: The signed distance (gap) at the contact point. Negative for penetration.
        restitution: The coefficient of restitution for the contact.
        dt: The simulation timestep.
        stabilization_factor: The factor for Baumgarte stabilization (e.g., 0.1-0.2).

    Returns:
        The computed complementarity argument, which represents the target
        post-collision relative normal velocity plus stabilization terms.
    """
    # Relative normal velocity at the current time step (J * v), positive if separating
    delta_v_n = wp.dot(grad_c_n_a, body_qd_a) + wp.dot(grad_c_n_b, body_qd_b)

    # Relative normal velocity at the previous time step (for restitution)
    delta_v_n_prev = wp.dot(grad_c_n_a, body_qd_prev_a) + wp.dot(
        grad_c_n_b, body_qd_prev_b
    )

    # Baumgarte stabilization bias to correct penetration depth over time
    b_err = stabilization_factor / dt * c_n

    # Restitution bias based on pre-collision velocity
    # We only apply restitution if the pre-collision velocity is approaching.
    b_rest = restitution * wp.min(delta_v_n_prev, 0.0)

    return delta_v_n + b_err + b_rest


@wp.kernel
def contact_constraint_kernel_2D(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_qd_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32, ndim=2),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=3),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=3),
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    # --- Velocity Impulse Variables (from current Newton iterate) ---
    lambda_n_offset: wp.int32,  # Start index for normal impulses in `_lambda`
    _lambda: wp.array(dtype=wp.float32, ndim=2),
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,  # alpha for scaled_fisher_burmeister
    fb_beta: wp.float32,  # beta for scaled_fisher_burmeister
    # --- Offsets for Output Arrays ---
    h_n_offset: wp.int32,
    J_n_offset: wp.int32,
    C_n_offset: wp.int32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.float32, ndim=2),
    h: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_values: wp.array(dtype=wp.float32, ndim=2),
):
    batch_idx, contact_idx = wp.tid()

    # Contact that are not penetrating
    if contact_gap[batch_idx, contact_idx] >= 0.0:
        h[batch_idx, h_n_offset + contact_idx] = _lambda[
            batch_idx, lambda_n_offset + contact_idx
        ]
        C_values[batch_idx, C_n_offset + contact_idx] = 1.0
        J_values[batch_idx, J_n_offset + contact_idx, 0] = wp.spatial_vector()
        J_values[batch_idx, J_n_offset + contact_idx, 1] = wp.spatial_vector()
        return

    c_n = contact_gap[batch_idx, contact_idx]
    body_a = contact_body_a[batch_idx, contact_idx]
    body_b = contact_body_b[batch_idx, contact_idx]

    # The normal direction Jacobian is the first of the three (normal, tangent1, tangent2)
    grad_c_n_a = J_contact_a[batch_idx, contact_idx, 0]
    grad_c_n_b = J_contact_b[batch_idx, contact_idx, 0]

    e = contact_restitution_coeff[batch_idx, contact_idx]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    body_qd_prev_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[batch_idx, body_a]
        body_qd_prev_a = body_qd_prev[batch_idx, body_a]

    body_qd_b = wp.spatial_vector()
    body_qd_prev_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[batch_idx, body_b]
        body_qd_prev_b = body_qd_prev[batch_idx, body_b]

    # Compute the velocity-level term for the complementarity function
    complementarity_arg = _compute_complementarity_argument(
        grad_c_n_a,
        grad_c_n_b,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        c_n,
        e,
        dt,
        stabilization_factor,
    )

    # Get the current normal impulse from the global impulse vector
    lambda_n = _lambda[batch_idx, lambda_n_offset + contact_idx]

    # Evaluate the Fisher-Burmeister function and its derivatives
    phi_n, dphi_dlambda_n, dphi_db = scaled_fisher_burmeister(
        lambda_n, complementarity_arg, fb_alpha, fb_beta
    )

    # Jacobian of the constraint w.r.t body velocities (∂φ/∂v = ∂φ/∂b * ∂b/∂v)
    J_n_a = dphi_db * grad_c_n_a
    J_n_b = dphi_db * grad_c_n_b

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual)
    if body_a >= 0:
        g_a = -grad_c_n_a * lambda_n
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, batch_idx, body_a * 6 + st_i, g_a[st_i])

    if body_b >= 0:
        g_b = -grad_c_n_b * lambda_n
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, batch_idx, body_b * 6 + st_i, g_b[st_i])

    # 2. Update `h` (constraint violation residual)
    h[batch_idx, h_n_offset + contact_idx] = phi_n

    # 3. Update `C` (diagonal compliance block of the system matrix: ∂h/∂λ)
    C_values[batch_idx, C_n_offset + contact_idx] = (
        dphi_dlambda_n + 1e-5
    )  # Add a small constant for numerical stability

    # 4. Update `J` (constraint Jacobian block of the system matrix: ∂h/∂u)
    offset = J_n_offset + contact_idx
    if body_a >= 0:
        J_values[batch_idx, offset, 0] = J_n_a

    if body_b >= 0:
        J_values[batch_idx, offset, 1] = J_n_b
