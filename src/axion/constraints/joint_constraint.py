"""
Defines the core NVIDIA Warp kernel for processing joint constraints.

This module is a key component of a Non-Smooth Newton (NSN) physics engine,
responsible for enforcing the kinematic constraints imposed by joints.
The current implementation focuses on revolute joints, which restrict relative
motion between two bodies to a single rotational degree of freedom.

For each revolute joint, the kernel computes the residuals and Jacobians for
five constraints:
- Three translational constraints to lock the joint's position.
- Two rotational constraints to align the bodies, allowing rotation only
  around the specified joint axis.

These outputs are used by the main solver to compute corrective impulses that
maintain the joint connections. The computations are designed for parallel
execution on the GPU [nvidia.github.io/warp](https://nvidia.github.io/warp/).
"""
import warp as wp
from axion.types import get_joint_term
from axion.types import JointManifold


@wp.kernel
def joint_constraint_kernel(
    # --- Iterative Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    lambda_j: wp.array(dtype=wp.float32),
    manifolds: wp.array(dtype=JointManifold),  # Precomputed data
    # --- Simulation & Solver Parameters ---
    dt: wp.float32,
    joint_stabilization_factor: wp.float32,
    # --- Outputs ---
    g: wp.array(dtype=wp.spatial_vector),
    h_j: wp.array(dtype=wp.float32),
    J_j_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_j_values: wp.array(dtype=wp.float32),
):
    # This kernel is still launched per-constraint to fill the flat output arrays
    constraint_idx, joint_idx = wp.tid()

    manifold = manifolds[joint_idx]
    if not manifold.is_active:
        return

    # Get the precomputed term for this specific constraint axis
    term = get_joint_term(manifold, constraint_idx)

    # --- Velocity-dependent computations ---
    child_idx = manifold.child_idx
    parent_idx = manifold.parent_idx

    body_qd_c = body_qd[child_idx]
    body_qd_p = body_qd[parent_idx]

    # Relative velocity projected onto constraint axis
    grad_c = wp.dot(term.J_c, body_qd_c) + wp.dot(term.J_p, body_qd_p)

    # Baumgarte stabilization bias using the precomputed position error
    bias = joint_stabilization_factor / dt * term.error

    # --- Update global system components using precomputed values ---
    global_constraint_idx = joint_idx * 5 + constraint_idx

    # Update residual `h`
    h_j[global_constraint_idx] = grad_c + bias

    # Update Jacobian `J` and compliance `C`
    J_j_values[global_constraint_idx, 0] = term.J_p
    J_j_values[global_constraint_idx, 1] = term.J_c
    C_j_values[global_constraint_idx] = term.compliance

    # Update force accumulator `g`
    lambda_current = lambda_j[global_constraint_idx]
    wp.atomic_add(g, child_idx, -term.J_c * lambda_current)
    wp.atomic_add(g, parent_idx, -term.J_p * lambda_current)
