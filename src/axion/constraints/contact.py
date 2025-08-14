"""
Defines the core NVIDIA Warp kernel for processing normal contact constraints.

This module is a central component of a Non-Smooth Newton (NSN) physics engine.
Its primary responsibility is to compute the residuals and Jacobians associated with
non-penetration and restitution constraints. These outputs are then used by a
larger linear solver to find the corrective impulses and velocity changes
for all bodies in the simulation. The implementation uses a scaled
Fisher-Burmeister (FB) function to formulate the complementarity problem,
which is robust for handling contact forces.
"""
import time

import numpy as np
import warp as wp

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
def contact_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_restitution_coeff: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables (from current Newton iterate) ---
    lambda_n_offset: wp.int32,  # Start index for normal impulses in `_lambda`
    _lambda: wp.array(dtype=wp.float32),
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
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    """
    Computes residuals and Jacobians for normal contact constraints.

    This kernel is launched once per potential contact. For each active contact
    (i.e., where penetration has occurred), it calculates the contributions to
    the global linear system being solved by the Non-Smooth Newton engine.

    It implements the complementarity condition `0 <= λ_n ⟂ G(u, λ) >= 0` using a
    Fisher-Burmeister function `φ(λ_n, G(v, λ)) = 0 = h_n`.

    Args:
        body_qd: (N_b, 6) Current spatial velocities of all bodies.
        body_qd_prev: (N_b, 6) Spatial velocities of all bodies from the previous timestep.
        contact_gap: (N_c,) Signed distance for each contact. Negative means penetration.
        J_contact_a: (N_c, 3) Contact Jacobian (normal and tangents) for body A. We use index 0 for the normal.
        J_contact_b: (N_c, 3) Contact Jacobian for body B.
        contact_body_a: (N_c,) Index of the first body in the contact pair.
        contact_body_b: (N_c,) Index of the second body.
        contact_restitution_coeff: (N_c,) Coefficient of restitution for each contact.
        lambda_n_offset: Integer offset to locate normal impulses in the global `_lambda` array.
        _lambda: (con_dim,) Full vector of constraint impulses from the current Newton iteration.
        dt: Simulation timestep duration.
        stabilization_factor: Baumgarte stabilization coefficient.
        fb_alpha/fb_beta: Parameters for the scaled Fisher-Burmeister function.
        h_n_offset: Start index in the global `h` vector for normal constraint residuals.
        J_n_offset: Start row in the global `J_values` matrix for normal constraint Jacobians.
        C_n_offset: Start index in the global `C_values` vector for normal compliance values.

    Outputs (written via atomic adds or direct indexing):
        g: (N_b * 6,) Accumulates generalized forces. This kernel adds `-J_n^T * λ_n`.
        h: (con_dim,) Stores the constraint residuals. This kernel writes `h_n` to `h[h_n_offset + tid]`.
        J_values: (con_dim, 2) Stores Jacobian blocks. This kernel writes `∂h_n/∂u` into the relevant rows.
        C_values: (con_dim,) Stores compliance blocks. This kernel writes `∂h_n/∂λ_n` into the relevant indices.
    """
    tid = wp.tid()

    # Contact that are not penetrating
    if contact_gap[tid] >= 0.0:
        h[h_n_offset + tid] = _lambda[lambda_n_offset + tid]
        C_values[C_n_offset + tid] = 1.0
        J_values[J_n_offset + tid, 0] = wp.spatial_vector()
        J_values[J_n_offset + tid, 1] = wp.spatial_vector()
        return

    c_n = contact_gap[tid]
    body_a = contact_body_a[tid]
    body_b = contact_body_b[tid]

    # The normal direction Jacobian is the first of the three (normal, tangent1, tangent2)
    grad_c_n_a = J_contact_a[tid, 0]
    grad_c_n_b = J_contact_b[tid, 0]

    e = contact_restitution_coeff[tid]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    body_qd_prev_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]
        body_qd_prev_a = body_qd_prev[body_a]

    body_qd_b = wp.spatial_vector()
    body_qd_prev_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]
        body_qd_prev_b = body_qd_prev[body_b]

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
    lambda_n = _lambda[lambda_n_offset + tid]

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
            wp.atomic_add(g, body_a * 6 + st_i, g_a[st_i])

    if body_b >= 0:
        g_b = -grad_c_n_b * lambda_n
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, body_b * 6 + st_i, g_b[st_i])

    # 2. Update `h` (constraint violation residual)
    h[h_n_offset + tid] = phi_n

    # 3. Update `C` (diagonal compliance block of the system matrix: ∂h/∂λ)
    C_values[C_n_offset + tid] = (
        dphi_dlambda_n + 1e-5
    )  # Add a small constant for numerical stability

    # 4. Update `J` (constraint Jacobian block of the system matrix: ∂h/∂u)
    offset = J_n_offset + tid
    if body_a >= 0:
        J_values[offset, 0] = J_n_a

    if body_b >= 0:
        J_values[offset, 1] = J_n_b


def setup_data(
    num_bodies,
    num_contacts,
    device,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """
    Generates random input and output arrays for the kernel benchmark.

    Args:
        num_bodies (int): Total number of bodies in the simulation.
        num_contacts (int): Total number of potential contacts.
        device (str): The Warp device to create arrays on.
        inactive_contact_ratio (float): The fraction (0.0 to 1.0) of contacts
            that should be inactive (i.e., not penetrating).
        fixed_body_ratio (float): The fraction (0.0 to 1.0) of contacts
            that should involve a fixed body (index -1).
    """
    N_b, N_c = num_bodies, num_contacts

    # --- Generate Contact Gaps ---
    # Start with all contacts penetrating, then make a fraction of them inactive.
    gaps = np.random.rand(N_c) * -0.1  # All negative (active)
    num_inactive = int(N_c * inactive_contact_ratio)
    if num_inactive > 0:
        inactive_indices = np.random.choice(N_c, num_inactive, replace=False)
        gaps[inactive_indices] = (
            np.random.rand(num_inactive) * 0.1
        )  # Make them positive

    # --- Generate Body Indices ---
    # Start with all valid body indices, ensuring no self-contacts.
    body_a_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(
            0, N_b, size=np.sum(mask), dtype=np.int32
        )
        mask = body_a_indices == body_b_indices

    # Now, introduce fixed bodies (-1) for a fraction of contacts.
    num_fixed = int(N_c * fixed_body_ratio)
    if num_fixed > 0:
        fixed_indices = np.random.choice(N_c, num_fixed, replace=False)
        # For each chosen contact, randomly set either body A or B to be fixed.
        # This avoids creating static-static (-1 vs -1) contacts.
        chooser = np.random.randint(0, 2, size=num_fixed)
        body_a_indices[fixed_indices[chooser == 0]] = -1
        body_b_indices[fixed_indices[chooser == 1]] = -1

    # The rest of the setup is the same
    num_j_constraints = 0
    con_dim = num_j_constraints + N_c * 3

    # Warp data creation using wp.from_numpy for better interoperability [nvidia.github.io/warp](https://nvidia.github.io/warp/modules/interoperability.html#warp.from_numpy)
    data = {
        "body_qd": wp.from_numpy(
            (np.random.rand(N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_qd_prev": wp.from_numpy(
            (np.random.rand(N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "contact_gap": wp.from_numpy(gaps.astype(np.float32), device=device),
        "J_contact_a": wp.from_numpy(
            (np.random.rand(N_c, 3, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "J_contact_b": wp.from_numpy(
            (np.random.rand(N_c, 3, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "contact_body_a": wp.from_numpy(body_a_indices, device=device),
        "contact_body_b": wp.from_numpy(body_b_indices, device=device),
        "contact_restitution_coeff": wp.from_numpy(
            (np.random.rand(N_c) * 0.5).astype(np.float32), device=device
        ),
        "_lambda": wp.from_numpy(
            (np.random.rand(con_dim) * 0.1).astype(np.float32), device=device
        ),
        "params": {
            "lambda_n_offset": num_j_constraints,
            "dt": 0.01,
            "stabilization_factor": 0.2,
            "fb_alpha": 0.25,
            "fb_beta": 0.25,
            "h_n_offset": num_j_constraints,
            "J_n_offset": num_j_constraints,
            "C_n_offset": num_j_constraints,
        },
        "g": wp.zeros(N_b * 6, dtype=wp.float32, device=device),
        "h": wp.zeros(con_dim, dtype=wp.float32, device=device),
        "J_values": wp.zeros((con_dim, 2), dtype=wp.spatial_vector, device=device),
        "C_values": wp.zeros(con_dim, dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(
    num_bodies,
    num_contacts,
    num_iterations=200,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking: N_b={num_bodies}, N_c={num_contacts}, "
        f"Inactive={inactive_contact_ratio:.0%}, Fixed={fixed_body_ratio:.0%}, "
        f"Iters={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(
        num_bodies,
        num_contacts,
        device,
        inactive_contact_ratio,
        fixed_body_ratio,
    )
    # The rest of the function remains the same...
    params = data["params"]
    kernel_args = [
        data["body_qd"],
        data["body_qd_prev"],
        data["contact_gap"],
        data["J_contact_a"],
        data["J_contact_b"],
        data["contact_body_a"],
        data["contact_body_b"],
        data["contact_restitution_coeff"],
        params["lambda_n_offset"],
        data["_lambda"],
        params["dt"],
        params["stabilization_factor"],
        params["fb_alpha"],
        params["fb_beta"],
        params["h_n_offset"],
        params["J_n_offset"],
        params["C_n_offset"],
        data["g"],
        data["h"],
        data["J_values"],
        data["C_values"],
    ]
    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        kernel=contact_constraint_kernel,
        dim=num_contacts,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        data["g"].zero_()
        with wp.ScopedCapture() as capture:
            wp.launch(
                kernel=contact_constraint_kernel,
                dim=num_contacts,
                inputs=kernel_args,
                device=device,
            )
        graph = capture.graph
        wp.capture_launch(graph)
        wp.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            data["g"].zero_()
            wp.capture_launch(graph)
        wp.synchronize()
        graph_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
        print(f"   Avg. Time: {graph_launch_ms:.3f} ms")
        if graph_launch_ms > 0:
            speedup = standard_launch_ms / graph_launch_ms
            print(f"   Speedup from Graph: {speedup:.2f}x")
    else:
        print("2. Skipping CUDA Graph benchmark (not on a CUDA-enabled device).")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print(
            "Warning: No CUDA device found. Performance will be poor and CUDA graph test will be skipped."
        )

    # --- Baseline Benchmarks (all active contacts, no fixed bodies) ---
    print("\n>>> Running Baseline Benchmarks (0% Inactive, 0% Fixed)...")
    run_benchmark(num_bodies=400, num_contacts=800)
    run_benchmark(num_bodies=400, num_contacts=5000)

    # --- Divergence Benchmarks (inactive contacts) ---
    # This is the key test for comparing the branching vs. branchless kernels.
    print("\n>>> Running Divergence Benchmarks (50% Inactive Contacts)...")
    run_benchmark(
        num_bodies=400,
        num_contacts=800,
        inactive_contact_ratio=0.5,
    )

    # --- Fixed Body Benchmarks ---
    # This tests the logic for safely handling body index -1.
    print("\n>>> Running Fixed Body Benchmarks (20% Fixed Bodies)...")
    run_benchmark(
        num_bodies=400,
        num_contacts=800,
        fixed_body_ratio=0.2,
    )

    # --- Combined Scenario ---
    print("\n>>> Running Combined Scenario Benchmarks (50% Inactive, 20% Fixed)...")
    run_benchmark(
        num_bodies=400,
        num_contacts=800,
        inactive_contact_ratio=0.5,
        fixed_body_ratio=0.2,
    )
