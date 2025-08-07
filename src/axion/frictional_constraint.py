"""
Defines the core NVIDIA Warp kernel for processing frictional contact constraints.

This module works in tandem with the normal contact constraint to simulate
Coulomb friction. Its primary role is to compute the residuals and Jacobians
for tangential forces that oppose relative motion at a contact point.

The implementation uses a velocity-based formulation derived from the
complementarity condition of the friction cone (`|λ_f| <= µ * λ_n`). It calculates
the necessary outputs for the main Non-Smooth Newton (NSN) solver to find the
corrective friction impulses.
"""
import time

import numpy as np
import warp as wp
from axion.utils import scaled_fisher_burmeister


@wp.kernel
def frictional_constraint_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    # --- Pre-computed Contact Kinematics ---
    contact_gap: wp.array(dtype=wp.float32),
    J_contact_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_contact_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    contact_friction_coeff: wp.array(dtype=wp.float32),
    # --- Velocity Impulse Variables ---
    lambda_n_offset: wp.int32,
    lambda_f_offset: wp.int32,
    _lambda: wp.array(dtype=wp.float32),
    _lambda_prev: wp.array(dtype=wp.float32),
    # --- Simulation & Solver Parameters ---
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # --- Offsets for Output Arrays ---
    h_f_offset: wp.int32,
    J_f_offset: wp.int32,
    C_f_offset: wp.int32,
    # --- Outputs (contributions to the linear system) ---
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    """
    Computes residuals and Jacobians for frictional contact constraints.

    This kernel is launched once per potential contact. It models the Coulomb
    friction cone constraint: the magnitude of the friction impulse cannot exceed
    the coefficient of friction multiplied by the normal impulse.

    Args:
        body_qd: (N_b, 6) Current spatial velocities of all bodies.
        contact_gap: (N_c,) Signed distance for each contact. Used to check if contact is active.
        J_contact_a: (N_c, 3) Contact Jacobians for body A. Indices 1 and 2 represent the tangent vectors.
        J_contact_b: (N_c, 3) Contact Jacobians for body B.
        contact_body_a: (N_c,) Index of the first body in the contact pair.
        contact_body_b: (N_c,) Index of the second body.
        contact_friction_coeff: (N_c,) Coefficient of friction (µ) for each contact.
        lambda_n_offset: Integer offset to find normal impulses in `_lambda` array.
        lambda_f_offset: Integer offset to find friction impulses in `_lambda` array.
        _lambda: (con_dim,) Full vector of impulses from the current Newton iteration.
        _lambda_prev: (con_dim,) Full vector of impulses from the previous Newton iteration.
        fb_alpha/fb_beta: Parameters for the scaled Fisher-Burmeister function.
        h_f_offset: Start index in the global `h` vector for friction constraint residuals.
        J_f_offset: Start row in the global `J_values` matrix for friction constraint Jacobians.
        C_f_offset: Start index in the global `C_values` vector for friction compliance values.

    Outputs (written via atomic adds or direct indexing):
        g: (N_b * 6,) Accumulates generalized forces from friction impulses (`-J_f^T * λ_f`).
        h: (con_dim,) Stores the constraint residuals. Writes 2 residual values per contact.
        J_values: (con_dim, 2) Stores Jacobian blocks.
        C_values: (con_dim,) Stores compliance blocks.
    """
    tid = wp.tid()
    mu = contact_friction_coeff[tid]
    lambda_n = _lambda_prev[lambda_n_offset + tid]

    # Early exit for inactive contacts
    if contact_gap[tid] >= 0.0 or lambda_n * mu <= 1e-2:
        h[h_f_offset + 2 * tid] = _lambda[lambda_f_offset + 2 * tid]
        h[h_f_offset + 2 * tid + 1] = _lambda[lambda_f_offset + 2 * tid + 1]

        C_values[C_f_offset + 2 * tid] = 1.0
        C_values[C_f_offset + 2 * tid + 1] = 1.0

        J_values[J_f_offset + 2 * tid, 0] = wp.spatial_vector()
        J_values[J_f_offset + 2 * tid, 1] = wp.spatial_vector()
        J_values[J_f_offset + 2 * tid + 1, 0] = wp.spatial_vector()
        J_values[J_f_offset + 2 * tid + 1, 1] = wp.spatial_vector()
        return

    body_a = contact_body_a[tid]
    body_b = contact_body_b[tid]

    # Safely get body velocities (handles fixed bodies with index -1)
    body_qd_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]

    body_qd_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]

    # Tangent vectors are at index 1 and 2
    grad_c_t1_a = J_contact_a[tid, 1]
    grad_c_t2_a = J_contact_a[tid, 2]
    grad_c_t1_b = J_contact_b[tid, 1]
    grad_c_t2_b = J_contact_b[tid, 2]

    # Relative tangential velocity at the contact point
    v_t1_rel = wp.dot(grad_c_t1_a, body_qd_a) + wp.dot(grad_c_t1_b, body_qd_b)
    v_t2_rel = wp.dot(grad_c_t2_a, body_qd_a) + wp.dot(grad_c_t2_b, body_qd_b)
    v_rel = wp.vec2(v_t1_rel, v_t2_rel)
    v_rel_norm = wp.length(v_rel)

    # Current friction impulse from the global impulse vector
    lambda_f_t1 = _lambda[lambda_f_offset + 2 * tid]
    lambda_f_t2 = _lambda[lambda_f_offset + 2 * tid + 1]
    lambda_f = wp.vec2(lambda_f_t1, lambda_f_t2)
    lambda_f_norm = wp.length(lambda_f)

    # REGULARIZATION: Use the normal impulse from the previous Newton iteration
    # to define the friction cone size. We clamp it to a minimum value to
    # prevent the cone from collapsing on new contacts, which causes instability.
    # lambda_n = wp.max(
    #     _lambda_prev[lambda_n_offset + tid], 10.0
    # )  # TODO: Resolve this problem
    lambda_n = _lambda_prev[lambda_n_offset + tid]
    friction_cone_limit = mu * lambda_n

    # Use a non-linear complementarity function to relate slip speed and friction force
    phi_f, _, _ = scaled_fisher_burmeister(
        v_rel_norm, friction_cone_limit - lambda_f_norm, fb_alpha, fb_beta
    )

    # Compliance factor `w` relates the direction of slip to the friction impulse direction.
    # It becomes the off-diagonal block in the system matrix.
    # TODO: This can be really large number
    w = (v_rel_norm - phi_f) / (lambda_f_norm + phi_f + 1e-6)

    # --- Update global system components ---

    # 1. Update `g` (momentum balance residual: -J^T * λ)
    if body_a >= 0:
        g_a = -grad_c_t1_a * lambda_f_t1 - grad_c_t2_a * lambda_f_t2
        for i in range(wp.static(6)):
            wp.atomic_add(g, body_a * 6 + i, g_a[i])

    if body_b >= 0:
        g_b = -grad_c_t1_b * lambda_f_t1 - grad_c_t2_b * lambda_f_t2
        for i in range(wp.static(6)):
            wp.atomic_add(g, body_b * 6 + i, g_b[i])

    # 2. Update `h` (constraint violation residual)
    h[h_f_offset + 2 * tid] = v_t1_rel + w * lambda_f_t1
    h[h_f_offset + 2 * tid + 1] = v_t2_rel + w * lambda_f_t2

    # 3. Update `C` (diagonal compliance block of the system matrix)
    # This `w` value forms the coupling between the two tangential directions.
    C_values[C_f_offset + 2 * tid] = w + 1e-3
    C_values[C_f_offset + 2 * tid + 1] = w + 1e-3

    # 4. Update `J` (constraint Jacobian block of the system matrix)
    if body_a >= 0:
        offset_t1 = J_f_offset + 2 * tid
        offset_t2 = J_f_offset + 2 * tid + 1
        J_values[offset_t1, 0] = grad_c_t1_a
        J_values[offset_t2, 0] = grad_c_t2_a

    if body_b >= 0:
        offset_t1 = J_f_offset + 2 * tid
        offset_t2 = J_f_offset + 2 * tid + 1
        J_values[offset_t1, 1] = grad_c_t1_b
        J_values[offset_t2, 1] = grad_c_t2_b


# =======================================================================
#
#   BENCHMARKING SUITE
#
# =======================================================================


def setup_data(
    num_bodies,
    num_contacts,
    device,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """
    Generates random input and output arrays for the kernel benchmark.
    This is very similar to the setup for the normal contact constraint kernel.
    """
    N_b, N_c = num_bodies, num_contacts

    # Generate contact gaps
    gaps = np.random.rand(N_c) * -0.1
    num_inactive = int(N_c * inactive_contact_ratio)
    if num_inactive > 0:
        inactive_indices = np.random.choice(N_c, num_inactive, replace=False)
        gaps[inactive_indices] = np.random.rand(num_inactive) * 0.1

    # Generate body indices
    body_a_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(
            0, N_b, size=np.sum(mask), dtype=np.int32
        )
        mask = body_a_indices == body_b_indices

    num_fixed = int(N_c * fixed_body_ratio)
    if num_fixed > 0:
        fixed_indices = np.random.choice(N_c, num_fixed, replace=False)
        chooser = np.random.randint(0, 2, size=num_fixed)
        body_a_indices[fixed_indices[chooser == 0]] = -1
        body_b_indices[fixed_indices[chooser == 1]] = -1

    con_dim = N_c * 3  # Assuming 1 normal + 2 friction constraints per contact

    data = {
        "body_qd": wp.from_numpy(
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
        "contact_friction_coeff": wp.from_numpy(
            np.random.rand(N_c).astype(np.float32), device=device
        ),
        "_lambda": wp.from_numpy(
            (np.random.rand(con_dim) * 10.0).astype(np.float32), device=device
        ),
        "_lambda_prev": wp.from_numpy(
            (np.random.rand(con_dim) * 10.0).astype(np.float32), device=device
        ),
        "params": {
            "lambda_n_offset": 0,
            "lambda_f_offset": N_c,
            "fb_alpha": 0.25,
            "fb_beta": 0.25,
            "h_f_offset": N_c,
            "J_f_offset": N_c,
            "C_f_offset": N_c,
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
        f"\n--- Benchmarking: N_b={num_bodies}, N_c={num_contacts}, Inactive={inactive_contact_ratio:.0%}, Fixed={fixed_body_ratio:.0%}, Iters={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(
        num_bodies, num_contacts, device, inactive_contact_ratio, fixed_body_ratio
    )

    params = data["params"]
    kernel_args = [
        data["body_qd"],
        data["contact_gap"],
        data["J_contact_a"],
        data["J_contact_b"],
        data["contact_body_a"],
        data["contact_body_b"],
        data["contact_friction_coeff"],
        params["lambda_n_offset"],
        params["lambda_f_offset"],
        data["_lambda"],
        data["_lambda_prev"],
        params["fb_alpha"],
        params["fb_beta"],
        params["h_f_offset"],
        params["J_f_offset"],
        params["C_f_offset"],
        data["g"],
        data["h"],
        data["J_values"],
        data["C_values"],
    ]

    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        kernel=frictional_constraint_kernel,
        dim=num_contacts,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=frictional_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        wp.capture_begin()
        data["g"].zero_()
        wp.launch(
            kernel=frictional_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
            device=device,
        )
        graph = wp.capture_end()

        wp.capture_launch(graph)
        wp.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_iterations):
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
            "\nWarning: No CUDA device found. Performance will be poor and CUDA graph test will be skipped."
        )

    # --- Baseline Benchmarks (all active contacts, no fixed bodies) ---
    print("\n>>> Running Baseline Benchmarks (0% Inactive, 0% Fixed)...")
    run_benchmark(num_bodies=400, num_contacts=800)
    run_benchmark(num_bodies=1000, num_contacts=5000)

    # --- Divergence Benchmarks (inactive contacts) ---
    print("\n>>> Running Divergence Benchmarks (50% Inactive Contacts)...")
    run_benchmark(num_bodies=400, num_contacts=800, inactive_contact_ratio=0.5)

    # --- Fixed Body Benchmarks ---
    print("\n>>> Running Fixed Body Benchmarks (20% Fixed Bodies)...")
    run_benchmark(num_bodies=400, num_contacts=800, fixed_body_ratio=0.2)

    # --- Combined Scenario ---
    print("\n>>> Running Combined Scenario Benchmarks (50% Inactive, 20% Fixed)...")
    run_benchmark(
        num_bodies=400,
        num_contacts=800,
        inactive_contact_ratio=0.5,
        fixed_body_ratio=0.2,
    )
