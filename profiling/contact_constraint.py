"""
Profiling script for the NVIDIA Warp normal contact constraint kernel.

This script is designed to be used with NVIDIA Nsight Compute (ncu) to analyze
the performance of the `contact_constraint_kernel` under two scenarios:
1. A standard 'vanilla' kernel launch using `wp.launch()`.
2. A CUDA graph replay using `wp.capture_launch()`.

It isolates a single kernel launch (after a warm-up) to provide a clean
and accurate target for profiling, avoiding the noise from loops and Python
overhead present in typical benchmark scripts.

Usage with Nsight Compute (ncu):

1. To profile the standard (`vanilla`) launch:
   $ ncu -o report_vanilla python <your_script_name>.py --mode vanilla

2. To profile the CUDA graph launch:
   $ ncu -o report_graph python <your_script_name>.py --mode graph

You can also adjust the problem size:
   $ ncu -o report_large_vanilla python <your_script_name>.py --mode vanilla --num-bodies 500 --num-contacts 10000
"""
import argparse
import time

import numpy as np
import warp as wp
from axion.contact_constraint import contact_constraint_kernel
from axion.utils import scaled_fisher_burmeister

wp.config.lineinfo = True


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
    b_rest = -restitution * wp.min(delta_v_n_prev, 0.0)

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

    It implements the complementarity condition `0 <= λ_n ⟂ G(v, λ) >= 0` using a
    Fisher-Burmeister function `φ(λ_n, G(v, λ)) = 0`.

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
        h: (con_dim,) Stores the constraint residuals. This kernel writes `φ_n` to `h[h_n_offset + tid]`.
        J_values: (con_dim, 2) Stores Jacobian blocks. This kernel writes `∂φ_n/∂v` into the relevant rows.
        C_values: (con_dim,) Stores compliance blocks. This kernel writes `∂φ_n/∂λ_n` into the relevant indices.
    """
    tid = wp.tid()

    # Ignore contacts with no penetration
    if contact_gap[tid] >= 0.0:
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

    # 3. Update `C` (diagonal compliance block of the system matrix: ∂φ/∂λ)
    C_values[C_n_offset + tid] = dphi_dlambda_n + 1e-6

    # 4. Update `J` (constraint Jacobian block of the system matrix: ∂φ/∂v)
    if body_a >= 0:
        offset = J_n_offset + tid
        J_values[offset, 0] = J_n_a

    if body_b >= 0:
        offset = J_n_offset + tid
        J_values[offset, 1] = J_n_b


def setup_data(num_bodies, num_contacts, device):
    """Generates random input and output arrays for the kernel benchmark."""
    N_b, N_c = num_bodies, num_contacts
    body_a_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=N_c, dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(
            0, N_b, size=np.sum(mask), dtype=np.int32
        )
        mask = body_a_indices == body_b_indices
    num_j_constraints = 0
    con_dim = num_j_constraints + N_c * 3
    data = {
        "body_qd": wp.array(
            np.random.rand(N_b, 6) - 0.5, dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(N_b, 6) - 0.5, dtype=wp.spatial_vector, device=device
        ),
        "contact_gap": wp.array(
            np.random.rand(N_c) * -0.1, dtype=wp.float32, device=device
        ),
        "J_contact_a": wp.array(
            np.random.rand(N_c, 3, 6) - 0.5,
            dtype=wp.spatial_vector,
            ndim=2,
            device=device,
        ),
        "J_contact_b": wp.array(
            np.random.rand(N_c, 3, 6) - 0.5,
            dtype=wp.spatial_vector,
            ndim=2,
            device=device,
        ),
        "contact_body_a": wp.array(body_a_indices, dtype=wp.int32, device=device),
        "contact_body_b": wp.array(body_b_indices, dtype=wp.int32, device=device),
        "contact_restitution_coeff": wp.array(
            np.random.rand(N_c) * 0.5, dtype=wp.float32, device=device
        ),
        "_lambda": wp.array(
            np.random.rand(con_dim) * 0.1, dtype=wp.float32, device=device
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


def main():
    """Main entry point for the profiling script."""
    parser = argparse.ArgumentParser(
        description="NCU profiling script for Warp kernel."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vanilla", "graph"],
        required=True,
        help="Select profiling mode: 'vanilla' for wp.launch or 'graph' for CUDA graph.",
    )
    parser.add_argument(
        "--num-bodies", type=int, default=1000, help="Number of rigid bodies."
    )
    parser.add_argument(
        "--num-contacts", type=int, default=10000, help="Number of contacts."
    )
    args = parser.parse_args()

    wp.init()
    device = wp.get_device()
    print(f"Initialized Warp on device: {device.name}")
    print(
        f"Configuration: mode={args.mode}, num_bodies={args.num_bodies}, num_contacts={args.num_contacts}"
    )

    # --- Data Setup ---
    data = setup_data(args.num_bodies, args.num_contacts, device)
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

    if args.mode == "vanilla":
        # --- Profile Standard `wp.launch` ---
        print("\n--- Preparing for vanilla wp.launch() profiling ---")
        print("Warming up kernel for JIT compilation...")
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=args.num_contacts,
            inputs=kernel_args,
            device=device,
        )
        wp.synchronize()
        print("Warm-up complete. Ready for profiling launch.")

        # This is the section ncu will profile
        # --- NCU PROFILING SECTION START ---
        data["g"].zero_()  # Ensure accumulator is clear before launch
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=args.num_contacts,
            inputs=kernel_args,
            device=device,
        )
        wp.synchronize()
        # --- NCU PROFILING SECTION END ---
        print("Profiling launch complete.")

    elif args.mode == "graph":
        # --- Profile CUDA Graph `wp.capture_launch` ---
        if not device.is_cuda:
            print("CUDA Graph profiling is only available on CUDA devices. Aborting.")
            return

        print("\n--- Preparing for CUDA graph wp.capture_launch() profiling ---")
        print("Capturing CUDA graph...")
        with wp.ScopedCapture() as capture:
            wp.launch(
                kernel=contact_constraint_kernel,
                dim=args.num_contacts,
                inputs=kernel_args,
                device=device,
            )
        graph = capture.graph

        print("Graph captured. Warming up graph replay...")
        wp.capture_launch(graph)
        wp.synchronize()
        print("Warm-up complete. Ready for profiling launch.")

        # This is the section ncu will profile
        # --- NCU PROFILING SECTION START ---
        data["g"].zero_()  # Ensure accumulator is clear before launch
        wp.capture_launch(graph)
        wp.synchronize()
        # --- NCU PROFILING SECTION END ---
        print("Profiling launch complete.")


if __name__ == "__main__":
    main()
