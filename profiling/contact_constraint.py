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


def setup_data(num_bodies, num_contacts, device):
    """Generates random input and output arrays for the kernel benchmark."""
    # ... (Your setup_data function remains unchanged)
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
