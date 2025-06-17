import time
from typing import Tuple

import numpy as np
import warp as wp
import warp.context as wpc
from axion.ncp import scaled_fisher_burmeister_derivatives
from axion.ncp import scaled_fisher_burmeister_evaluate
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


# Helper function 1: Computes the average restitution coefficient.
@wp.func
def _compute_restitution_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
    """Computes the average coefficient of restitution for a contact pair."""
    e = 0.0
    num_bodies = 0
    if shape_a >= 0:
        e += shape_materials.restitution[shape_a]
        num_bodies += 1
    if shape_b >= 0:
        e += shape_materials.restitution[shape_b]
        num_bodies += 1

    if num_bodies > 0:
        e /= float(num_bodies)

    return e


# Helper function 2: Computes the core argument for the NCP function.
@wp.func
def _compute_complementarity_argument(
    J_n_a: wp.spatial_vector,
    J_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    gap: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    """
    Computes the second argument for the Fisher-Burmeister function,
    which represents the complementarity condition (v_n + bias).
    """
    # Relative normal velocity at the current time step
    delta_v_n = wp.dot(J_n_a, body_qd_a) - wp.dot(J_n_b, body_qd_b)

    # Relative normal velocity at the previous time step
    delta_v_n_prev = wp.dot(J_n_a, body_qd_prev_a) - wp.dot(J_n_b, body_qd_prev_b)

    # Baumgarte stabilization bias from penetration depth
    b_err = -(stabilization_factor / dt) * gap

    # Restitution bias from previous velocity
    b_rest = restitution * delta_v_n_prev

    return delta_v_n + b_err + b_rest


@wp.kernel
def compute_contact_residuals_and_derivatives_fused(
    # --- Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    contact_count: wp.array(dtype=wp.int32),
    lambda_n: wp.array(dtype=wp.float32),
    gap_function: wp.array(dtype=wp.float32),
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    # Scalar parameters
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    # --- Outputs (for all three results) ---
    res: wp.array(dtype=wp.float32),
    dres_n_dlambda_n: wp.array(dtype=wp.float32, ndim=2),
    dres_n_dbody_qd: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Computes contact residuals and their derivatives in a single, fused kernel.
    It uses helper functions to improve readability and maintainability.
    """
    tid = wp.tid()

    # Guard clause for threads outside the contact count.
    # This handles the "inactive constraint" case.
    if tid >= contact_count[0]:
        res[tid] = -lambda_n[tid]
        dres_n_dlambda_n[tid, tid] = -1.0
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    # Guard clause for self-contacts.
    if shape_a == shape_b:
        return

    # --- 1. Gather Inputs and Compute Intermediate Values ---

    # Get body indices
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Compute restitution using a helper function
    e = _compute_restitution_coefficient(shape_a, shape_b, shape_materials)

    # Compute the complementarity argument using a helper function
    complementarity_arg = _compute_complementarity_argument(
        J_n[tid, 0],
        J_n[tid, 1],  # Jacobians
        body_qd[body_a],
        body_qd[body_b],  # Velocities
        body_qd_prev[body_a],
        body_qd_prev[body_b],  # Previous velocities
        gap_function[tid],  # Gap
        e,  # Restitution
        dt,
        stabilization_factor,
    )

    # --- 2. Compute Final Outputs ---

    # Compute the residual using the NCP evaluation function
    res[tid] = scaled_fisher_burmeister_evaluate(
        lambda_n[tid], complementarity_arg, fb_alpha, fb_beta, fb_epsilon
    )

    # Compute NCP derivatives (called only ONCE)
    da, db = scaled_fisher_burmeister_derivatives(
        lambda_n[tid], complementarity_arg, fb_alpha, fb_beta, fb_epsilon
    )

    # Store the derivative with respect to lambda_n (a diagonal term)
    dres_n_dlambda_n[tid, tid] = da

    # Store the derivative with respect to body velocities
    for i in range(wp.static(6)):
        dres_n_dbody_qd[tid, body_a * 6 + i] = (
            db * J_n[tid, 0][i]
        )  # Contribution from body A
        dres_n_dbody_qd[tid, body_b * 6 + i] = (
            -db * J_n[tid, 1][i]
        )  # Contribution from body B


# =================================================================================
# PART 2: BENCHMARKING APPARATUS
# =================================================================================


def setup_fused_kernel_data(num_bodies, num_contacts, device):
    """Generates all necessary input and output arrays for the fused kernel."""
    B, C = num_bodies, num_contacts

    shape0 = np.random.randint(0, B, size=C, dtype=np.int32)
    shape1 = np.random.randint(0, B, size=C, dtype=np.int32)
    mask = shape0 == shape1
    while np.any(mask):
        shape1[mask] = np.random.randint(0, B, size=np.sum(mask), dtype=np.int32)
        mask = shape0 == shape1

    data = {
        # --- Inputs ---
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "shape_body": wp.array(np.arange(B, dtype=np.int32), dtype=int, device=device),
        "shape_materials": ModelShapeMaterials(),
        "contact_count": wp.array(
            [C], dtype=wp.int32, device=device
        ),  # Kept as array per requirement
        "lambda_n": wp.array(np.random.rand(C), dtype=wp.float32, device=device),
        "gap_function": wp.array(
            np.random.rand(C) * -0.1, dtype=wp.float32, device=device
        ),
        "J_n": wp.array(
            np.random.rand(C, 2, 6), dtype=wp.spatial_vector, device=device
        ),
        "contact_shape0": wp.array(shape0, dtype=wp.int32, device=device),
        "contact_shape1": wp.array(shape1, dtype=wp.int32, device=device),
        # --- Scalar Parameters ---
        "params": {
            "dt": 0.01,
            "stabilization_factor": 0.2,
            "fb_alpha": 0.25,
            "fb_beta": 0.25,
            "fb_epsilon": 1e-6,
        },
        # --- Outputs (pre-allocated) ---
        "res": wp.zeros(C, dtype=wp.float32, device=device),
        "dres_n_dlambda_n": wp.zeros((C, C), dtype=wp.float32, device=device),
        "dres_n_dbody_qd": wp.zeros((C, 6 * B), dtype=wp.float32, device=device),
    }

    data["shape_materials"].restitution = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    return data


def run_fused_kernel_benchmark(num_bodies, num_contacts, num_iterations=200):
    """Measures kernel execution time using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Fused Kernel: B={num_bodies}, C={num_contacts}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_fused_kernel_data(num_bodies, num_contacts, device)
    params = data["params"]

    kernel_inputs = [
        data["body_qd"],
        data["body_qd_prev"],
        data["shape_body"],
        data["shape_materials"],
        data["contact_count"],
        data["lambda_n"],
        data["gap_function"],
        data["J_n"],
        data["contact_shape0"],
        data["contact_shape1"],
        params["dt"],
        params["stabilization_factor"],
        params["fb_alpha"],
        params["fb_beta"],
        params["fb_epsilon"],
    ]
    kernel_outputs = [data["res"], data["dres_n_dlambda_n"], data["dres_n_dbody_qd"]]

    # --- 1. Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        compute_contact_residuals_and_derivatives_fused,
        dim=num_contacts,
        inputs=kernel_inputs,
        outputs=kernel_outputs,
        device=device,
    )
    wp.synchronize()  # Warm-up

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            compute_contact_residuals_and_derivatives_fused,
            dim=num_contacts,
            inputs=kernel_inputs,
            outputs=kernel_outputs,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark (only on GPU) ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        with wp.ScopedCapture() as capture:
            wp.launch(
                compute_contact_residuals_and_derivatives_fused,
                dim=num_contacts,
                inputs=kernel_inputs,
                outputs=kernel_outputs,
                device=device,
            )
        graph = capture.graph

        wp.capture_launch(graph)
        wp.synchronize()  # Warm-up

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
        print("2. Skipping CUDA Graph benchmark (not on a GPU device).")

    print("--------------------------------------------------------------------")


# =================================================================================
# PART 3: MAIN EXECUTION BLOCK
# =================================================================================

if __name__ == "__main__":
    wp.init()
    if "cuda" not in wp.get_device().name:
        print("Warning: No CUDA device found. Performance will be poor.")

    # Run benchmarks at various scales to observe performance characteristics
    run_fused_kernel_benchmark(num_bodies=100, num_contacts=100)
    run_fused_kernel_benchmark(num_bodies=500, num_contacts=500)
    run_fused_kernel_benchmark(num_bodies=1000, num_contacts=2000)
    run_fused_kernel_benchmark(num_bodies=2000, num_contacts=5000)
