"""
Defines the core NVIDIA Warp kernel for calculating unconstrained dynamics.

This module is the starting point for each step in the Non-Smooth Newton (NSN)
physics engine. Its sole purpose is to compute the momentum balance residual
for each body in the simulation, based on the discrete-time Newton-Euler
equations of motion. This residual vector, denoted `g`, represents the violation
of the laws of motion given the current state, before any constraint forces
(from contacts or joints) are applied.

The dynamics equation solved here is:
    g(v) = H * (v - v_prev) - f_ext * dt
where:
    - H is the generalized mass matrix (inertia block-diagonal matrix).
    - v is the current spatial velocity.
    - v_prev is the spatial velocity from the previous timestep.
    - f_ext represents all external forces (including gravity).
    - dt is the simulation timestep.

This `g` vector is a fundamental input to the subsequent constraint satisfaction steps.
"""
import time

import numpy as np
import warp as wp


@wp.kernel
def unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    # --- Body Property Inputs ---
    body_mass: wp.array(dtype=wp.float32),
    body_inertia: wp.array(dtype=wp.mat33),
    # --- Simulation Parameters ---
    dt: wp.float32,
    gravity: wp.vec3,
    # --- Output ---
    g: wp.array(dtype=wp.float32),
):
    """
    Computes the residual of the unconstrained Newton-Euler equations of motion.

    This kernel is launched once per body. It calculates the difference between the
    required change in momentum (left-hand side) and the applied external impulses
    (right-hand side) over a timestep. The result is the dynamics residual `g`.

    Args:
        body_qd: (N_b,) Current spatial velocities of all bodies.
        body_qd_prev: (N_b,) Spatial velocities from the previous timestep.
        body_f: (N_b,) Externally applied spatial forces (torques and linear forces).
        body_mass: (N_b,) Scalar mass of each body.
        body_inertia: (N_b,) 3x3 inertia tensor for each body in its local frame.
        dt: The simulation timestep duration.
        gravity: The global gravity vector.

    Outputs:
        g: (N_b * 6,) The dynamics residual vector. This kernel writes 6 values per body.
    """
    tid = wp.tid()

    # Decompose spatial vectors into angular (top) and linear (bottom) components
    w = wp.spatial_top(body_qd[tid])
    v = wp.spatial_bottom(body_qd[tid])
    w_prev = wp.spatial_top(body_qd_prev[tid])
    v_prev = wp.spatial_bottom(body_qd_prev[tid])
    t = wp.spatial_top(body_f[tid])
    f = wp.spatial_bottom(body_f[tid])

    m = body_mass[tid]
    I = body_inertia[tid]

    # Angular momentum balance: I * Δω - τ_ext * dt
    res_ang = I * (w - w_prev) - t * dt

    # Linear momentum balance: m * Δv - f_ext * dt
    # Gravity is treated as an external force f_g = m * g. The impulse is f_g * dt.
    res_lin = m * (v - v_prev) - f * dt - m * gravity * dt

    # Store the 6 components of the residual vector g for the current body
    g_b = wp.spatial_vector(res_ang, res_lin)
    for i in range(wp.static(6)):
        st_i = wp.static(i)
        g[tid * 6 + st_i] = g_b[st_i]


# =======================================================================
#
#   BENCHMARKING SUITE
#
# =======================================================================


def setup_data(num_bodies, device):
    """
    Generates random but physically plausible input arrays for the kernel benchmark.

    Args:
        num_bodies (int): The number of bodies to create data for.
        device (str): The Warp device ('cpu' or 'cuda') to create arrays on.
    """
    B = num_bodies

    # Generate random inertia tensors that are symmetric positive-definite
    rand_mat = np.random.rand(B, 3, 3)
    inertia_tensors = (rand_mat + rand_mat.transpose((0, 2, 1))) / 2.0  # Make symmetric
    inertia_tensors += np.expand_dims(
        np.identity(3) * 0.1, axis=0
    )  # Ensure positive-definite

    mass = np.random.rand(B).astype(np.float32) + 1.0

    data = {
        "body_qd": wp.from_numpy(
            (np.random.rand(B, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_qd_prev": wp.from_numpy(
            (np.random.rand(B, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_f": wp.from_numpy(
            np.zeros((B, 6), dtype=np.float32), dtype=wp.spatial_vector, device=device
        ),
        "body_mass": wp.from_numpy(mass, dtype=wp.float32, device=device),
        "body_inertia": wp.from_numpy(
            inertia_tensors.astype(np.float32), dtype=wp.mat33, device=device
        ),
        "params": {
            "dt": 1.0 / 60.0,
            "gravity": wp.vec3(0.0, -9.8, 0.0),
        },
        "g": wp.zeros(B * 6, dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(num_bodies, num_iterations=200):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Unconstrained Dynamics Kernel: N_b={num_bodies}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, device)
    params = data["params"]

    # Assemble the list of arguments in the exact order the kernel expects
    kernel_args = [
        data["body_qd"],
        data["body_qd_prev"],
        data["body_f"],
        data["body_mass"],
        data["body_inertia"],
        params["dt"],
        params["gravity"],
        data["g"],
    ]

    # --- Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    # Warm-up launch to compile the kernel
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=num_bodies,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- CUDA Graph Benchmark (only on GPU) ---
    # CUDA graphs significantly reduce CPU launch overhead for repeated calls.
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        wp.capture_begin()
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=kernel_args,
            device=device,
        )
        graph = wp.capture_end()

        # Warm-up launch
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
            "\nWarning: No CUDA device found. Performance will be limited and CUDA Graph test will be skipped."
        )

    # Benchmark with a range of body counts to observe scaling
    run_benchmark(num_bodies=100)
    run_benchmark(num_bodies=500)
    run_benchmark(num_bodies=1000)
    run_benchmark(num_bodies=4000)
    run_benchmark(num_bodies=10000)
