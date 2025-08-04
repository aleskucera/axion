"""
Defines the core Taichi kernel for calculating unconstrained dynamics.

This script is a Taichi implementation of the NVIDIA Warp kernel for
calculating the momentum balance residual `g`. It is designed to be
directly comparable in performance and functionality.

The dynamics equation solved here is identical:
    g(v) = H * (v - v_prev) - f_ext * dt
where:
    - H is the generalized mass matrix (inertia block-diagonal matrix).
    - v is the current spatial velocity.
    - v_prev is the spatial velocity from the previous timestep.
    - f_ext represents all external forces (including gravity).
    - dt is the simulation timestep.
"""
import time

import numpy as np
import taichi as ti

# Define a 6D vector type for spatial quantities, similar to wp.spatial_vector
spatial_vector_ti = ti.types.vector(6, ti.f32)


@ti.kernel
def unconstrained_dynamics_kernel_ti(
    # --- Body State Inputs ---
    body_qd: ti.types.ndarray(dtype=spatial_vector_ti, ndim=1),
    body_qd_prev: ti.types.ndarray(dtype=spatial_vector_ti, ndim=1),
    body_f: ti.types.ndarray(dtype=spatial_vector_ti, ndim=1),
    # --- Body Property Inputs ---
    body_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
    body_inertia: ti.types.ndarray(dtype=ti.types.matrix(3, 3, ti.f32), ndim=1),
    # --- Simulation Parameters ---
    dt: ti.f32,
    gravity: ti.types.vector(3, ti.f32),
    # --- Output ---
    g: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    """
    Computes the residual of the unconstrained Newton-Euler equations of motion.

    This kernel is launched once per body. It calculates the difference between the
    required change in momentum and the applied external impulses.

    Args:
        body_qd: (N_b,) Current spatial velocities of all bodies.
        body_qd_prev: (N_b,) Spatial velocities from the previous timestep.
        body_f: (N_b,) Externally applied spatial forces.
        body_mass: (N_b,) Scalar mass of each body.
        body_inertia: (N_b,) 3x3 inertia tensor for each body.
        dt: The simulation timestep duration.
        gravity: The global gravity vector.

    Outputs:
        g: (N_b * 6,) The dynamics residual vector.
    """
    # Taichi kernels implicitly parallelize over the 0-th dimension of the first argument.
    for tid in body_qd:
        # Decompose spatial vectors into angular (top) and linear (bottom) components
        w = body_qd[tid][:3]
        v = body_qd[tid][3:]
        w_prev = body_qd_prev[tid][:3]
        v_prev = body_qd_prev[tid][3:]
        t = body_f[tid][:3]
        f = body_f[tid][3:]

        m = body_mass[tid]
        I = body_inertia[tid]

        # Angular momentum balance: I * Δω - τ_ext * dt
        res_ang = I @ (w - w_prev) - t * dt

        # Linear momentum balance: m * Δv - f_ext * dt
        # Gravity is treated as an external force f_g = m * g. The impulse is f_g * dt.
        res_lin = m * (v - v_prev) - f * dt - m * gravity * dt

        # Store the 6 components of the residual vector g for the current body
        for i in ti.static(range(3)):
            g[tid * 6 + i] = res_ang[i]
            g[tid * 6 + 3 + i] = res_lin[i]


# =======================================================================
#
#   BENCHMARKING SUITE
#
# =======================================================================


def setup_data(num_bodies):
    """
    Generates random but physically plausible input arrays for the kernel benchmark.

    This version creates Taichi Ndarrays for use with the Taichi kernel.

    Args:
        num_bodies (int): The number of bodies to create data for.
        arch (ti.Arch): The Taichi backend (ti.cpu or ti.gpu) to use.
    """
    B = num_bodies

    # Generate random inertia tensors that are symmetric positive-definite
    rand_mat = np.random.rand(B, 3, 3)
    inertia_tensors = (rand_mat + rand_mat.transpose((0, 2, 1))) / 2.0
    inertia_tensors += np.expand_dims(np.identity(3) * 0.1, axis=0)

    mass = np.random.rand(B).astype(np.float32) + 1.0

    # Create Taichi Ndarrays and copy data from NumPy
    data = {}
    data["body_qd"] = ti.Vector.ndarray(6, ti.f32, shape=B)
    data["body_qd"].from_numpy((np.random.rand(B, 6) - 0.5).astype(np.float32))

    data["body_qd_prev"] = ti.Vector.ndarray(6, ti.f32, shape=B)
    data["body_qd_prev"].from_numpy((np.random.rand(B, 6) - 0.5).astype(np.float32))

    data["body_f"] = ti.Vector.ndarray(6, ti.f32, shape=B)
    data["body_f"].from_numpy(np.zeros((B, 6), dtype=np.float32))

    data["body_mass"] = ti.ndarray(ti.f32, shape=B)
    data["body_mass"].from_numpy(mass)

    data["body_inertia"] = ti.Matrix.ndarray(3, 3, ti.f32, shape=B)
    data["body_inertia"].from_numpy(inertia_tensors.astype(np.float32))

    data["params"] = {
        "dt": 1.0 / 60.0,
        "gravity": ti.Vector([0.0, -9.8, 0.0], dt=ti.f32),
    }
    data["g"] = ti.ndarray(ti.f32, shape=B * 6)
    data["g"].fill(0)

    return data


def run_benchmark(num_bodies, num_iterations=200):
    """Measures execution time of the Taichi kernel."""
    print(
        f"\n--- Benchmarking Unconstrained Dynamics Kernel (Taichi): N_b={num_bodies}, Iterations={num_iterations} ---"
    )

    data = setup_data(num_bodies)
    params = data["params"]

    # Assemble the list of arguments
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

    # Warm-up launch to trigger JIT compilation
    unconstrained_dynamics_kernel_ti(*kernel_args)
    ti.sync()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        unconstrained_dynamics_kernel_ti(*kernel_args)
    ti.sync()  # Crucial: wait for all GPU work to finish before stopping the timer

    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")


if __name__ == "__main__":
    # Initialize Taichi to use the best available backend (CUDA > Vulkan > OpenGL > CPU)
    # You can force a specific backend with ti.init(arch=ti.cuda) or ti.init(arch=ti.cpu)
    ti.init(arch=ti.gpu)

    # In Taichi, the equivalent data structures for rigid bodies are built up in a similar fashion,
    # often starting by defining body properties like mass and inertia.
    # See: https://nvidia.github.io/warp/modules/sim.html
    # And then creating shapes associated with those bodies.
    # See: https://github.com/NVIDIA/warp/blob/main/warp/examples/sim/example_rigid_soft_contact.py

    # Benchmark with a range of body counts
    run_benchmark(num_bodies=100)
    run_benchmark(num_bodies=500)
    run_benchmark(num_bodies=1000)
    run_benchmark(num_bodies=4000)
    run_benchmark(num_bodies=10000)
