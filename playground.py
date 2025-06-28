import time

import numpy as np
import warp as wp


@wp.kernel
def compute_norm_kernel(
    bodies: wp.array(dtype=wp.vec3),  # [B]
    points: wp.array(dtype=wp.vec3),  # [N]
    distances: wp.array(dtype=wp.float32, ndim=2),  # [B, N]
):
    """
    Computes the Euclidean distance between each body and each point.
    Launched with a 2D grid of threads (B x N).
    """
    b, n = wp.tid()
    if b >= bodies.shape[0] or n >= points.shape[0]:
        return

    diff = bodies[b] - points[n]
    norm = wp.length(diff)
    distances[b, n] = norm


def setup_data(num_bodies, num_points, device):
    """Generates all necessary input and output arrays for the kernel."""
    B = num_bodies
    N = num_points

    # Generate random data on the CPU
    bodies_np = np.random.rand(B, 3).astype(np.float32)
    points_np = np.random.rand(N, 3).astype(np.float32)

    # Transfer data to the specified Warp device
    data = {
        # --- Inputs ---
        "bodies": wp.array(bodies_np, dtype=wp.vec3, device=device),
        "points": wp.array(points_np, dtype=wp.vec3, device=device),
        # --- Outputs ---
        "distances": wp.zeros((B, N), dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(num_bodies, num_points, num_iterations=200):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Norm Kernel: B={num_bodies}, N={num_points}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, num_points, device)

    # Assemble the list of arguments in the exact order the kernel expects.
    kernel_args = [
        data["bodies"],
        data["points"],
        data["distances"],
    ]

    # The kernel is 2D, so the launch dimension must match.
    launch_dim = (num_bodies, num_points)

    # --- 1. Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    # Warm-up launch to compile the kernel, etc.
    wp.launch(
        kernel=compute_norm_kernel,
        dim=launch_dim,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            kernel=compute_norm_kernel,
            dim=launch_dim,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark (only on GPU) ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")

        # Capture the kernel launch into a graph
        with wp.ScopedCapture() as capture:
            wp.launch(
                kernel=compute_norm_kernel,
                dim=launch_dim,
                inputs=kernel_args,
                device=device,
            )
        graph = capture.graph

        # Warm-up launch using the captured graph
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
        print("2. Skipping CUDA Graph benchmark (not on a GPU device).")

    print("-" * 65)


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print("Warning: No CUDA device found. Performance will be poor.")

    # Run benchmarks with different sizes for bodies and points
    run_benchmark(num_bodies=100, num_points=1000)
    run_benchmark(num_bodies=500, num_points=1000)
    run_benchmark(num_bodies=1000, num_points=1000)
    run_benchmark(num_bodies=1000, num_points=5000)
    run_benchmark(num_bodies=2000, num_points=2000)
