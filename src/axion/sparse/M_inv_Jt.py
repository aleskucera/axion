import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

wp.init()


# -----------------------------
# Kernel Definitions
# -----------------------------
@wp.kernel
def kernel_M_inv_Jt_matvec_scatter(
    dlambda: wp.array(dtype=wp.float32),
    J: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    du: wp.array(dtype=wp.spatial_vector),
):
    c_idx = wp.tid()
    b_idx = constraint_to_body[c_idx]
    delta_impulse = J[c_idx] * dlambda[c_idx]
    m_inv = M_inv[b_idx]
    wp.atomic_add(du, b_idx, m_inv * delta_impulse)


def create_M_inv_Jt_matvec_tiled(
    num_bodies: int,
    constraints_per_body: int,
    bodies_in_tile: int,
):
    assert num_bodies % bodies_in_tile == 0
    tile_size = bodies_in_tile * constraints_per_body

    @wp.kernel
    def kernel_M_inv_Jt_matvec_tiled(
        lambda_: wp.array(dtype=wp.float32),
        J: wp.array(dtype=wp.spatial_vector),
        M_inv: wp.array(dtype=wp.spatial_matrix),
        u: wp.array(dtype=wp.spatial_vector),
    ):
        tile_idx = wp.tid()
        constraint_start = tile_idx * tile_size

        jacobian_tile = wp.tile_load(J, tile_size, offset=constraint_start)
        lambda_tile = wp.tile_load(lambda_, tile_size, offset=constraint_start)

        impulse_tile = wp.tile_map(wp.mul, jacobian_tile, lambda_tile)
        impulse_by_body = wp.tile_reshape(
            impulse_tile, shape=(bodies_in_tile, constraints_per_body)
        )
        total_impulse_per_body = wp.tile_reduce(wp.add, impulse_by_body, axis=1)

        body_offset = tile_idx * bodies_in_tile
        m_inv_tile = wp.tile_load(M_inv, bodies_in_tile, offset=body_offset)
        u_tile = wp.tile_map(wp.mul, m_inv_tile, total_impulse_per_body)
        wp.tile_store(u, u_tile, offset=tile_idx * bodies_in_tile)

    return kernel_M_inv_Jt_matvec_tiled


# -----------------------------
# Data Builder
# -----------------------------
def build_data(num_bodies: int, constraints_per_body: int):
    num_constraints = num_bodies * constraints_per_body

    dlambda_np = np.random.randn(num_constraints).astype(np.float32)
    J_np = np.random.randn(num_constraints, 6).astype(np.float32)
    M_inv_np = np.random.randn(num_bodies, 6, 6).astype(np.float32)
    constraint_to_body_np = np.repeat(np.arange(num_bodies, dtype=np.int32), constraints_per_body)

    dlambda = wp.array(dlambda_np, dtype=wp.float32)
    J = wp.array(J_np, dtype=wp.spatial_vector)
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    constraint_to_body = wp.array(constraint_to_body_np, dtype=wp.int32)
    out = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    return dlambda, J, constraint_to_body, M_inv, out


# -----------------------------
# Timing Functions
# -----------------------------
def measure_scatter_time(
    num_bodies: int,
    constraints_per_body: int,
    num_repeats: int = 50,
    use_cuda_graph: bool = False,
    num_graph_iters: int = 100,
    num_ops_in_graph: int = 10,
):
    num_constraints = num_bodies * constraints_per_body
    dlambda, J, constraint_to_body, M_inv, out = build_data(num_bodies, constraints_per_body)

    # Warmup
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=num_constraints,
        inputs=[dlambda, J, constraint_to_body, M_inv],
        outputs=[out],
    )
    wp.synchronize()

    if use_cuda_graph:
        with wp.ScopedCapture() as cap:
            for _ in range(num_ops_in_graph):
                out.zero_()
                wp.launch(
                    kernel_M_inv_Jt_matvec_scatter,
                    dim=num_constraints,
                    inputs=[dlambda, J, constraint_to_body, M_inv],
                    outputs=[out],
                )
        wp.synchronize()

        t0 = time.time()
        for _ in range(num_graph_iters):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        t1 = time.time()
        return (t1 - t0) / (num_graph_iters * num_ops_in_graph)
    else:
        t0 = time.time()
        for _ in range(num_repeats):
            out.zero_()
            wp.launch(
                kernel_M_inv_Jt_matvec_scatter,
                dim=num_constraints,
                inputs=[dlambda, J, constraint_to_body, M_inv],
                outputs=[out],
            )
        wp.synchronize()
        t1 = time.time()
        return (t1 - t0) / num_repeats


def measure_tiled_time(
    num_bodies: int,
    constraints_per_body: int,
    bodies_in_tile: int,
    num_repeats: int = 50,
    use_cuda_graph: bool = False,
    num_graph_iters: int = 100,
    num_ops_in_graph: int = 10,
):
    tile_size = bodies_in_tile * constraints_per_body
    num_constraints = num_bodies * constraints_per_body
    num_tiles = num_constraints // tile_size

    dlambda, J, constraint_to_body, M_inv, out = build_data(num_bodies, constraints_per_body)

    kernel_tiled = create_M_inv_Jt_matvec_tiled(
        num_bodies=num_bodies,
        constraints_per_body=constraints_per_body,
        bodies_in_tile=bodies_in_tile,
    )

    # Warmup
    wp.launch_tiled(
        kernel=kernel_tiled,
        dim=num_tiles,
        inputs=[dlambda, J, M_inv],
        outputs=[out],
        block_dim=tile_size,
    )
    wp.synchronize()

    if use_cuda_graph:
        with wp.ScopedCapture() as cap:
            for _ in range(num_ops_in_graph):
                wp.launch_tiled(
                    kernel=kernel_tiled,
                    dim=num_tiles,
                    inputs=[dlambda, J, M_inv],
                    outputs=[out],
                    block_dim=tile_size,
                )
        wp.synchronize()

        t0 = time.time()
        for _ in range(num_graph_iters):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        t1 = time.time()
        return (t1 - t0) / (num_graph_iters * num_ops_in_graph)
    else:
        t0 = time.time()
        for _ in range(num_repeats):
            wp.launch_tiled(
                kernel=kernel_tiled,
                dim=num_tiles,
                inputs=[dlambda, J, M_inv],
                outputs=[out],
                block_dim=tile_size,
            )
        wp.synchronize()
        t1 = time.time()
        return (t1 - t0) / num_repeats


# -----------------------------
# Benchmark Sweep
# -----------------------------
def run_benchmark_sweep(
    body_counts: list[int],
    constraints_per_body: int = 16,
    bodies_in_tile: int = 8,
    num_repeats: int = 50,
    use_cuda_graph: bool = False,
):
    scatter_times = []
    tiled_times = []

    for num_bodies in body_counts:
        print(f"Benchmarking num_bodies={num_bodies}...")

        t_scatter = measure_scatter_time(
            num_bodies=num_bodies,
            constraints_per_body=constraints_per_body,
            num_repeats=num_repeats,
            use_cuda_graph=use_cuda_graph,
        )
        scatter_times.append(t_scatter * 1e3)  # Convert to ms

        t_tiled = measure_tiled_time(
            num_bodies=num_bodies,
            constraints_per_body=constraints_per_body,
            bodies_in_tile=bodies_in_tile,
            num_repeats=num_repeats,
            use_cuda_graph=use_cuda_graph,
        )
        tiled_times.append(t_tiled * 1e3)  # Convert to ms

        print(f"  Scatter: {scatter_times[-1]:.4f} ms, Tiled: {tiled_times[-1]:.4f} ms")

    return scatter_times, tiled_times


def plot_results(
    body_counts: list[int],
    scatter_times: list[float],
    tiled_times: list[float],
    scatter_times_graph: list[float] | None = None,
    tiled_times_graph: list[float] | None = None,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute times
    ax1 = axes[0]
    ax1.plot(
        body_counts, scatter_times, "o-", label="Scatter (standard)", linewidth=2, markersize=6
    )
    ax1.plot(body_counts, tiled_times, "s-", label="Tiled (standard)", linewidth=2, markersize=6)

    if scatter_times_graph is not None:
        ax1.plot(
            body_counts,
            scatter_times_graph,
            "^--",
            label="Scatter (CUDA graph)",
            linewidth=2,
            markersize=6,
        )
    if tiled_times_graph is not None:
        ax1.plot(
            body_counts,
            tiled_times_graph,
            "d--",
            label="Tiled (CUDA graph)",
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Number of Bodies", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Kernel Execution Time vs Number of Bodies", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Plot 2: Speedup (scatter / tiled)
    ax2 = axes[1]
    speedup_std = [s / t if t > 0 else 0 for s, t in zip(scatter_times, tiled_times)]
    ax2.plot(
        body_counts, speedup_std, "o-", label="Standard", linewidth=2, markersize=6, color="green"
    )

    if scatter_times_graph is not None and tiled_times_graph is not None:
        speedup_graph = [
            s / t if t > 0 else 0 for s, t in zip(scatter_times_graph, tiled_times_graph)
        ]
        ax2.plot(
            body_counts,
            speedup_graph,
            "s--",
            label="CUDA graph",
            linewidth=2,
            markersize=6,
            color="purple",
        )

    ax2.axhline(y=1.0, color="r", linestyle=":", linewidth=1.5, label="Break-even")
    ax2.set_xlabel("Number of Bodies", fontsize=12)
    ax2.set_ylabel("Speedup (Scatter / Tiled)", fontsize=12)
    ax2.set_title("Tiled Kernel Speedup vs Scatter", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    # Configuration
    constraints_per_body = 32
    bodies_in_tile = 4

    # Body counts to benchmark (must be divisible by bodies_in_tile)
    body_counts = [8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384]

    # Filter to ensure divisibility
    body_counts = [n for n in body_counts if n % bodies_in_tile == 0]

    print("=" * 60)
    print("Benchmark: M_inv * J^T * x (Scatter vs Tiled)")
    print("=" * 60)
    print(f"constraints_per_body = {constraints_per_body}")
    print(f"bodies_in_tile = {bodies_in_tile}")
    print(f"body_counts = {body_counts}")
    print("=" * 60)

    # Run standard benchmarks
    print("\n--- Standard Launches ---")
    scatter_times, tiled_times = run_benchmark_sweep(
        body_counts=body_counts,
        constraints_per_body=constraints_per_body,
        bodies_in_tile=bodies_in_tile,
        use_cuda_graph=False,
    )

    # Run CUDA graph benchmarks
    print("\n--- CUDA Graph Launches ---")
    scatter_times_graph, tiled_times_graph = run_benchmark_sweep(
        body_counts=body_counts,
        constraints_per_body=constraints_per_body,
        bodies_in_tile=bodies_in_tile,
        use_cuda_graph=True,
    )

    # Plot results
    plot_results(
        body_counts=body_counts,
        scatter_times=scatter_times,
        tiled_times=tiled_times,
        scatter_times_graph=scatter_times_graph,
        tiled_times_graph=tiled_times_graph,
        save_path="kernel_benchmark.png",
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(
        f"{'Bodies':>8} | {'Scatter (ms)':>12} | {'Tiled (ms)':>12} | {'Speedup':>8} | {'Graph Speedup':>13}"
    )
    print("-" * 80)
    for i, n in enumerate(body_counts):
        speedup_std = scatter_times[i] / tiled_times[i] if tiled_times[i] > 0 else 0
        speedup_graph = (
            scatter_times_graph[i] / tiled_times_graph[i] if tiled_times_graph[i] > 0 else 0
        )
        print(
            f"{n:>8} | {scatter_times[i]:>12.4f} | {tiled_times[i]:>12.4f} | {speedup_std:>8.2f}x | {speedup_graph:>12.2f}x"
        )


if __name__ == "__main__":
    main()
