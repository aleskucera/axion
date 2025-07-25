import time

import numpy as np
import warp as wp

# Initialize Warp
wp.init()
device = "cuda"


@wp.kernel
def branching_kernel(
    input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    x = input[tid]
    if x > 0.0:
        result = float(0.0)
        for i in range(1000):  # Increased iterations for more work
            temp = x + float(i)  # Vary the input each iteration
            result += temp * temp  # Square the adjusted value
        output[tid] = result
    else:
        result = float(0.0)
        output[tid] = 0.0
        for i in range(1000):  # Increased iterations for more work
            temp = x + float(i)  # Vary the input each iteration
            result += temp * temp  # Square the adjusted value
        output[tid] = result


@wp.kernel
def branchless_kernel(
    input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    x = input[tid]
    result = float(0.0)
    for i in range(1000):
        temp = x + float(i)
        result += temp * temp
    output[tid] = wp.where(x > 0.0, result, 0.0)


# Benchmark function
def run_benchmark(kernel, input_data, num_iterations=100):
    input_wp = wp.array(input_data, dtype=wp.float32, device=device)
    output_wp = wp.zeros_like(input_wp)
    wp.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            kernel, dim=len(input_data), inputs=[input_wp, output_wp], device=device
        )
    wp.synchronize()
    elapsed_time = (
        (time.perf_counter() - start_time) / num_iterations * 1000
    )  # Convert to ms
    return elapsed_time


# Set array size
N = 1000000

# Generate input patterns
all_positive = np.random.rand(N).astype(np.float32)  # All values > 0
all_negative = -np.random.rand(N).astype(np.float32)  # All values < 0
alternating = np.array(
    [0.5 if i % 2 == 0 else -0.5 for i in range(N)], dtype=np.float32
)  # Alternates +/-
random_mix = (np.random.rand(N) * 2 - 1).astype(np.float32)  # Random between -1 and 1

# List of test cases
test_cases = [
    ("All Positive", all_positive),
    ("All Negative", all_negative),
    ("Alternating", alternating),
    ("Random Mix", random_mix),
]

# Run benchmarks and print results
print("=== Benchmarking Branching vs. Branchless Kernels ===")
print(f"Array size: {N} elements, Device: {device}, Iterations: 100\n")

print("**Branching Kernel Results:**")
for name, data in test_cases:
    time_ms = run_benchmark(branching_kernel, data)
    print(f"  {name}: {time_ms:.3f} ms")

print("\n**Branchless Kernel Results:**")
for name, data in test_cases:
    time_ms = run_benchmark(branchless_kernel, data)
    print(f"  {name}: {time_ms:.3f} ms")
print(f"Cache saved to the file {wp.config.kernel_cache_dir}")
