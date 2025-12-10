import numpy as np
import warp as wp


@wp.kernel
def kernel_repeat(
    a: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    assert len(out) >= len(a), "Array to repeat must be smaller or equal"
    assert len(out) % len(a), "Something"

    n = len(out) // len(a)

    out[tid] = a[tid // n]


if __name__ == "__main__":
    N = 4
    num_repeats = 4
    arr_np = np.arange(N)
    arr = wp.from_numpy(arr_np, dtype=wp.float32)

    out = wp.zeros(N * num_repeats, dtype=wp.float32)

    wp.launch(kernel=kernel_repeat, dim=N * num_repeats, inputs=[arr], outputs=[out])

    print(out)
