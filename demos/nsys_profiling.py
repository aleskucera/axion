import warp as wp


@wp.kernel
def inc_loop(a: wp.array(dtype=float), num_iters: int):
    i = wp.tid()
    for j in range(num_iters):
        a[i] += 1.0


n = 10
devices = wp.get_cuda_devices()

# pre-allocate host arrays for readback
host_arrays = [wp.empty(n, dtype=float, device="cpu", pinned=True) for _ in devices]

# code for profiling
with wp.ScopedTimer("Demo"):
    for i, device in enumerate(devices):
        a = wp.zeros(n, dtype=float, device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 5], device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 2], device=device)
        wp.launch(inc_loop, dim=n, inputs=[a, 2], device=device)
        wp.copy(host_arrays[i], a)
