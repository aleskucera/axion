import time

import numpy as np
import warp as wp


# --- Helper Functions (Unchanged) ---
@wp.func
def _compute_contact_kinematics(
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal: wp.vec3,
    shape_a: wp.int32,
    shape_b: wp.int32,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    thickness: wp.array(dtype=float),
):

    if shape_a == shape_b:
        return False, wp.spatial_vector(), wp.spatial_vector(), -1, -1

    body_a, body_b = -1, -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    n, r_a, r_b = normal, wp.vec3(0.0), wp.vec3(0.0)
    p_world_a, p_world_b = point_a, point_b

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        offset_a = thickness[shape_a] * normal
        p_world_a = wp.transform_point(X_wb_a, point_a) - offset_a
        r_a = p_world_a - wp.transform_point(X_wb_a, body_com[body_a])
    if body_b >= 0:
        X_wb_b = body_q[body_b]
        offset_b = thickness[shape_b] * normal
        p_world_b = wp.transform_point(X_wb_b, point_b) + offset_b
        r_b = p_world_b - wp.transform_point(X_wb_b, body_com[body_b])

    gap = wp.dot(n, p_world_a - p_world_b)
    if gap >= 0.0:
        return False, wp.spatial_vector(), wp.spatial_vector(), body_a, body_b

    J_n_a = wp.spatial_vector(-wp.cross(r_a, n), -n)
    J_n_b = wp.spatial_vector(wp.cross(r_b, n), n)

    return True, J_n_a, J_n_b, body_a, body_b


# --- Fused CUDA Kernels ---


@wp.kernel
def body_dynamics_kernel(
    # Inputs
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    body_f_ext: wp.array(dtype=wp.spatial_vector),  # [B]
    gravity: wp.vec3,  # [3]
    body_mass: wp.array(dtype=wp.float32),  # [B]
    body_inertia: wp.array(dtype=wp.mat33),  # [B]
    dt: wp.float32,
    # Outputs (Writes to its OWN buffers)
    res_body: wp.array(dtype=wp.spatial_vector),  # [B]
    dres_dbody_qd: wp.array(dtype=wp.float32, ndim=2),  # [6B, 6B]
):
    """Computes the unconstrained dynamics residual and its derivative w.r.t. body_qd."""
    tid = wp.tid()
    if tid >= body_qd.shape[0]:
        return

    m, I, inv_dt = body_mass[tid], body_inertia[tid], 1.0 / dt

    res_ang = I * (
        wp.spatial_top(body_qd[tid]) - wp.spatial_top(body_qd_prev[tid])
    ) * inv_dt - wp.spatial_top(body_f_ext[tid])
    res_lin = (
        m
        * (wp.spatial_bottom(body_qd[tid]) - wp.spatial_bottom(body_qd_prev[tid]))
        * inv_dt
        - wp.spatial_bottom(body_f_ext[tid])
        - m * gravity
    )
    res_body[tid] = wp.spatial_vector(res_ang, res_lin)

    for i in wp.static(range(3)):
        dres_dbody_qd[tid * 6 + 3 + i, tid * 6 + 3 + i] = m * inv_dt
    for i in wp.static(range(3)):
        for j in wp.static(range(3)):
            dres_dbody_qd[tid * 6 + i, tid * 6 + j] = I[i, j] * inv_dt


@wp.kernel
def contact_contribution_kernel(
    # Inputs
    body_q: wp.array(dtype=wp.transform),  # [B]
    body_com: wp.array(dtype=wp.vec3),  # [B]
    shape_body: wp.array(dtype=wp.int32),  # [NumShapes]
    shape_thickness: wp.array(dtype=wp.float32),  # [NumShapes]
    contact_point0: wp.array(dtype=wp.vec3),  # [C]
    contact_point1: wp.array(dtype=wp.vec3),  # [C]
    contact_normal: wp.array(dtype=wp.vec3),  # [C]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    dt: wp.float32,
    # Outputs (Writes to its OWN buffers)
    res_contact: wp.array(dtype=wp.spatial_vector),  # [B]
    dres_dlambda_n: wp.array(dtype=wp.float32, ndim=2),  # [6B, C]
):
    """Computes contact impulse contributions and derivative w.r.t. lambda_n."""
    tid = wp.tid()
    if tid >= lambda_n.shape[0]:
        return

    inv_dt = 1.0 / dt
    is_active, J_n_a, J_n_b, body_a, body_b = _compute_contact_kinematics(
        contact_point0[tid],
        contact_point1[tid],
        contact_normal[tid],
        contact_shape0[tid],
        contact_shape1[tid],
        body_q,
        body_com,
        shape_body,
        shape_thickness,
    )

    if not is_active:
        return

    impulse_force = lambda_n[tid] * inv_dt
    if body_a >= 0:
        wp.atomic_add(res_contact, body_a, -J_n_a * impulse_force)
        for i in wp.static(range(6)):
            dres_dlambda_n[body_a * 6 + i, tid] = -J_n_a[i] * inv_dt
    if body_b >= 0:
        wp.atomic_add(res_contact, body_b, -J_n_b * impulse_force)
        for i in wp.static(range(6)):
            dres_dlambda_n[body_b * 6 + i, tid] = -J_n_b[i] * inv_dt


@wp.kernel
def sum_residuals_kernel(
    res_body: wp.array(dtype=wp.spatial_vector),  # [B]
    res_contact: wp.array(dtype=wp.spatial_vector),  # [B]
    res_final: wp.array(dtype=wp.spatial_vector),  # [B] (Output)
):
    """A simple kernel to sum the two residual contributions."""
    tid = wp.tid()
    if tid >= res_body.shape[0]:
        return

    res_final[tid] = res_body[tid] + res_contact[tid]


# --- Python Orchestrator and Benchmark ---


def setup_data(num_bodies, num_contacts, device):
    B, C, num_shapes = num_bodies, num_contacts, num_bodies
    shape0 = np.random.randint(0, num_shapes, size=C, dtype=np.int32)
    shape1 = np.random.randint(0, num_shapes, size=C, dtype=np.int32)
    mask = shape0 == shape1
    while np.any(mask):
        shape1[mask] = np.random.randint(
            0, num_shapes, size=np.sum(mask), dtype=np.int32
        )
        mask = shape0 == shape1
    rand_mat = np.random.rand(B, 3, 3)
    inertia_tensors = (rand_mat + rand_mat.transpose((0, 2, 1))) / 2.0

    data = {
        "body_q": wp.array(np.random.rand(B, 7), dtype=wp.transform, device=device),
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_f_ext": wp.array(
            np.zeros((B, 6)), dtype=wp.spatial_vector, device=device
        ),
        "body_com": wp.array(np.zeros((B, 3)), dtype=wp.vec3, device=device),
        "body_mass": wp.array(np.random.rand(B) + 1.0, dtype=wp.float32, device=device),
        "body_inertia": wp.array(inertia_tensors, dtype=wp.mat33, device=device),
        "shape_body": wp.array(np.arange(num_shapes, dtype=np.int32), device=device),
        "shape_thickness": wp.array(
            np.full(num_shapes, 0.01), dtype=wp.float32, device=device
        ),
        "contact_point0": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_point1": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_normal": wp.array(
            np.random.rand(C, 3) - 0.5, dtype=wp.vec3, device=device
        ),
        "contact_shape0": wp.array(shape0, dtype=wp.int32, device=device),
        "contact_shape1": wp.array(shape1, dtype=wp.int32, device=device),
        "lambda_n": wp.array(np.random.rand(C), dtype=wp.float32, device=device),
        "params": {"dt": 1.0 / 60.0, "gravity": wp.vec3(0.0, -9.8, 0.0)},
        # *** NEW: Separated residual buffers ***
        "res_body": wp.zeros(B, dtype=wp.spatial_vector, device=device),
        "res_contact": wp.zeros(B, dtype=wp.spatial_vector, device=device),
        "res_final": wp.zeros(B, dtype=wp.spatial_vector, device=device),
        "dres_dbody_qd": wp.zeros((6 * B, 6 * B), dtype=wp.float32, device=device),
        "dres_dlambda_n": wp.zeros((6 * B, C), dtype=wp.float32, device=device),
    }
    return data


def run_benchmark(num_bodies, num_contacts, num_iterations=200):
    print(
        f"\n--- Benchmarking DYNAMIC CONSTRAINT (Parallel Streams): B={num_bodies}, C={num_contacts}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, num_contacts, device)
    params = data["params"]

    # Create streams for concurrent execution
    stream1 = wp.Stream(device=device)
    stream2 = wp.Stream(device=device)

    # --- Prepare kernel arguments ---
    body_kernel_args = [
        data["body_qd"],
        data["body_qd_prev"],
        data["body_f_ext"],
        params["gravity"],
        data["body_mass"],
        data["body_inertia"],
        params["dt"],
        data["res_body"],
        data["dres_dbody_qd"],
    ]

    contact_kernel_args = [
        data["body_q"],
        data["body_com"],
        data["shape_body"],
        data["shape_thickness"],
        data["contact_point0"],
        data["contact_point1"],
        data["contact_normal"],
        data["contact_shape0"],
        data["contact_shape1"],
        data["lambda_n"],
        params["dt"],
        data["res_contact"],
        data["dres_dlambda_n"],
    ]

    sum_kernel_args = [data["res_body"], data["res_contact"], data["res_final"]]

    def clear_outputs():
        # Clear all output buffers
        data["res_body"].zero_()
        data["res_contact"].zero_()
        data["res_final"].zero_()
        data["dres_dbody_qd"].zero_()
        data["dres_dlambda_n"].zero_()

    # --- 1. Standard Launch Benchmark (with streams) ---
    print("1. Benching Standard Kernel Launch (with concurrent streams)...")
    clear_outputs()
    with wp.ScopedStream(stream1):
        wp.launch(body_dynamics_kernel, dim=num_bodies, inputs=body_kernel_args)
    with wp.ScopedStream(stream2):
        wp.launch(
            contact_contribution_kernel, dim=num_contacts, inputs=contact_kernel_args
        )
    wp.synchronize()  # Wait for both streams to finish
    wp.launch(sum_residuals_kernel, dim=num_bodies, inputs=sum_kernel_args)
    wp.synchronize()  # Warm-up

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        clear_outputs()
        with wp.ScopedStream(stream1):
            wp.launch(body_dynamics_kernel, dim=num_bodies, inputs=body_kernel_args)
        with wp.ScopedStream(stream2):
            wp.launch(
                contact_contribution_kernel,
                dim=num_contacts,
                inputs=contact_kernel_args,
            )
        wp.synchronize()  # Wait for independent kernels
        wp.launch(sum_residuals_kernel, dim=num_bodies, inputs=sum_kernel_args)
    wp.synchronize()  # Wait for the final sum
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch (with concurrent streams)...")
        with wp.ScopedCapture() as capture:
            # The entire sequence, including clears and stream usage, is captured.
            clear_outputs()
            with wp.ScopedStream(stream1):
                wp.launch(body_dynamics_kernel, dim=num_bodies, inputs=body_kernel_args)
            with wp.ScopedStream(stream2):
                wp.launch(
                    contact_contribution_kernel,
                    dim=num_contacts,
                    inputs=contact_kernel_args,
                )
            # Graph capture understands that the next launch must wait for the streams
            # it depends on (implicitly all, since there's no finer-grained dependency).
            # The most correct way is to have the summation on a stream that depends
            # on the others, but a simple synchronize works for graph capture.
            # Warp's graph capture will serialize these three launches, but the CUDA
            # runtime may still overlap stream1 and stream2 during execution.
            wp.launch(sum_residuals_kernel, dim=num_bodies, inputs=sum_kernel_args)

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


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print("Warning: No CUDA device found. Performance will be poor.")

    run_benchmark(num_bodies=100, num_contacts=100)
    run_benchmark(num_bodies=500, num_contacts=500)
    run_benchmark(num_bodies=1000, num_contacts=2000)
    run_benchmark(num_bodies=2000, num_contacts=4000)
