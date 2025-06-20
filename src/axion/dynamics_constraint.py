import time

import numpy as np
import warp as wp
from axion.contact_constraint import _compute_contact_kinematics
from warp.sim import ModelShapeGeometry


@wp.kernel
def unconstrained_dynamics_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    body_f: wp.array(dtype=wp.spatial_vector),  # [B]
    body_mass: wp.array(dtype=wp.float32),  # [B]
    body_inertia: wp.array(dtype=wp.mat33),  # [B]
    # --- Parameters ---
    dt: wp.float32,
    gravity: wp.vec3,  # [3]
    # Indices for output arrays:
    res_d_offset: wp.int32,
    dres_d_dbody_qd_offset: wp.vec2i,
    # --- Outputs ---
    neg_res: wp.array(dtype=wp.float32),  # [B]
    jacobian: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()
    if tid >= body_qd.shape[0]:
        return

    w = wp.spatial_top(body_qd[tid])
    v = wp.spatial_bottom(body_qd[tid])
    w_prev = wp.spatial_top(body_qd_prev[tid])
    v_prev = wp.spatial_bottom(body_qd_prev[tid])
    t = wp.spatial_top(body_f[tid])
    f = wp.spatial_bottom(body_f[tid])

    m, I = body_mass[tid], body_inertia[tid]

    res_ang = I * (w - w_prev) - t * dt
    res_lin = m * (v - v_prev) - f * dt - m * gravity * dt

    res_d = wp.spatial_vector(res_ang, res_lin)

    offset = res_d_offset + tid * 6
    for i in range(wp.static(6)):
        st_i = wp.static(i)
        neg_res[offset + st_i] = -res_d[st_i]

    # ∂res_d / ∂body_qd [6B, 6B]
    # Angular part:
    for i in range(wp.static(3)):
        for j in range(wp.static(3)):
            st_i, st_j = wp.static(i), wp.static(j)
            x_off = dres_d_dbody_qd_offset.x + tid * 6
            y_off = dres_d_dbody_qd_offset.y + tid * 6
            jacobian[x_off + st_i, y_off + st_j] = I[st_i, st_j]

    # Linear part:
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        x_off = dres_d_dbody_qd_offset.x + tid * 6 + 3
        y_off = dres_d_dbody_qd_offset.y + tid * 6 + 3
        jacobian[x_off + st_i, y_off + st_i] = m


@wp.kernel
def contact_contribution_kernel(
    body_q: wp.array(dtype=wp.transform),  # [B]
    body_com: wp.array(dtype=wp.vec3),  # [B]
    shape_body: wp.array(dtype=wp.int32),  # [NumShapes]
    shape_geo: ModelShapeGeometry,  # [NumShapes]
    contact_count: wp.array(dtype=wp.int32),  # [C]
    contact_point0: wp.array(dtype=wp.vec3),  # [C]
    contact_point1: wp.array(dtype=wp.vec3),  # [C]
    contact_normal: wp.array(dtype=wp.vec3),  # [C]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    # Indices for output arrays:
    res_d_offset: wp.int32,
    dres_d_dlambda_n_offset: wp.vec2i,
    # --- Outputs ---
    neg_res: wp.array(dtype=wp.float32),
    jacobian: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    is_active, _, J_n_a, J_n_b, body_a, body_b = _compute_contact_kinematics(
        contact_point0[tid],
        contact_point1[tid],
        contact_normal[tid],
        contact_shape0[tid],
        contact_shape1[tid],
        body_q,
        body_com,
        shape_body,
        shape_geo,
    )

    if not is_active:
        return

    if body_a >= 0:
        # Accumulate the residual for body_a
        res_offset = res_d_offset + body_a * 6
        res_body_a = -J_n_a * lambda_n[tid]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(neg_res, res_offset + st_i, -res_body_a[st_i])

        # Update the Jacobian derivative for body_a
        x_offset = dres_d_dlambda_n_offset.x + body_a * 6
        y_offset = dres_d_dlambda_n_offset.y + tid
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            jacobian[x_offset + st_i, y_offset] = -J_n_a[st_i]

    if body_b >= 0:
        # Accumulate the residual for body_b
        res_offset = res_d_offset + body_b * 6
        res_body_b = -J_n_b * lambda_n[tid]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(neg_res, res_offset + st_i, -res_body_b[st_i])

        # Update the Jacobian derivative for body_b
        x_offset = dres_d_dlambda_n_offset.x + body_b * 6
        y_offset = dres_d_dlambda_n_offset.y + tid
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            jacobian[x_offset + st_i, y_offset] = -J_n_b[st_i]


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
        "shape_geo": ModelShapeGeometry(),
        "contact_point0": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_point1": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_normal": wp.array(
            np.random.rand(C, 3) - 0.5, dtype=wp.vec3, device=device
        ),
        "contact_shape0": wp.array(shape0, dtype=wp.int32, device=device),
        "contact_shape1": wp.array(shape1, dtype=wp.int32, device=device),
        "lambda_n": wp.array(np.random.rand(C), dtype=wp.float32, device=device),
        "params": {
            "dt": 1.0 / 60.0,
            "gravity": wp.vec3(0.0, -9.8, 0.0),
            "res_d_offset": 0,
            "dres_d_dbody_qd_offset": wp.vec2i(0, 0),
            "dres_d_dlambda_n_offset": wp.vec2i(6 * B, 6 * B),
        },
        "neg_res": wp.zeros(B * 6, dtype=wp.float32, device=device),
        "jacobian": wp.zeros((12 * B, 6 * B + C), dtype=wp.float32, device=device),
    }

    data["shape_geo"].thickness = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    return data


def run_benchmark(num_bodies, num_contacts, num_iterations=200):
    print(
        f"\n--- Benchmarking DYNAMIC CONSTRAINT: B={num_bodies}, C={num_contacts}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, num_contacts, device)
    params = data["params"]

    # --- Prepare kernel arguments ---
    unconstrained_dynamics_kernel_inputs = [
        data["body_qd"],
        data["body_qd_prev"],
        data["body_f_ext"],
        data["body_mass"],
        data["body_inertia"],
        params["dt"],
        params["gravity"],
        params["res_d_offset"],
        params["dres_d_dbody_qd_offset"],
    ]
    unconstrained_dynamics_kernel_outputs = [
        data["neg_res"],
        data["jacobian"],
    ]

    contact_kernel_inputs = [
        data["body_q"],
        data["body_com"],
        data["shape_body"],
        data["shape_geo"],
        data["contact_point0"],
        data["contact_point1"],
        data["contact_normal"],
        data["contact_shape0"],
        data["contact_shape1"],
        data["lambda_n"],
        params["dt"],
        params["res_d_offset"],
        params["dres_d_dlambda_n_offset"],
    ]

    contact_kernel_outputs = [
        data["neg_res"],
        data["jacobian"],
    ]

    # --- 1. Standard Launch Benchmark (with streams) ---
    print("1. Benching Standard Kernel Launch (with concurrent streams)...")
    wp.launch(
        unconstrained_dynamics_kernel,
        dim=num_bodies,
        inputs=unconstrained_dynamics_kernel_inputs,
        outputs=unconstrained_dynamics_kernel_outputs,
    )
    wp.launch(
        contact_contribution_kernel,
        dim=num_contacts,
        inputs=contact_kernel_inputs,
        outputs=contact_kernel_outputs,
    )
    wp.synchronize()  # Warm-up

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=unconstrained_dynamics_kernel_inputs,
            outputs=unconstrained_dynamics_kernel_outputs,
        )
        wp.launch(
            contact_contribution_kernel,
            dim=num_contacts,
            inputs=contact_kernel_inputs,
            outputs=contact_kernel_outputs,
        )

    wp.synchronize()  # Wait for the final sum
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch (with concurrent streams)...")
        with wp.ScopedCapture() as capture:
            wp.launch(
                unconstrained_dynamics_kernel,
                dim=num_bodies,
                inputs=unconstrained_dynamics_kernel_inputs,
                outputs=unconstrained_dynamics_kernel_outputs,
            )
            wp.launch(
                contact_contribution_kernel,
                dim=num_contacts,
                inputs=contact_kernel_inputs,
                outputs=contact_kernel_outputs,
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
