import time
from typing import Tuple

import numpy as np
import torch
import warp as wp
import warp.context as wpc
from axion.ncp import scaled_fisher_burmeister_derivatives
from axion.ncp import scaled_fisher_burmeister_evaluate
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.kernel
def contact_info_kernel(
    body_q: wp.array(dtype=wp.transform),  # [B]
    body_com: wp.array(dtype=wp.vec3),  # [B]
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,  # [B]
    contact_count: wp.array(dtype=int),  # [1]
    contact_point0: wp.array(dtype=wp.vec3),  # [C]
    contact_point1: wp.array(dtype=wp.vec3),  # [C]
    contact_normal: wp.array(dtype=wp.vec3),  # [C]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    # Outputs:
    gap_function: wp.array(dtype=wp.float32),  # [C]
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
):
    # Get the contact index
    tid = wp.tid()

    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    thickness_a = 0.0
    thickness_b = 0.0
    if shape_a >= 0:
        body_a = shape_body[shape_a]
        thickness_a = geo.thickness[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
        thickness_b = geo.thickness[shape_b]

    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        bx_a = wp.transform_point(X_wb_a, bx_a) + thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)
    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    d = wp.dot(n, bx_a - bx_b)

    if d >= 0.0:
        # No penetration, no contact
        gap_function[tid] = 0.0
        return

    gap_function[tid] = d

    # Compute r × n for the rotational component of the Jacobian
    r_cross_n_a = wp.cross(r_a, n)
    r_cross_n_b = wp.cross(r_b, n)

    J_n[tid, 0] = wp.spatial_vector(-r_cross_n_a, -n)
    J_n[tid, 1] = wp.spatial_vector(r_cross_n_b, n)


@wp.kernel
def contact_residual_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    shape_body: wp.array(dtype=int),  # [B]
    shape_materials: ModelShapeMaterials,  # [B]
    contact_count: wp.array(dtype=int),  # [1]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    gap_function: wp.array(dtype=wp.float32),  # [C]
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    # Outputs:
    res: wp.array(dtype=wp.float32),  # [C]
):
    """Compute the contact residuals for each body and contact point.
    The residuals are computed using the scaled Fisher-Burmeister function.
    The residuals are defined as:
    res = phi(lambda_n, v_n + b_err + b_rest)

    Where:
    - lambda_n is the normal impulse
    - v_n is the normal velocity (J_n * body_vel)
    - b_err is the Baumgarte stabilization bias based on penetration depth
    - b_rest is the restitution term based on previous velocity
    """
    # Get the contact index
    tid = wp.tid()

    if tid >= contact_count[0]:
        res[tid] = -lambda_n[tid]
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    J_n_a = J_n[tid, 0]  # Jacobian for body A
    J_n_b = J_n[tid, 1]  # Jacobian for body B

    # Get the contact velocities
    v_n_a = wp.dot(J_n_a, body_qd[body_a])
    v_n_b = wp.dot(J_n_b, body_qd[body_b])
    delta_v_n = v_n_a - v_n_b

    # Get the contact velocities at the previous time step
    v_n_a_prev = wp.dot(J_n_a, body_qd_prev[body_a])
    v_n_b_prev = wp.dot(J_n_b, body_qd_prev[body_b])
    delta_v_n_prev = v_n_a_prev - v_n_b_prev

    # Compute the contact restitution
    e = 0.0
    num_bodies = 0
    if shape_a >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_a]
    if shape_b >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_b]
    if num_bodies > 0:
        e /= float(num_bodies)

    b_err = -(stabilization_factor / dt) * gap_function[tid]
    b_rest = e * delta_v_n_prev

    res[tid] = scaled_fisher_burmeister_evaluate(
        lambda_n[tid], delta_v_n + b_err + b_rest, fb_alpha, fb_beta, fb_epsilon
    )


@wp.kernel
def contact_residual_derivative_wrt_lambda_n_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    shape_body: wp.array(dtype=int),  # [B]
    shape_materials: ModelShapeMaterials,  # [B]
    contact_count: wp.array(dtype=int),  # [1]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    gap_function: wp.array(dtype=wp.float32),  # [C]
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    # Outputs:
    dres_n_dlambda_n: wp.array(dtype=wp.float32, ndim=2),  # [C, C]
):
    # Get the contact index
    tid = wp.tid()

    if tid >= contact_count[0]:
        dres_n_dlambda_n[tid, tid] = -1.0
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    J_n_a = J_n[tid, 0]  # Jacobian for body A
    J_n_b = J_n[tid, 1]  # Jacobian for body B

    # Get the contact velocities
    v_n_a = wp.dot(J_n_a, body_qd[body_a])
    v_n_b = wp.dot(J_n_b, body_qd[body_b])
    delta_v_n = v_n_a - v_n_b

    # Get the contact velocities at the previous time step
    v_n_a_prev = wp.dot(J_n_a, body_qd_prev[body_a])
    v_n_b_prev = wp.dot(J_n_b, body_qd_prev[body_b])
    delta_v_n_prev = v_n_a_prev - v_n_b_prev

    # Compute the contact restitution
    e = 0.0
    num_bodies = 0
    if shape_a >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_a]
    if shape_b >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_b]
    if num_bodies > 0:
        e /= float(num_bodies)

    b_err = -(stabilization_factor / dt) * gap_function[tid]
    b_rest = e * delta_v_n_prev

    da, _ = scaled_fisher_burmeister_derivatives(
        lambda_n[tid], delta_v_n + b_err + b_rest, fb_alpha, fb_beta, fb_epsilon
    )

    dres_n_dlambda_n[tid, tid] = da


@wp.kernel
def contact_residual_derivative_wrt_body_qd_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    shape_body: wp.array(dtype=int),  # [B]
    shape_materials: ModelShapeMaterials,  # [B]
    contact_count: wp.array(dtype=int),  # [1]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    gap_function: wp.array(dtype=wp.float32),  # [C]
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    # Outputs:
    dres_n_dbody_qd: wp.array(dtype=wp.float32, ndim=2),  # [C, 6B]
):
    # Get the contact index
    tid = wp.tid()
    # dres_n_dbody_qd[:, :, :] = 0.0  # Initialize the Jacobian to zero

    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    J_n_a = J_n[tid, 0]  # Jacobian for body A
    J_n_b = J_n[tid, 1]  # Jacobian for body B

    # Get the contact velocities
    v_n_a = wp.dot(J_n_a, body_qd[body_a])
    v_n_b = wp.dot(J_n_b, body_qd[body_b])
    delta_v_n = v_n_a - v_n_b

    # Get the contact velocities at the previous time step
    v_n_a_prev = wp.dot(J_n_a, body_qd_prev[body_a])
    v_n_b_prev = wp.dot(J_n_b, body_qd_prev[body_b])
    delta_v_n_prev = v_n_a_prev - v_n_b_prev

    # Compute the contact restitution
    e = 0.0
    num_bodies = 0
    if shape_a >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_a]
    if shape_b >= 0:
        num_bodies += 1
        e += shape_materials.restitution[shape_b]
    if num_bodies > 0:
        e /= float(num_bodies)

    b_err = -(stabilization_factor / dt) * gap_function[tid]
    b_rest = e * delta_v_n_prev

    _, db = scaled_fisher_burmeister_derivatives(
        lambda_n[tid], delta_v_n + b_err + b_rest, fb_alpha, fb_beta, fb_epsilon
    )

    # Use static unrolling of the for cycle to compute the derivatives
    for i in range(wp.static(6)):
        # dres_n / dbody_qd[body_a]
        dres_n_dbody_qd[tid, body_a * 6 + i] = db * J_n_a[i]
        # dres_n / dbody_qd[body_b]
        dres_n_dbody_qd[tid, body_b * 6 + i] = -db * J_n_b[i]


def setup_benchmark_data(num_bodies, num_contacts, device):
    """Generates large-scale random data for benchmarking."""
    B, C = num_bodies, num_contacts

    # Generate random contact pairs, ensuring they are not self-contacts
    shape0 = np.random.randint(0, B, size=C, dtype=np.int32)
    shape1 = np.random.randint(0, B, size=C, dtype=np.int32)
    mask = shape0 == shape1
    while np.any(mask):
        shape1[mask] = np.random.randint(0, B, size=np.sum(mask), dtype=np.int32)
        mask = shape0 == shape1

    data = {
        "body_q": wp.array(np.random.rand(B, 7), dtype=wp.transform, device=device),
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_com": wp.array(np.random.rand(B, 3), dtype=wp.vec3, device=device),
        "shape_body": wp.array(np.arange(B, dtype=np.int32), dtype=int, device=device),
        "contact_count": wp.array([C], dtype=int, device=device),
        "contact_point0": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_point1": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_normal": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_shape0": wp.array(shape0, dtype=wp.int32, device=device),
        "contact_shape1": wp.array(shape1, dtype=wp.int32, device=device),
        "lambda_n": wp.array(np.random.rand(C), dtype=wp.float32, device=device),
        "geo": ModelShapeGeometry(),
        "shape_materials": ModelShapeMaterials(),
        # Simulation constants
        "dt": 0.01,
        "stabilization_factor": 0.2,
        "fb_alpha": 0.25,
        "fb_beta": 0.25,
        "fb_epsilon": 1e-6,
        # Output arrays (pre-allocated)
        "gap_function": wp.zeros(C, dtype=wp.float32, device=device),
        "J_n": wp.zeros((C, 2), dtype=wp.spatial_vector, device=device),
        "res": wp.zeros(C, dtype=wp.float32, device=device),
        "dres_n_dlambda_n": wp.zeros((C, C), dtype=wp.float32, device=device),
        "dres_n_dbody_qd": wp.zeros((C, 6 * B), dtype=wp.float32, device=device),
    }

    data["geo"].thickness = wp.array(np.random.rand(B), dtype=wp.float32, device=device)
    data["shape_materials"].restitution = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    return data


def run_performance_benchmark(num_bodies, num_contacts, num_iterations=100):
    """
    Measures kernel performance using standard launches vs. a captured CUDA graph.
    """
    print(
        f"\n--- Running Benchmark: {num_bodies} Bodies, {num_contacts} Contacts, {num_iterations} Iterations ---"
    )
    device = wp.get_device()
    data = setup_benchmark_data(num_bodies, num_contacts, device)
    C = num_contacts

    # Unpack all data for convenience
    (
        body_q,
        body_qd,
        body_qd_prev,
        body_com,
        shape_body,
        geo,
        shape_materials,
        contact_count,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_shape0,
        contact_shape1,
        lambda_n,
        dt,
        sf,
        fba,
        fbb,
        fbe,
        gap_function,
        J_n,
        res,
        dres_n_dlambda_n,
        dres_n_dbody_qd,
    ) = (
        data["body_q"],
        data["body_qd"],
        data["body_qd_prev"],
        data["body_com"],
        data["shape_body"],
        data["geo"],
        data["shape_materials"],
        data["contact_count"],
        data["contact_point0"],
        data["contact_point1"],
        data["contact_normal"],
        data["contact_shape0"],
        data["contact_shape1"],
        data["lambda_n"],
        data["dt"],
        data["stabilization_factor"],
        data["fb_alpha"],
        data["fb_beta"],
        data["fb_epsilon"],
        data["gap_function"],
        data["J_n"],
        data["res"],
        data["dres_n_dlambda_n"],
        data["dres_n_dbody_qd"],
    )

    # --- 1. Benchmark: Standard Kernel Launching ---
    wp.synchronize()  # Ensure device is ready

    # Warm-up run to compile kernels
    wp.launch(
        contact_info_kernel,
        dim=C,
        inputs=[
            body_q,
            body_com,
            shape_body,
            geo,
            contact_count,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
        ],
        outputs=[gap_function, J_n],
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            contact_info_kernel,
            dim=C,
            inputs=[
                body_q,
                body_com,
                shape_body,
                geo,
                contact_count,
                contact_point0,
                contact_point1,
                contact_normal,
                contact_shape0,
                contact_shape1,
            ],
            outputs=[gap_function, J_n],
        )
        wp.launch(
            contact_residual_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[res],
        )
        wp.launch(
            contact_residual_derivative_wrt_lambda_n_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[dres_n_dlambda_n],
        )
        wp.launch(
            contact_residual_derivative_wrt_body_qd_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[dres_n_dbody_qd],
        )
    wp.synchronize()
    standard_time = (time.perf_counter() - start_time) / num_iterations

    print(f"Standard Sequential Launch..: {standard_time * 1000:.4f} ms per iteration")

    # --- 2. Benchmark: CUDA Graph ---
    with wp.ScopedCapture() as capture:
        wp.launch(
            contact_info_kernel,
            dim=C,
            inputs=[
                body_q,
                body_com,
                shape_body,
                geo,
                contact_count,
                contact_point0,
                contact_point1,
                contact_normal,
                contact_shape0,
                contact_shape1,
            ],
            outputs=[gap_function, J_n],
        )
        wp.launch(
            contact_residual_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[res],
        )
        wp.launch(
            contact_residual_derivative_wrt_lambda_n_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[dres_n_dlambda_n],
        )
        wp.launch(
            contact_residual_derivative_wrt_body_qd_kernel,
            dim=C,
            inputs=[
                body_qd,
                body_qd_prev,
                shape_body,
                shape_materials,
                contact_count,
                contact_shape0,
                contact_shape1,
                lambda_n,
                gap_function,
                J_n,
                dt,
                sf,
                fba,
                fbb,
                fbe,
            ],
            outputs=[dres_n_dbody_qd],
        )
    graph = capture.graph

    wp.synchronize()
    # Warm-up graph
    wp.capture_launch(graph)
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.capture_launch(graph)
    wp.synchronize()
    graph_time = (time.perf_counter() - start_time) / num_iterations

    print(f"CUDA Graph Launch...........: {graph_time * 1000:.4f} ms per iteration")

    # --- 3. Report Results ---
    if graph_time > 1e-9:
        speedup = standard_time / graph_time
        print(f"Speedup from CUDA Graph.....: {speedup:.2f}x")
    else:
        print("CUDA Graph time is negligible.")
    print("--------------------------------------------------------------------------")


if __name__ == "__main__":
    wp.init()

    # Run tests with different scales
    run_performance_benchmark(num_bodies=50, num_contacts=100)
    run_performance_benchmark(num_bodies=500, num_contacts=500)
    run_performance_benchmark(num_bodies=1000, num_contacts=2000)
    run_performance_benchmark(num_bodies=2000, num_contacts=5000)
