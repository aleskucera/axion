import time

import numpy as np
import warp as wp
from axion.ncp import scaled_fisher_burmeister
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.func
def _compute_restitution_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
    """Computes the average coefficient of restitution for a contact pair."""
    e = 0.0
    if shape_a >= 0 and shape_b >= 0:
        e_a = shape_materials.restitution[shape_a]
        e_b = shape_materials.restitution[shape_b]
        e = (e_a + e_b) * 0.5
    elif shape_a >= 0:
        e = shape_materials.restitution[shape_a]
    elif shape_b >= 0:
        e = shape_materials.restitution[shape_b]
    return e


@wp.func
def _compute_complementarity_argument(
    J_n_a: wp.spatial_vector,
    J_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    gap: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    """
    Computes the second argument for the Fisher-Burmeister function,
    which represents the complementarity condition (v_n + bias).
    """
    # Relative normal velocity at the current time step
    delta_v_n = wp.dot(J_n_a, body_qd_a) + wp.dot(J_n_b, body_qd_b)

    # Relative normal velocity at the previous time step
    delta_v_n_prev = wp.dot(J_n_a, body_qd_prev_a) + wp.dot(J_n_b, body_qd_prev_b)

    # Baumgarte stabilization bias from penetration depth
    b_err = stabilization_factor / dt * gap
    # b_err = 0.0

    # Restitution bias from previous velocity
    b_rest = -restitution * delta_v_n_prev
    # b_rest = 0.0

    return delta_v_n + b_err + b_rest


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
    geo: ModelShapeGeometry,
):
    """
    Calculates gap, Jacobians, and body indices for a single contact.
    Returns a bool 'is_active' and the computed values.
    """
    # Guard against self-contact
    if shape_a == shape_b:
        return False, 0.0, wp.spatial_vector(), wp.spatial_vector(), -1, -1

    # Get body indices and thickness
    body_a, body_b = -1, -1
    thickness_a, thickness_b = 0.0, 0.0
    if shape_a >= 0:
        body_a = shape_body[shape_a]
        thickness_a = geo.thickness[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
        thickness_b = geo.thickness[shape_b]

    # Compute world-space contact points and lever arms (r_a, r_b)
    n = normal
    p_world_a = point_a
    p_world_b = point_b
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        offset_a = -thickness_a * normal
        p_world_a = wp.transform_point(X_wb_a, point_a) + offset_a
        r_a = p_world_a - wp.transform_point(X_wb_a, body_com[body_a])
    if body_b >= 0:
        X_wb_b = body_q[body_b]
        offset_b = thickness_b * normal
        p_world_b = wp.transform_point(X_wb_b, point_b) + offset_b
        r_b = p_world_b - wp.transform_point(X_wb_b, body_com[body_b])

    # Calculate penetration depth (gap)
    gap = wp.dot(n, p_world_a - p_world_b)

    # If gap is non-negative, the contact is not penetrating and thus inactive
    if gap >= 0.0:
        return False, 0.0, wp.spatial_vector(), wp.spatial_vector(), body_a, body_b

    # Compute Jacobians
    J_n_a = wp.spatial_vector(wp.cross(r_a, n), n)
    J_n_b = wp.spatial_vector(-wp.cross(r_b, n), -n)

    return True, gap, J_n_a, J_n_b, body_a, body_b


@wp.kernel
def contact_constraint_kernel(
    body_q: wp.array(dtype=wp.transform),  # [B, 7]
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B, 6]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B, 6]
    body_com: wp.array(dtype=wp.vec3),  # [B, 3]
    shape_body: wp.array(dtype=int),  # [B]
    shape_geo: ModelShapeGeometry,
    shape_materials: ModelShapeMaterials,
    contact_count: wp.array(dtype=wp.int32),  # [1]
    contact_point0: wp.array(dtype=wp.vec3),  # [C, 3]
    contact_point1: wp.array(dtype=wp.vec3),  # [C, 3]
    contact_normal: wp.array(dtype=wp.vec3),  # [C, 3]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    # --- Parameters ---
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    # Indices for outputs
    res_n_offset: wp.int32,
    dres_n_dbody_qd_offset: wp.vec2i,
    dres_n_dlambda_n_offset: wp.vec2i,
    # --- Outputs ---
    neg_res: wp.array(dtype=wp.float32),
    jacobian: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()

    # Get the indices for the scalar outputs (residual and derivatives wrt lambda_n)
    res_idx = res_n_offset + tid
    jac_lambda_n_idx = dres_n_dlambda_n_offset + wp.vec2i(tid, tid)

    # Handle inactive constraints (outside the contact count)
    if tid >= contact_count[0]:
        # res[res_idx] = -lambda_n[tid]
        # jacobian[jac_lambda_n_idx.x, jac_lambda_n_idx.y] = -1.0
        neg_res[res_idx] = 0.0
        jacobian[jac_lambda_n_idx.x, jac_lambda_n_idx.y] = 1e-6
        return

    is_active, gap, J_n_a, J_n_b, body_a, body_b = _compute_contact_kinematics(
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
        neg_res[res_idx] = 0.0
        return

    e = _compute_restitution_coefficient(
        contact_shape0[tid], contact_shape1[tid], shape_materials
    )

    # Safely get body velocities, handling ground body (-1)
    body_qd_a = wp.spatial_vector()
    body_qd_prev_a = wp.spatial_vector()
    if body_a >= 0:
        body_qd_a = body_qd[body_a]
        body_qd_prev_a = body_qd_prev[body_a]

    body_qd_b = wp.spatial_vector()
    body_qd_prev_b = wp.spatial_vector()
    if body_b >= 0:
        body_qd_b = body_qd[body_b]
        body_qd_prev_b = body_qd_prev[body_b]

    complementarity_arg = _compute_complementarity_argument(
        J_n_a,
        J_n_b,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        gap,
        e,
        dt,
        stabilization_factor,
    )

    res_n, dfb_da, dfb_db = scaled_fisher_burmeister(
        lambda_n[tid], complementarity_arg, fb_alpha, fb_beta
    )

    # Store the residual
    neg_res[res_idx] = -res_n

    # ∂res_n / ∂lambda_n [C, C]
    jacobian[jac_lambda_n_idx.x, jac_lambda_n_idx.y] = dfb_da

    # ∂res_n / ∂body_qd [C, B6]
    if body_a >= 0:
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            x_off = dres_n_dbody_qd_offset.x + tid
            y_off = dres_n_dbody_qd_offset.y + body_a * 6
            jacobian[x_off, y_off + st_i] = dfb_db * J_n_a[i]

    if body_b >= 0:
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            x_off = dres_n_dbody_qd_offset.x + tid
            y_off = dres_n_dbody_qd_offset.y + body_b * 6
            jacobian[x_off, y_off + st_i] = dfb_db * J_n_b[i]


def setup_data(num_bodies, num_contacts, device):
    """Generates all necessary input and output arrays for the fully fused kernel."""
    B, C = num_bodies, num_contacts
    shape0 = np.random.randint(0, B, size=C, dtype=np.int32)
    shape1 = np.random.randint(0, B, size=C, dtype=np.int32)
    mask = shape0 == shape1
    while np.any(mask):
        shape1[mask] = np.random.randint(0, B - 1, size=np.sum(mask), dtype=np.int32)
        mask = shape0 == shape1

    data = {
        "body_q": wp.array(np.random.rand(B, 7), dtype=wp.transform, device=device),
        "body_com": wp.array(np.random.rand(B, 3), dtype=wp.vec3, device=device),
        "shape_geo": ModelShapeGeometry(),
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "shape_body": wp.array(np.arange(B, dtype=np.int32), dtype=int, device=device),
        "shape_materials": ModelShapeMaterials(),
        "contact_count": wp.array([C], dtype=wp.int32, device=device),
        "contact_point0": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_point1": wp.array(np.random.rand(C, 3), dtype=wp.vec3, device=device),
        "contact_normal": wp.array(
            np.random.rand(C, 3) - 0.5, dtype=wp.vec3, device=device
        ),
        "contact_shape0": wp.array(shape0, dtype=wp.int32, device=device),
        "contact_shape1": wp.array(shape1, dtype=wp.int32, device=device),
        "lambda_n": wp.array(np.random.rand(C), dtype=wp.float32, device=device),
        "params": {
            "dt": 0.01,
            "stabilization_factor": 0.2,
            "fb_alpha": 0.25,
            "fb_beta": 0.25,
            "res_n_offset": 0,
            "dres_n_dbody_qd_offset": wp.vec2i(0, 0),
            "dres_n_dlambda_n_offset": wp.vec2i(C, 6 * B),
        },
        "res": wp.zeros(C, dtype=wp.float32, device=device),
        "jacobian": wp.zeros((2 * C, C + 6 * B), dtype=wp.float32, device=device),
    }

    data["geo"].thickness = wp.array(np.random.rand(B), dtype=wp.float32, device=device)
    data["shape_materials"].restitution = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    return data


def run_benchmark(num_bodies, num_contacts, num_iterations=200):
    """Measures execution time of the fully fused kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking FULLY FUSED Kernel: B={num_bodies}, C={num_contacts}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, num_contacts, device)
    params = data["params"]

    kernel_inputs = [
        data["body_q"],
        data["body_qd"],
        data["body_qd_prev"],
        data["body_com"],
        data["shape_geo"],
        data["shape_body"],
        data["shape_materials"],
        data["contact_count"],
        data["contact_point0"],
        data["contact_point1"],
        data["contact_normal"],
        data["contact_shape0"],
        data["contact_shape1"],
        data["lambda_n"],
        params["dt"],
        params["stabilization_factor"],
        params["fb_alpha"],
        params["fb_beta"],
        params["res_n_offset"],
        params["dres_n_dbody_qd_offset"],
        params["dres_n_dlambda_n_offset"],
    ]
    kernel_outputs = [data["res"], data["jacobian"]]

    # --- 1. Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    wp.launch(
        contact_constraint_kernel,
        dim=num_contacts,
        inputs=kernel_inputs,
        outputs=kernel_outputs,
        device=device,
    )
    wp.synchronize()  # Warm-up

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            contact_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_inputs,
            outputs=kernel_outputs,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- 2. CUDA Graph Benchmark (only on GPU) ---
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        with wp.ScopedCapture() as capture:
            wp.launch(
                contact_constraint_kernel,
                dim=num_contacts,
                inputs=kernel_inputs,
                outputs=kernel_outputs,
                device=device,
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
        print(
            "Warning: No CUDA device found. Performance will be poor and CUDA graph test will be skipped."
        )

    # Run benchmarks at various scales
    run_benchmark(num_bodies=100, num_contacts=100)
    run_benchmark(num_bodies=500, num_contacts=500)
    run_benchmark(num_bodies=1000, num_contacts=2000)
    run_benchmark(num_bodies=2000, num_contacts=4000)
