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
    grad_c_n_a: wp.spatial_vector,
    grad_c_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    c_n: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    # Relative normal velocity at the current time step
    delta_v_n = wp.dot(grad_c_n_a, body_qd_a) + wp.dot(grad_c_n_b, body_qd_b)

    # Relative normal velocity at the previous time step
    delta_v_n_prev = wp.dot(grad_c_n_a, body_qd_prev_a) + wp.dot(
        grad_c_n_b, body_qd_prev_b
    )

    # Baumgarte stabilization bias from penetration depth
    b_err = stabilization_factor / dt * c_n

    # Restitution bias from previous velocity
    b_rest = -restitution * delta_v_n_prev

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
    Calculates contact constraint (c_n), time derivatives (grad_c_n_a, grad_c_n_b),
    and body indices for a single contact.

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
    c_n = wp.dot(n, p_world_a - p_world_b)

    # If gap is non-negative, the contact is not penetrating and thus inactive
    if c_n >= 0.0:
        return False, 0.0, wp.spatial_vector(), wp.spatial_vector(), body_a, body_b

    # Compute Jacobians
    grad_c_n_a = wp.spatial_vector(wp.cross(r_a, n), n)
    grad_c_n_b = wp.spatial_vector(-wp.cross(r_b, n), -n)

    return True, c_n, grad_c_n_a, grad_c_n_b, body_a, body_b


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
    h_n_offset: wp.int32,  # Goes into vector h
    J_n_offset: wp.int32,  # Goes into the matrix J
    C_n_offset: wp.int32,  # Goes into the matrix C
    # --- Outputs ---
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
    contact_body_a: wp.array(dtype=wp.int32),  # [C]
    contact_body_b: wp.array(dtype=wp.int32),  # [C]
):
    tid = wp.tid()

    # Handle inactive constraints (not detected by collide func)
    if tid >= contact_count[0]:
        return

    is_active, c_n, grad_c_n_a, grad_c_n_b, body_a, body_b = (
        _compute_contact_kinematics(
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
    )
    # Store body indices for the contact
    contact_body_a[tid] = body_a
    contact_body_b[tid] = body_b

    # Check that the constraint is really violated
    if not is_active:
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

    # Compute complementarity argument to the constraint impulse lambda_n
    complementarity_arg = _compute_complementarity_argument(
        grad_c_n_a,
        grad_c_n_b,
        body_qd_a,
        body_qd_b,
        body_qd_prev_a,
        body_qd_prev_b,
        c_n,
        e,
        dt,
        stabilization_factor,
    )

    phi_n, dphi_dlambda_n, dphi_db = scaled_fisher_burmeister(
        lambda_n[tid], complementarity_arg, fb_alpha, fb_beta
    )

    J_n_a = dphi_db * grad_c_n_a
    J_n_b = dphi_db * grad_c_n_b

    # --- g --- (momentum balance)
    if body_a >= 0:
        g_a = -J_n_a * lambda_n[tid]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, body_a * 6 + st_i, g_a[st_i])

    if body_b >= 0:
        g_b = -J_n_b * lambda_n[tid]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(g, body_b * 6 + st_i, g_b[st_i])

    # --- h --- (vector of the constraint errors)
    h[h_n_offset + tid] = phi_n

    # --- C --- (compliance block)
    C_values[C_n_offset + tid] = dphi_dlambda_n

    # --- J --- (constraint Jacobian block)
    if body_a >= 0:
        offset = J_n_offset + tid
        J_values[offset, 0] = J_n_a

    if body_b >= 0:
        offset = J_n_offset + tid
        J_values[offset, 1] = J_n_b


def setup_data(num_bodies, num_contacts, device):
    """Generates all necessary input and output arrays for the kernel."""
    B, C = num_bodies, num_contacts
    # We assume num_shapes == num_bodies for this test
    # Generate contacts between two different random bodies
    shape0 = np.random.randint(0, B, size=C, dtype=np.int32)
    shape1 = np.random.randint(0, B, size=C, dtype=np.int32)
    mask = shape0 == shape1
    while np.any(mask):
        shape1[mask] = np.random.randint(0, B, size=np.sum(mask), dtype=np.int32)
        mask = shape0 == shape1

    data = {
        # --- Inputs ---
        "body_q": wp.array(np.random.rand(B, 7), dtype=wp.transform, device=device),
        "body_qd": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_qd_prev": wp.array(
            np.random.rand(B, 6), dtype=wp.spatial_vector, device=device
        ),
        "body_com": wp.array(np.random.rand(B, 3), dtype=wp.vec3, device=device),
        "shape_body": wp.array(np.arange(B, dtype=wp.int32), dtype=int, device=device),
        "shape_geo": ModelShapeGeometry(),
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
        # --- Parameters ---
        "params": {
            "dt": 0.01,
            "stabilization_factor": 0.2,
            "fb_alpha": 0.25,
            "fb_beta": 0.25,
            "h_n_offset": 0,
            "J_n_offset": 0,
            "C_n_offset": 0,
        },
        # --- Outputs ---
        "g": wp.zeros(B * 6, dtype=wp.float32, device=device),
        "h": wp.zeros(C, dtype=wp.float32, device=device),
        # Each contact contributes two 6-dof Jacobians (12 floats)
        "J_values": wp.zeros(C * 12, dtype=wp.float32, device=device),
        "C_values": wp.zeros(C, dtype=wp.float32, device=device),
    }

    # Populate geometry and material properties for all bodies/shapes
    data["shape_geo"].thickness = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    data["shape_materials"].restitution = wp.array(
        np.random.rand(B), dtype=wp.float32, device=device
    )
    return data


def run_benchmark(num_bodies, num_contacts, num_iterations=200):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Kernel: B={num_bodies}, C={num_contacts}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, num_contacts, device)
    params = data["params"]

    # Assemble the list of arguments in the exact order the kernel expects.
    # This includes all inputs, parameters, and outputs.
    kernel_args = [
        data["body_q"],
        data["body_qd"],
        data["body_qd_prev"],
        data["body_com"],
        data["shape_body"],
        data["shape_geo"],
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
        params["h_n_offset"],
        params["J_n_offset"],
        params["C_n_offset"],
        data["g"],
        data["h"],
        data["J_values"],
        data["C_values"],
    ]

    # --- 1. Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    # Warm-up launch
    wp.launch(
        kernel=contact_constraint_kernel,
        dim=num_contacts,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=num_contacts,
            inputs=kernel_args,
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
                kernel=contact_constraint_kernel,
                dim=num_contacts,
                inputs=kernel_args,
                device=device,
            )
        graph = capture.graph
        # Warm-up launch
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
