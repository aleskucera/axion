import numpy as np
import warp as wp
from axion.contact_constraint import contact_info_kernel
from axion.contact_constraint import contact_residual_derivative_wrt_body_qd_kernel
from axion.contact_constraint import contact_residual_derivative_wrt_lambda_n_kernel
from axion.contact_constraint import contact_residual_kernel
from axion.warp_utils import add_value_to_array_item
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


def test_derivative_wrt_lambda_n(seed: int = 0):
    wp.init()

    B = 2  # Two bodies
    C = 1  # One contact

    dt = 0.01
    stabilization_factor = 0.1
    fb_alpha = 0.25
    fb_beta = 0.25
    fb_epsilon = 1e-6

    body_q = wp.array(
        np.random.rand(B, 7).astype(np.float32),
        dtype=wp.transform,
    )

    # TODO: Normalize the quaternions in body_q

    body_qd = wp.array(
        np.random.rand(B, 6).astype(np.float32),
        dtype=wp.spatial_vector,
    )

    body_qd_prev = wp.array(
        np.random.rand(B, 6).astype(np.float32),
        dtype=wp.spatial_vector,
    )

    body_com = wp.array(
        np.random.rand(B, 3).astype(np.float32),
        dtype=wp.vec3,
    )

    shape_body = wp.array([0, 1], dtype=int)

    contact_count = wp.array([1], dtype=int)
    contact_point0 = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_point1 = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_normal = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_shape0 = wp.array([0], dtype=wp.int32)
    contact_shape1 = wp.array([1], dtype=wp.int32)

    lambda_n = wp.array(np.random.rand(C).astype(np.float32), dtype=wp.float32)

    geo = ModelShapeGeometry()
    geo.thickness = wp.array(np.zeros(2).astype(np.float32), dtype=wp.float32)
    shape_materials = ModelShapeMaterials()
    shape_materials.restitution = wp.array(
        np.random.rand(2).astype(np.float32), dtype=wp.float32
    )

    # Output arrays
    gap_function = wp.zeros(C, dtype=wp.float32)
    J_n = wp.zeros((C, 2), dtype=wp.spatial_vector)
    res = wp.zeros(C, dtype=wp.float32)
    dres_n_dlambda_n = wp.zeros((C, C), dtype=wp.float32)

    # Compute the derivative
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
            stabilization_factor,
            fb_alpha,
            fb_beta,
            fb_epsilon,
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
            stabilization_factor,
            fb_alpha,
            fb_beta,
            fb_epsilon,
        ],
        outputs=[dres_n_dlambda_n],
    )
    res_np = res.numpy()
    dres_n_dlambda_n_np = dres_n_dlambda_n.numpy()

    # Compute the derivative using the finite difference method
    eps = 1e-3
    dres_n_dlambda_n_fd_np = np.zeros((C, C), dtype=np.float32)
    for i in range(C):  # Loop over each input (lambda_n dimension)
        # Initialize the perturbed array
        res_perturbed = wp.zeros(C, dtype=wp.float32)

        # Perturb the lambda_n value
        add_value_to_array_item(lambda_n, (i,), eps)  # lambda_n[i] += eps

        # Compute the perturbed residual
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
                stabilization_factor,
                fb_alpha,
                fb_beta,
                fb_epsilon,
            ],
            outputs=[res_perturbed],
        )
        res_perturbed_np = res_perturbed.numpy()
        for j in range(C):  # Loop over each output (residual dimension)
            dres_n_dlambda_n_fd_np[j, i] = (res_perturbed_np[j] - res_np[j]) / eps

        add_value_to_array_item(
            lambda_n, (i,), -eps
        )  # lambda_n[i] -= eps, resetting it

    print(f"Residual: {res_np}")
    print(f"Derivative of residual wrt lambda_n: {dres_n_dlambda_n_np}")
    print(f"Finite difference derivative: {dres_n_dlambda_n_fd_np}")

    # Check if the computed derivative matches the finite difference method
    assert np.allclose(
        dres_n_dlambda_n_np, dres_n_dlambda_n_fd_np, atol=1e-3
    ), "The computed derivative does not match the finite difference method."


def test_derivative_wrt_body_qd(seed: int = 0):
    wp.init()

    B = 2  # Two bodies
    C = 1  # One contact

    dt = 0.01
    stabilization_factor = 0.1
    fb_alpha = 0.16
    fb_beta = 0.16
    fb_epsilon = 1e-6

    body_q = wp.array(
        np.random.rand(B, 7).astype(np.float32),
        dtype=wp.transform,
    )

    # TODO: Normalize the quaternions in body_q

    body_qd = wp.array(
        10 * np.random.rand(B, 6).astype(np.float32),
        dtype=wp.spatial_vector,
    )

    body_qd_prev = wp.array(
        10 * np.random.rand(B, 6).astype(np.float32),
        dtype=wp.spatial_vector,
    )

    body_com = wp.array(
        np.random.rand(B, 3).astype(np.float32),
        dtype=wp.vec3,
    )

    shape_body = wp.array([0, 1], dtype=int)

    contact_count = wp.array([1], dtype=int)
    contact_point0 = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_point1 = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_normal = wp.array(np.random.rand(C, 3).astype(np.float32), dtype=wp.vec3)
    contact_shape0 = wp.array([0], dtype=wp.int32)
    contact_shape1 = wp.array([1], dtype=wp.int32)

    lambda_n = wp.array(np.random.rand(C).astype(np.float32), dtype=wp.float32)

    geo = ModelShapeGeometry()
    geo.thickness = wp.array(np.zeros(2).astype(np.float32), dtype=wp.float32)
    shape_materials = ModelShapeMaterials()
    shape_materials.restitution = wp.array(
        np.random.rand(2).astype(np.float32), dtype=wp.float32
    )

    # Output arrays
    gap_function = wp.zeros(C, dtype=wp.float32)
    J_n = wp.zeros((C, 2), dtype=wp.spatial_vector)
    res = wp.zeros(C, dtype=wp.float32)
    dres_n_dbody_qd = wp.zeros((C, 6 * B), dtype=wp.float32)

    # Compute the derivative
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
            stabilization_factor,
            fb_alpha,
            fb_beta,
            fb_epsilon,
        ],
        outputs=[res],
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
            stabilization_factor,
            fb_alpha,
            fb_beta,
            fb_epsilon,
        ],
        outputs=[dres_n_dbody_qd],
    )
    res_np = res.numpy()
    dres_n_dbody_qd_np = dres_n_dbody_qd.numpy()

    # Compute the derivative using the finite difference method
    eps = 1e-3
    dres_n_dbody_qd_fd_np = np.zeros((C, 6 * B), dtype=np.float32)
    for b in range(B):  # Loop over each input (body_qd dimension)
        for i in range(6):  # Loop over each dimension of body_qd
            res_perturbed = wp.zeros(C, dtype=wp.float32)

            # Perturb the lambda_n value
            perturbation = np.zeros(6, dtype=np.float32)
            perturbation[i] = eps

            add_value_to_array_item(body_qd, (b,), wp.spatial_vector(perturbation))

            # Compute the perturbed residual
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
                    stabilization_factor,
                    fb_alpha,
                    fb_beta,
                    fb_epsilon,
                ],
                outputs=[res_perturbed],
            )
            res_perturbed_np = res_perturbed.numpy()

            for c in range(C):  # Loop over each output (residual dimension)
                dres_n_dbody_qd_fd_np[c, b * 6 + i] = (
                    res_perturbed_np[c] - res_np[c]
                ) / eps

            # Reset the perturbation
            add_value_to_array_item(body_qd, (b,), wp.spatial_vector(-perturbation))

    print(f"Residual: {res_np}")
    print(f"Derivative of residual wrt body_qd: {dres_n_dbody_qd_np}")
    print(f"Finite difference derivative: {dres_n_dbody_qd_fd_np}")

    # Check if the computed derivative matches the finite difference method
    assert np.allclose(
        dres_n_dbody_qd_np, dres_n_dbody_qd_fd_np, atol=1e-3
    ), "The computed derivative does not match the finite difference method."


if __name__ == "__main__":
    test_derivative_wrt_lambda_n()
    test_derivative_wrt_body_qd()
