import warp as wp


@wp.kernel
def unconstrained_dynamics_residual_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),  # [B]
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # [B]
    body_f: wp.array(dtype=wp.spatial_vector),  # [B]
    gravity: wp.vec3,  # [3]
    body_mass: wp.array(dtype=wp.float32),  # [B]
    body_inertia: wp.array(dtype=wp.mat3),  # [B]
    dt: wp.float32,
    # Outputs:
    res: wp.array(dtype=wp.float32),  # [6B]
):
    tid = wp.tid()

    # Make sure we don't access out of bounds
    if tid >= body_qd.shape[0]:
        return

    # Get the body velocities and forces
    w = wp.spatial_top(body_qd[tid])  # Angular velocity
    v = wp.spatial_bottom(body_qd[tid])  # Linear velocity

    w_prev = wp.spatial_top(body_qd_prev[tid])
    v_prev = wp.spatial_bottom(body_qd_prev[tid])

    t = wp.spatial_top(body_f[tid])  # Torque
    f = wp.spatial_bottom(body_f[tid])  # Force

    # Compute the residuals
    res_lin = body_mass[tid] * (v - v_prev) - f * dt - body_mass[tid] * gravity
    res_ang = wp.dot(body_inertia[tid], (w - w_prev)) - t * dt
    res = wp.spatial_vector(res_lin, res_ang)

    res[tid * 6 : (tid + 1) * 6] = res


@wp.kernel
def dynamics_residual_derivative_wrt_body_qd(
    body_mass: wp.array(dtype=wp.float32),  # [B]
    body_inertia: wp.array(dtype=wp.mat3),  # [B]
    # Outputs:
    dres_d_dbody_qd: wp.array(dtype=wp.float32),  # [6B, 6B]
):
    tid = wp.tid()

    # Make sure we don't access out of bounds
    if tid >= body_mass.shape[0]:
        return

    # Linear components: mass matrix (diagonal)
    dres_d_dbody_qd[6 * tid + 0, 6 * tid + 0] = body_mass[tid]
    dres_d_dbody_qd[6 * tid + 1, 6 * tid + 1] = body_mass[tid]
    dres_d_dbody_qd[6 * tid + 2, 6 * tid + 2] = body_mass[tid]

    # Angular components: full inertia matrix
    dres_d_dbody_qd[6 * tid + 3, 6 * tid + 3] = body_inertia[tid][0, 0]
    dres_d_dbody_qd[6 * tid + 3, 6 * tid + 4] = body_inertia[tid][0, 1]
    dres_d_dbody_qd[6 * tid + 3, 6 * tid + 5] = body_inertia[tid][0, 2]
    dres_d_dbody_qd[6 * tid + 4, 6 * tid + 3] = body_inertia[tid][1, 0]
    dres_d_dbody_qd[6 * tid + 4, 6 * tid + 4] = body_inertia[tid][1, 1]
    dres_d_dbody_qd[6 * tid + 4, 6 * tid + 5] = body_inertia[tid][1, 2]
    dres_d_dbody_qd[6 * tid + 5, 6 * tid + 3] = body_inertia[tid][2, 0]
    dres_d_dbody_qd[6 * tid + 5, 6 * tid + 4] = body_inertia[tid][2, 1]
    dres_d_dbody_qd[6 * tid + 5, 6 * tid + 5] = body_inertia[tid][2, 2]


@wp.kernel
def accumulate_contact_impulse_kernel(
    lambda_n: wp.array(dtype=wp.float32),  # [C]
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
    shape_body: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    # Outputs:
    res: wp.array(dtype=wp.float32),  # [6B]
):
    tid = wp.tid()

    # Make sure we don't access out of bounds
    if tid >= lambda_n.shape[0]:
        return

    # Calculate the contribution of the contact impulse to the residual
    contact_contrib_a = -J_n[tid, 0] * lambda_n[tid]  # Contribution from body A
    contact_contrib_b = -J_n[tid, 1] * lambda_n[tid]  # Contribution from body B

    # Get the shapes involved in the contact
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return  # Skip self-contact

    # Get the bodies associated with the shapes
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Accumulate the contributions to the residual vector
    if body_a >= 0:
        wp.atomic_add(res, body_a * 6, contact_contrib_a)
    if body_b >= 0:
        wp.atomic_add(res, body_b * 6, contact_contrib_b)


@wp.kernel
def dynamics_residual_derivative_wrt_lambda_n(
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [C, 2]
    shape_body: wp.array(dtype=int),  # [B]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    # Outputs:
    dres_d_dlambda_n: wp.array(dtype=wp.float32),  # [6B, C]
):
    tid = wp.tid()

    # Make sure we don't access out of bounds
    if tid >= J_n.shape[0]:
        return

    # Get the shapes involved in the contact
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    # Get the bodies associated with the shapes
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Get the Jacobian for the contact
    J_n_a = J_n[tid, 0]
    J_n_b = J_n[tid, 1]

    # Compute the derivative
    if body_a >= 0:
        for i in range(wp.static(6)):
            dres_d_dlambda_n[body_a * 6 + wp.static(i), tid] = -J_n_a[wp.static(i)]
    if body_b >= 0:
        for i in range(wp.static(6)):
            dres_d_dlambda_n[body_b * 6 + wp.static(i), tid] = -J_n_b[wp.static(i)]


def dynamics_residual(
    body_qd: wp.array,  # [B] (dtype=wp.spatial_vector)
    body_qd_prev: wp.array,  # [B] (dtype=wp.spatial_vector)
    body_f: wp.array,  # [B] (dtype=wp.spatial_vector)
    gravity: wp.vec3,  # [3]
    body_mass: wp.array,  # [B] (dtype=wp.float32)
    body_inertia: wp.array,  # [B] (dtype=wp.mat3)
    lambda_n: wp.array,  # [C] (dtype=wp.float32)
    J_n: wp.array,  # [C, 2] (dtype=wp.spatial_vector)
    shape_body: wp.array,  # [B] (dtype=int)
    contact_shape0: wp.array,  # [C] (dtype=wp.int32)
    contact_shape1: wp.array,  # [C] (dtype=wp.int32)
    dt: float,
    # Outputs:
    res: wp.array,  # [6B] (dtype=wp.spatial_vector)
) -> None:
    B = body_qd.shape[0]
    C = lambda_n.shape[0]

    # Run the unconstrained dynamics residual kernel
    wp.launch(
        unconstrained_dynamics_residual_kernel,
        dim=B,
        inputs=[
            body_qd,
            body_qd_prev,
            body_f,
            gravity,
            body_mass,
            body_inertia,
            dt,
        ],
        outputs=[res],
    )
    # Run the contact impulse accumulation kernel
    wp.launch(
        accumulate_contact_impulse_kernel,
        dim=C,
        inputs=[
            lambda_n,
            J_n,
            shape_body,
            contact_shape0,
            contact_shape1,
        ],
        outputs=[res],
    )
