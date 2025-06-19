import warp as wp
from axion.contact_constraint import contact_constraint_kernel
from axion.dynamics_constraint import contact_contribution_kernel
from axion.dynamics_constraint import unconstrained_dynamics_kernel
from warp.sim import Model
from warp.sim import State


CONTACT_CONSTRAINT_STABILIZATION = 0.1  # Baumgarte stabilization factor
CONTACT_FB_ALPHA = 1.0  # Fisher-Burmeister scaling factor of the first argument
CONTACT_FB_BETA = 0.0  # Fisher-Burmeister scaling factor of the second argument


def linearize_system(
    model: Model,
    state: State,
    state_prev: State,
    dt: float,
    lambda_n: wp.array,
    # --- Outputs ---
    res: wp.array,
    jacobian: wp.array,
):
    B = model.body_count
    C = model.rigid_contact_max

    # Assume the residual is shape [6B + C]
    # Assume that the jacobian is shape [6B + C, 6B + C]

    # Get the offset for the residuals
    res_d_offset = 0
    res_n_offset = 6 * B

    # Get the offset for the derivatives in the jacobian
    dres_d_dbody_qd_offset = wp.vec2i(0, 0)
    dres_d_dlambda_n_offset = wp.vec2i(6 * B, 0)
    dres_n_dbody_qd_offset = wp.vec2i(0, 6 * B)
    dres_n_dlambda_n_offset = wp.vec2i(6 * B, 6 * B)

    # Clean up the output arrays
    res.zero_()
    jacobian.zero_()

    # Compute the dynamics contact constraint
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=B,
        inputs=[
            state.body_qd,
            state_prev.body_qd,
            model.body_mass,
            model.body_inertia,
            dt,
            model.gravity,
            res_d_offset,
            dres_d_dbody_qd_offset,
        ],
        outputs=[res, jacobian],
        device=model.device,
    )
    wp.launch(
        kernel=contact_contribution_kernel,
        dim=C,
        inputs=[
            state.body_q,
            model.body_com,
            model.shape_body,
            model.shape_geo,
            model.rigid_contact_count,
            model.rigid_contact_point0,
            model.rigid_contact_point1,
            model.rigid_contact_normal,
            model.rigid_contact_shape0,
            model.rigid_contact_shape1,
            lambda_n,
            dt,
            res_d_offset,
            dres_d_dlambda_n_offset,
        ],
        outputs=[res, jacobian],
    )

    # Compute the contact constraint
    wp.launch(
        kernel=contact_constraint_kernel,
        dim=C,
        inputs=[
            state.body_q,
            state.body_qd,
            state_prev.body_q,
            model.body_com,
            model.shape_body,
            model.shape_geo,
            model.shape_materials,
            model.rigid_contact_count,
            model.rigid_contact_point0,
            model.rigid_contact_point1,
            model.rigid_contact_normal,
            model.rigid_contact_shape0,
            model.rigid_contact_shape1,
            lambda_n,
            dt,
            CONTACT_CONSTRAINT_STABILIZATION,
            CONTACT_FB_ALPHA,
            CONTACT_FB_BETA,
            res_n_offset,
            dres_n_dbody_qd_offset,
            dres_n_dlambda_n_offset,
        ],
        outputs=[res, jacobian],
    )
