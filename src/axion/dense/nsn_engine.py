import numpy as np
import warp as wp
from axion.contact_constraint import contact_constraint_kernel
from axion.dynamics_constraint import contact_contribution_kernel
from axion.dynamics_constraint import unconstrained_dynamics_kernel
from axion.utils.add_inplace import add_inplace
from warp.optim.linear import bicgstab
from warp.optim.linear import cg
from warp.optim.linear import gmres
from warp.optim.linear import preconditioner
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State


CONTACT_CONSTRAINT_STABILIZATION = 0.1  # Baumgarte stabilization factor
CONTACT_FB_ALPHA = 0.25  # Fisher-Burmeister scaling factor of the first argument
CONTACT_FB_BETA = 0.25  # Fisher-Burmeister scaling factor of the second argument


def linearize_system(
    model: Model,
    state_in: State,
    state_out: State,
    dt: float,
    lambda_n: wp.array,
    # --- Outputs ---
    neg_res: wp.array,
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
    dres_d_dlambda_n_offset = wp.vec2i(0, 6 * B)
    dres_n_dbody_qd_offset = wp.vec2i(6 * B, 0)
    dres_n_dlambda_n_offset = wp.vec2i(6 * B, 6 * B)

    # Clean up the output arrays
    neg_res.zero_()
    jacobian.zero_()

    # Compute the dynamics contact constraint
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=B,
        inputs=[
            state_out.body_qd,
            state_in.body_qd,
            state_out.body_f,
            model.body_mass,
            model.body_inertia,
            dt,
            model.gravity,
            res_d_offset,
            dres_d_dbody_qd_offset,
        ],
        outputs=[neg_res, jacobian],
        device=model.device,
    )
    wp.launch(
        kernel=contact_contribution_kernel,
        dim=C,
        inputs=[
            state_out.body_q,
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
            res_d_offset,
            dres_d_dlambda_n_offset,
        ],
        outputs=[neg_res, jacobian],
    )

    # Compute the contact constraint
    wp.launch(
        kernel=contact_constraint_kernel,
        dim=C,
        inputs=[
            state_out.body_q,
            state_out.body_qd,
            state_in.body_qd,
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
        outputs=[neg_res, jacobian],
    )


def add_delta_x(
    delta_x: wp.array,
    body_qd: wp.array,
    lambda_n: wp.array,
    d_offset: int,
    n_offset: int,
):

    B = body_qd.shape[0]
    C = lambda_n.shape[0]
    add_inplace(body_qd, delta_x, 0, d_offset, B)
    add_inplace(lambda_n, delta_x, 0, n_offset, C)


class NSNEngine(Integrator):
    def __init__(
        self,
        tolerance: float = 1e-4,
        max_iterations: int = 10,
        regularization: float = 1e-4,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.regularization = regularization

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
    ):
        B = model.body_count
        C = model.rigid_contact_max

        if B == 0:
            raise ValueError("State must contain at least one body.")

        if state_in.particle_count > 0:
            raise ValueError("NSNEngine does not support particles.")

        if control is None:
            control = model.control(clone_variables=False)

        # Get the initial guess for the output state. This will be used as the starting point for the iterative solver.
        self.integrate_bodies(model, state_in, state_out, dt)

        lambda_n = wp.zeros((C,), dtype=wp.float32, device=model.device)
        neg_res = wp.zeros((6 * B + C,), dtype=wp.float32, device=model.device)
        jacobian = wp.zeros(
            (6 * B + C, 6 * B + C), dtype=wp.float32, device=model.device
        )
        delta_x = wp.zeros((6 * B + C), device=model.device)

        is_collision = model.rigid_contact_count.numpy()[0] > 0

        # if is_collision:
        #     print("COLLISION DETECTED")

        for _ in range(self.max_iterations):
            # Compute the linearized system
            linearize_system(
                model, state_in, state_out, dt, lambda_n, neg_res, jacobian
            )

            # print("Residual: ", -neg_res.numpy())
            res_norm = np.linalg.norm(neg_res.numpy())
            if res_norm < 0.1:
                # print(f"Converged with residual norm: {res_norm}")
                break

            M = preconditioner(jacobian, ptype="diag")

            # Solve the linear system
            delta_x.zero_()
            _, _, _ = gmres(
                A=jacobian,
                b=neg_res,
                x=delta_x,
                restart=64,
                tol=self.tolerance,
                # atol=self.tolerance,
                maxiter=300,
                M=M,
            )

            add_delta_x(delta_x, state_out.body_qd, lambda_n, 0, 6 * B)
