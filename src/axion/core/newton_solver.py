import warp as wp
from axion.constraints import linesearch_contact_residuals_kernel
from axion.constraints import linesearch_dynamics_residuals_kernel
from axion.constraints import linesearch_friction_residuals_kernel
from axion.constraints import linesearch_joint_residuals_kernel
from axion.optim.cr import cr_solver
from axion.types import *
from axion.types import GeneralizedMass
from axion.utils import add_inplace

MAX_BODIES = 10
RES_BUFFER_DIM = MAX_BODIES * 6 + 50
ALPHA_DIM = 5

res_buffer_vec = wp.types.vector(length=RES_BUFFER_DIM, dtype=wp.float32)
res_norm_sq_vec = wp.types.vector(length=ALPHA_DIM, dtype=wp.float32)


@wp.kernel
def compute_JT_delta_lambda_kernel(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    delta_lambda: wp.array(dtype=wp.float32),
    # Output array
    JT_delta_lambda: wp.array(dtype=wp.spatial_vector),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]
    dl = delta_lambda[constraint_idx]

    if body_a >= 0:
        JT_delta_lambda[body_a] += dl * J_ia

    if body_b >= 0:
        JT_delta_lambda[body_b] += dl * J_ib


@wp.kernel
def compute_delta_body_qd_kernel(
    gen_inv_mass: wp.array(dtype=GeneralizedMass),
    JT_delta_lambda: wp.array(dtype=wp.spatial_vector),
    g: wp.array(dtype=wp.spatial_vector),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()

    if body_idx >= gen_inv_mass.shape[0]:
        return

    delta_body_qd[body_idx] = gen_inv_mass[body_idx] * (
        JT_delta_lambda[body_idx] - g[body_idx]
    )


@wp.kernel
def buffer_to_sq_norm_kernel(
    res_buffers_vectorized: wp.array(dtype=res_buffer_vec),
    res_norm_sq: wp.array(dtype=wp.float32, ndim=2),
):
    assert res_norm_sq.shape[0] == 1, "Invalid shape of the res_norm_sq array"

    alpha_idx, buff_idx = wp.tid()

    res_buff = res_buffers_vectorized[alpha_idx]

    if buff_idx < MAX_BODIES * 6:
        res_norm_sq[0, alpha_idx] += wp.pow(res_buff[buff_idx], 2.0)
    else:
        res_norm_sq[0, alpha_idx] += res_buff[buff_idx]


@wp.kernel
def update_best_alpha_idx(
    res_norm_sq: wp.array(dtype=res_norm_sq_vec),
    best_alpha_idx: wp.array(dtype=wp.uint32),
):
    assert res_norm_sq.shape[0] == 1, "Invalid shape of the res_norm_sq array"
    assert best_alpha_idx.shape[0] == 1, "Invalid shape of the best_alpha_idx"

    # for i in range(ALPHA_DIM):
    #     wp.printf("\tRes %f\n", res_norm_sq[0][i])
    # wp.printf("\tBest alpha %d", wp.argmin(res_norm_sq[0]))
    # wp.printf("\n")

    best_alpha_idx[0] = wp.argmin(res_norm_sq[0])


@wp.kernel
def update_variables_kernel(
    best_alpha_idx: wp.array(dtype=wp.uint32),
    alphas: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.spatial_vector),
    delta_lambda: wp.array(dtype=wp.float32),
    # Outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
    _lambda: wp.array(dtype=wp.float32),
):
    assert best_alpha_idx.shape[0] == 1, "Invalid shape of the best_alpha_idx"

    tid = wp.tid()

    body_qd_dim = body_qd.shape[0]
    lambda_dim = _lambda.shape[0]

    alpha = alphas[best_alpha_idx[0]]

    if tid < body_qd_dim:
        idx = tid
        body_qd[idx] += alpha * delta_body_qd[idx]
    elif tid < body_qd_dim + lambda_dim:
        idx = tid - body_qd_dim
        _lambda[idx] += alpha * delta_lambda[idx]
    else:
        return


class NewtonSolverMixin:
    def solve_linear_system(self):
        cr_solver(
            A=self.A_op,
            b=self._b,
            x=self._delta_lambda,
            iters=self.linear_iters,
            preconditioner=self.preconditioner,
            logger=self.logger,
        )

        # The post-solve steps are the same: apply the computed impulses.
        wp.launch(
            kernel=compute_JT_delta_lambda_kernel,
            dim=self.con_dim,
            inputs=[
                self._constraint_body_idx,
                self._J_values,
                self._delta_lambda,
            ],
            outputs=[self._JT_delta_lambda],
        )
        wp.launch(
            kernel=compute_delta_body_qd_kernel,
            dim=self.dyn_dim,
            inputs=[
                self.gen_inv_mass,
                self._JT_delta_lambda,
                self._g_v,
            ],
            outputs=[self._delta_body_qd_v],
        )

    def fill_residual_buffer(self):
        wp.launch(
            kernel=linesearch_dynamics_residuals_kernel,
            dim=(self.N_alpha, self.N_b),
            inputs=[
                self.alphas,
                self._delta_body_qd_v,
                self._body_qd,
                self._body_qd_prev,
                self._body_f,
                self.body_mass,
                self.body_inertia,
                self._dt,
                self.gravity,
            ],
            outputs=[self._res_buffer],
            device=self.device,
        )
        wp.launch(
            kernel=linesearch_contact_residuals_kernel,
            dim=(self.N_alpha, self.N_c),
            inputs=[
                self.alphas,
                self._delta_lambda,
                self._delta_body_qd_v,
                self._body_qd,
                self._body_qd_prev,
                self._contact_gap,
                self._J_contact_a,
                self._J_contact_b,
                self._contact_body_a,
                self._contact_body_b,
                self._contact_restitution_coeff,
                # Velocity impulse variables
                self.lambda_n_offset,
                self._lambda,
                # Parameters
                self._dt,
                self.contact_stabilization_factor,
                self.contact_fb_alpha,
                self.contact_fb_beta,
            ],
            outputs=[self._res_buffer],
        )
        wp.launch(
            kernel=linesearch_joint_residuals_kernel,
            dim=(self.N_alpha, self.N_j),
            inputs=[
                self.alphas,
                self._delta_lambda,
                self._delta_body_qd_v,
                self._body_q,
                self._body_qd,
                self.body_com,
                self.joint_type,
                self.joint_enabled,
                self.joint_parent,
                self.joint_child,
                self.joint_X_p,
                self.joint_X_c,
                self.joint_axis_start,
                self.joint_axis_dim,
                self.joint_axis,
                self.joint_linear_compliance,
                self.joint_angular_compliance,
                # Velocity impulse variables
                self.lambda_j_offset,
                self._lambda,
                # Parameters
                self._dt,
                self.joint_stabilization_factor,
            ],
            outputs=[self._res_buffer],
        )

        wp.launch(
            kernel=linesearch_friction_residuals_kernel,
            dim=(self.N_alpha, self.N_c),
            inputs=[
                self.alphas,
                self._delta_lambda,
                self._delta_body_qd_v,
                self._body_qd,
                self._contact_gap,
                self._J_contact_a,
                self._J_contact_b,
                self._contact_body_a,
                self._contact_body_b,
                self._contact_friction_coeff,
                # Velocity impulse variables
                self.lambda_n_offset,
                self.lambda_f_offset,
                self._lambda,
                # Parameters
                self.friction_fb_alpha,
                self.friction_fb_beta,
            ],
            outputs=[self._res_buffer],
        )

    def update_variables(self):
        wp.launch(
            kernel=buffer_to_sq_norm_kernel,
            dim=(self.N_alpha, RES_BUFFER_DIM),
            inputs=[self._res_buffer_v],
            outputs=[self._res_norm_sq],
        )
        wp.launch(
            kernel=update_best_alpha_idx,
            dim=1,
            inputs=[self._res_norm_sq_v],
            outputs=[self._best_alpha_idx],
        )
        wp.launch(
            kernel=update_variables_kernel,
            dim=self.N_b + self.con_dim,
            inputs=[
                self._best_alpha_idx,
                self.alphas,
                self._delta_body_qd_v,
                self._delta_lambda,
            ],
            outputs=[self._body_qd, self._lambda],
        )

    def solve_newton_linesearch(self):
        for i in range(self.newton_iters):
            wp.copy(dest=self._lambda_prev, src=self._lambda)
            self.update_system_values()
            self.solve_linear_system()

            self._res_buffer.zero_()
            self._res_norm_sq.zero_()

            self.fill_residual_buffer()
            self.update_variables()

    def solve_newton(self):
        for i in range(self.newton_iters):
            wp.copy(dest=self._lambda_prev, src=self._lambda)
            self.update_system_values()

            # if self.logger:
            #     with self.logger.scope(f"newton_{i:02d}"):
            #         self.log_newton_state()
            #         self.solve_linear_system()
            # else:
            self.solve_linear_system()

            # Add the changes to the state variables.
            add_inplace(self._body_qd, self._delta_body_qd, 0, 0, self.N_b)
            add_inplace(self._lambda, self._delta_lambda, 0, 0, self.con_dim)
