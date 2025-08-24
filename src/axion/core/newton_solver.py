import warp as wp
from axion.constraints import linesearch_contact_residuals_kernel
from axion.constraints import linesearch_dynamics_residuals_kernel
from axion.constraints import linesearch_friction_residuals_kernel
from axion.constraints import linesearch_joint_residuals_kernel
from axion.optim.cr import cr_solver
from axion.types import *
from axion.types import GeneralizedMass
from axion.utils import add_inplace


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

    delta_body_qd[body_idx] = gen_inv_mass[body_idx] * (JT_delta_lambda[body_idx] - g[body_idx])


@wp.kernel
def update_sq_norm(
    res_alpha: wp.array(dtype=wp.float32, ndim=2),
    res_alpha_norm_sq: wp.array(dtype=wp.float32),
):
    alpha_idx = wp.tid()

    norm_sq = float(0.0)
    for i in range(res_alpha.shape[1]):
        norm_sq += wp.pow(res_alpha[alpha_idx, i], 2.0)

    res_alpha_norm_sq[alpha_idx] = norm_sq


@wp.kernel
def update_best_alpha_idx(
    res_alpha_norm_sq: wp.array(dtype=wp.float32),
    best_alpha_idx: wp.array(dtype=wp.uint32),
):
    tid = wp.tid()
    if tid > 0:
        return

    # make them dynamic (mutable) locals:
    best_idx = wp.uint32(0)  # not: best_idx = 0
    min_value = wp.float32(3.4e38)  # Largest finite float32 value

    for i in range(res_alpha_norm_sq.shape[0]):
        value = res_alpha_norm_sq[i]
        if value < min_value:
            best_idx = wp.uint32(i)  # keep dtype consistent
            min_value = value

    best_alpha_idx[0] = best_idx


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
            iters=self.config.linear_iters,
            preconditioner=self.preconditioner,
            logger=self.logger,
        )

        # The post-solve steps are the same: apply the computed impulses.
        wp.launch(
            kernel=compute_JT_delta_lambda_kernel,
            dim=self.dims.con_dim,
            inputs=[
                self._constraint_body_idx,
                self._J_values,
                self._delta_lambda,
            ],
            outputs=[self._JT_delta_lambda],
        )
        wp.launch(
            kernel=compute_delta_body_qd_kernel,
            dim=self.dims.dyn_dim,
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
            dim=(self.dims.N_alpha, self.dims.N_b),
            inputs=[
                self.alphas,
                self._delta_body_qd_v,
                # ---
                self._body_qd,
                self._body_qd_prev,
                self._body_f,
                self.gen_mass,
                self._dt,
                self.model.gravity,
            ],
            outputs=[self._g_alpha_v],
            device=self.device,
        )

        wp.launch(
            kernel=linesearch_joint_residuals_kernel,
            dim=(self.dims.N_alpha, 5, self.dims.N_j),
            inputs=[
                self.alphas,
                self._delta_body_qd_v,
                self._delta_lambda_j,
                # ---
                self._body_qd,
                self._lambda_j,
                self._joint_interaction,
                # Parameters
                self._dt,
                self.config.joint_stabilization_factor,
            ],
            outputs=[self._g_alpha_v, self._h_alpha_j],
            device=self.device,
        )

        wp.launch(
            kernel=linesearch_contact_residuals_kernel,
            dim=(self.dims.N_alpha, self.dims.N_c),
            inputs=[
                self.alphas,
                self._delta_body_qd_v,
                self._delta_lambda_n,
                # ---
                self._body_qd,
                self._body_qd_prev,
                self._lambda_n,
                self._contact_interaction,
                # Parameters
                self._dt,
                self.config.contact_stabilization_factor,
                self.config.contact_fb_alpha,
                self.config.contact_fb_beta,
                self.config.contact_compliance,
            ],
            outputs=[self._g_alpha_v, self._h_alpha_n],
            device=self.device,
        )

        wp.launch(
            kernel=linesearch_friction_residuals_kernel,
            dim=(self.dims.N_alpha, self.dims.N_c),
            inputs=[
                self.alphas,
                self._delta_body_qd_v,
                self._delta_lambda_f,
                self._delta_lambda_n,
                # ---
                self._body_qd,
                self._lambda_f,
                self._lambda_n,
                self._contact_interaction,
                # Parameters
                self.config.friction_fb_alpha,
                self.config.friction_fb_beta,
                self.config.friction_compliance,
            ],
            outputs=[self._g_alpha_v, self._h_alpha_f],
            device=self.device,
        )

    def update_variables(self):
        wp.launch(
            kernel=update_sq_norm,
            dim=self.dims.N_alpha,
            inputs=[self._res_alpha],
            outputs=[self._res_alpha_norm_sq],
            device=self.device,
        )

        wp.launch(
            kernel=update_best_alpha_idx,
            dim=1,
            inputs=[self._res_alpha_norm_sq],
            outputs=[self._best_alpha_idx],
            device=self.device,
        )

        wp.launch(
            kernel=update_variables_kernel,
            dim=self.dims.N_b + self.dims.con_dim,
            inputs=[
                self._best_alpha_idx,
                self.alphas,
                self._delta_body_qd_v,
                self._delta_lambda,
            ],
            outputs=[self._body_qd, self._lambda],
        )

    def solve_newton_linesearch(self):
        for i in range(self.config.newton_iters):
            wp.copy(dest=self._lambda_prev, src=self._lambda)
            self.update_system_values()
            self.solve_linear_system()

            self._g_alpha.zero_()
            # self._res_norm_sq.zero_()

            self.fill_residual_buffer()
            self.update_variables()

    def solve_newton(self):
        for i in range(self.config.newton_iters):
            wp.copy(dest=self._lambda_prev, src=self._lambda)
            self.update_system_values()

            # if self.logger:
            #     with self.logger.scope(f"newton_{i:02d}"):
            #         self.log_newton_state()
            #         self.solve_linear_system()
            # else:
            self.solve_linear_system()

            # Add the changes to the state variables.
            add_inplace(self._body_qd, self._delta_body_qd, 0, 0, self.dims.N_b)
            add_inplace(self._lambda, self._delta_lambda, 0, 0, self.dims.con_dim)
