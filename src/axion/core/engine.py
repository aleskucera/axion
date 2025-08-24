from typing import Literal
from typing import Optional

import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.constraints import friction_constraint_kernel
from axion.constraints import joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.constraints import update_constraint_body_idx_kernel
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from axion.types import contact_interaction_kernel
from axion.types import ContactInteraction
from axion.types import generalized_mass_kernel
from axion.types import GeneralizedMass
from axion.types import joint_interaction_kernel
from axion.types import JointInteraction
from axion.utils import HDF5Logger
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

from .control import apply_control
from .engine_config import EngineConfig
from .engine_data import create_engine_arrays
from .engine_dims import EngineDimensions
from .logging_utils import LoggingMixin
from .newton_solver import NewtonSolverMixin
from .scipy_solver import ScipySolverMixin
from .utils import update_system_rhs_kernel


class AxionEngine(Integrator, LoggingMixin, NewtonSolverMixin, ScipySolverMixin):
    def __init__(
        self,
        model: Model,
        config: Optional[EngineConfig] = None,
        logger: Optional[HDF5Logger] = None,
    ):
        super().__init__()
        self.device = model.device

        self.model = model
        self.logger = logger
        self.config = config if config is not None else EngineConfig()

        self.dims = EngineDimensions(
            N_b=self.model.body_count,
            N_c=self.model.rigid_contact_max,
            N_j=self.model.joint_count,
            N_alpha=self.config.linesearch_steps,
        )

        alphas = self.config.linesearch_alphas
        self.data = create_engine_arrays(self.dims, self.device, alphas)

        def _zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype)

        def slice_if(cond, arr, sl):
            return arr[sl] if cond else None

        def slice_if_2D(cond, arr, sl):
            return arr[:, sl] if cond else None

        with wp.ScopedDevice(self.device):
            # --- Allocate top-level vectors ---
            self._res = _zeros(self.dims.res_dim)
            self._J_values = _zeros((self.dims.con_dim, 2), wp.spatial_vector)
            self._C_values = _zeros(self.dims.con_dim)
            self._lambda = _zeros(self.dims.con_dim)
            self._lambda_prev = _zeros(self.dims.con_dim)
            self._constraint_body_idx = _zeros((self.dims.con_dim, 2), wp.int32)

            # --- Views into residual ---
            self._g = self._res[: self.dims.dyn_dim]
            self._g_v = wp.array(self._g, shape=self.dims.N_b, dtype=wp.spatial_vector)
            self._h = self._res[self.dims.dyn_dim :]

            # --- Slice-based views ---
            self._h_j, self._h_n, self._h_f = (
                slice_if(self.dims.N_j > 0, self._h, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._h, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._h, self.dims.friction_slice),
            )

            self._J_j_values, self._J_n_values, self._J_f_values = (
                slice_if(self.dims.N_j > 0, self._J_values, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._J_values, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._J_values, self.dims.friction_slice),
            )

            self._C_j_values, self._C_n_values, self._C_f_values = (
                slice_if(self.dims.N_j > 0, self._C_values, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._C_values, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._C_values, self.dims.friction_slice),
            )

            self._lambda_j, self._lambda_n, self._lambda_f = (
                slice_if(self.dims.N_j > 0, self._lambda, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._lambda, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._lambda, self.dims.friction_slice),
            )

            self._lambda_j_prev, self._lambda_n_prev, self._lambda_f_prev = (
                slice_if(self.dims.N_j > 0, self._lambda_prev, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._lambda_prev, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._lambda_prev, self.dims.friction_slice),
            )

            # --- Other working vectors ---
            self._JT_delta_lambda = _zeros(self.dims.N_b, wp.spatial_vector)
            self._delta_body_qd = _zeros(self.dims.dyn_dim)
            self._delta_body_qd_v = wp.array(
                self._delta_body_qd, shape=self.dims.N_b, dtype=wp.spatial_vector
            )

            self._delta_lambda = _zeros(self.dims.con_dim)
            self._delta_lambda_j, self._delta_lambda_n, self._delta_lambda_f = (
                slice_if(self.dims.N_j > 0, self._delta_lambda, self.dims.joint_slice),
                slice_if(self.dims.N_c > 0, self._delta_lambda, self.dims.normal_slice),
                slice_if(self.dims.N_c > 0, self._delta_lambda, self.dims.friction_slice),
            )

            self._b = _zeros(self.dims.con_dim)

            # --- Structures for dynamics ---
            self.gen_mass = wp.empty(self.dims.N_b, dtype=GeneralizedMass)
            self.gen_inv_mass = wp.empty(self.dims.N_b, dtype=GeneralizedMass)

            wp.launch(
                kernel=generalized_mass_kernel,
                dim=self.dims.N_b,
                inputs=[
                    self.model.body_mass,
                    self.model.body_inertia,
                ],
                outputs=[self.gen_mass],
            )

            wp.launch(
                kernel=generalized_mass_kernel,
                dim=self.dims.N_b,
                inputs=[
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                ],
                outputs=[self.gen_inv_mass],
            )

            self._joint_interaction = wp.empty(self.dims.N_j, dtype=JointInteraction)
            self._contact_interaction = wp.empty(self.dims.N_c, dtype=ContactInteraction)

            # --- Dense matrices for logging (if enabled) ---
            if self.logger:
                self.Hinv_dense = _zeros((self.dims.dyn_dim, self.dims.dyn_dim))
                self.J_dense = _zeros((self.dims.con_dim, self.dims.dyn_dim))
                self.C_dense = _zeros((self.dims.con_dim, self.dims.con_dim))

            self.alphas = wp.array(self.config.linesearch_alphas, dtype=wp.float32)
            self._res_alpha = _zeros((self.dims.N_alpha, self.dims.res_dim))

            self._g_alpha = self._res_alpha[:, : self.dims.dyn_dim]
            self._g_alpha_v = wp.array(
                self._g_alpha,
                shape=(self.dims.N_alpha, self.dims.N_b),
                dtype=wp.spatial_vector,
            )
            self._h_alpha = self._res_alpha[:, self.dims.dyn_dim :]

            self._h_alpha_j, self._h_alpha_n, self._h_alpha_f = (
                slice_if_2D(self.dims.N_j > 0, self._h_alpha, self.dims.joint_slice),
                slice_if_2D(self.dims.N_c > 0, self._h_alpha, self.dims.normal_slice),
                slice_if_2D(self.dims.N_c > 0, self._h_alpha, self.dims.friction_slice),
            )

            self._res_alpha_norm_sq = _zeros(self.dims.N_alpha)
            self._best_alpha_idx = _zeros(1, wp.uint32)

        if self.config.matrixfree_representation:
            self.A_op = MatrixFreeSystemOperator(self)
        else:
            self.A_op = MatrixSystemOperator(self)

        self.preconditioner = JacobiPreconditioner(self)

    def _clear_values(self):
        self._g.zero_()
        self._h.zero_()
        self._J_values.zero_()
        self._C_values.zero_()
        self._JT_delta_lambda.zero_()

        self._b.zero_()

    def update_state_variables(self, model: Model, state_in: State, state_out: State, dt: float):
        self._dt = dt
        self._body_q = state_out.body_q
        self._body_qd = state_out.body_qd
        self._body_qd_prev = state_in.body_qd
        self._body_f = state_in.body_f

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=self.dims.N_c,
            inputs=[
                self._body_q,
                self.model.body_com,
                self.model.shape_body,
                self.model.shape_geo,
                self.model.shape_materials,
                model.rigid_contact_count,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
            ],
            outputs=[
                self._contact_interaction,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=joint_interaction_kernel,
            dim=self.dims.N_j,
            inputs=[
                self._body_q,
                self.model.body_com,
                self.model.joint_type,
                self.model.joint_enabled,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis_start,
                self.model.joint_axis,
                self.model.joint_linear_compliance,
                self.model.joint_angular_compliance,
            ],
            outputs=[
                self._joint_interaction,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=update_constraint_body_idx_kernel,
            dim=self.dims.con_dim,
            inputs=[
                self.model.shape_body,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                self.model.joint_parent,
                self.model.joint_child,
                self.dims.N_j,
                self.dims.N_c,
            ],
            outputs=[
                self._constraint_body_idx,
            ],
            device=self.device,
        )

    def update_system_values(self):
        self._clear_values()

        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=self.dims.N_b,
            inputs=[
                self._body_qd,
                self._body_qd_prev,
                self._body_f,
                self.gen_mass,
                self._dt,
                self.model.gravity,
            ],
            outputs=[self._g_v],
            device=self.device,
        )

        wp.launch(
            kernel=joint_constraint_kernel,
            dim=(5, self.dims.N_j),
            inputs=[
                self._body_qd,
                self._lambda_j,
                self._joint_interaction,
                self._dt,
                self.config.joint_stabilization_factor,
            ],
            outputs=[
                self._g_v,
                self._h_j,
                self._J_j_values,
                self._C_j_values,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=contact_constraint_kernel,
            dim=self.dims.N_c,
            inputs=[
                self._body_qd,
                self._body_qd_prev,
                self._lambda_n,
                self._contact_interaction,
                self._dt,
                self.config.contact_stabilization_factor,
                self.config.contact_fb_alpha,
                self.config.contact_fb_beta,
                self.config.contact_compliance,
            ],
            outputs=[
                self._g_v,
                self._h_n,
                self._J_n_values,
                self._C_n_values,
            ],
        )

        wp.launch(
            kernel=friction_constraint_kernel,
            dim=self.dims.N_c,
            inputs=[
                self._body_qd,
                self._lambda_f,
                self._lambda_n_prev,
                self._contact_interaction,
                self.config.friction_fb_alpha,
                self.config.friction_fb_beta,
                self.config.friction_compliance,
            ],
            outputs=[
                self._g_v,
                self._h_f,
                self._J_f_values,
                self._C_f_values,
            ],
        )

        if not self.config.matrixfree_representation:
            self.A_op.update()

        self.preconditioner.update()

        wp.launch(
            kernel=update_system_rhs_kernel,
            dim=(self.dims.con_dim,),
            inputs=[
                self.gen_inv_mass,
                self._constraint_body_idx,
                self._J_values,
                self._g_v,
                self._h,
            ],
            outputs=[self._b],
        )

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
        solver: Literal["newton", "scipy"] = "newton",
    ):
        apply_control(model, state_in, state_out, dt, control)
        self.integrate_bodies(model, state_in, state_out, dt)
        self.update_state_variables(model, state_in, state_out, dt)

        if self.logger:
            contact_count = model.rigid_contact_count.numpy().item()
            self.logger.log_scalar("contact_count", contact_count)

        # TODO: Check the warm startup
        # self._lambda.zero_()

        if solver == "newton":
            self.solve_newton()
        elif solver == "newton_linesearch":
            self.solve_newton_linesearch()
        elif solver == "scipy":
            self.solve_scipy()
        else:
            raise ValueError(f"Invalid solver '{solver}'. Must be 'newton' or 'scipy'.")

        wp.copy(dest=state_out.body_qd, src=self._body_qd)
        wp.copy(dest=state_out.body_q, src=self._body_q)
