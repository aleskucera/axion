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

from .constraint_dimensions import ConstraintDimensions
from .control import apply_control
from .engine_config import EngineConfig
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
        self.config = config if config is not None else EngineConfig()

        self.logger = logger

        # Basic model properties
        self.device = model.device
        self.N_b = model.body_count
        self.N_c = model.rigid_contact_max
        self.N_j = model.joint_count
        self.N_alpha = len(self.config.linesearch_alphas)

        with wp.ScopedDevice(self.device):
            self.alphas = wp.array(self.config.linesearch_alphas, dtype=wp.float32)

        self.dims = ConstraintDimensions(self.N_b, self.N_c, self.N_j)
        self._initialize_state_vectors(self.dims)

        # Store model properties
        self._store_model_properties(model)

        # Initialize system operators
        self._initialize_system_operators()

    def _initialize_state_vectors(self, dims: ConstraintDimensions):
        def _zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        def slice_if(cond, arr, sl):
            return arr[sl] if cond else None

        with wp.ScopedDevice(self.device):
            # --- Allocate top-level vectors ---
            self._res = _zeros(dims.res_dim)
            self._J_values = _zeros((dims.con_dim, 2), wp.spatial_vector)
            self._C_values = _zeros(dims.con_dim)
            self._lambda = _zeros(dims.con_dim)
            self._lambda_prev = _zeros(dims.con_dim)
            self._constraint_body_idx = _zeros((dims.con_dim, 2), wp.int32)

            # --- Views into residual ---
            self._g = self._res[: dims.dyn_dim]
            self._g_v = wp.array(self._g, shape=dims.N_b, dtype=wp.spatial_vector)
            self._h = self._res[dims.dyn_dim :]

            # --- Slice-based views ---
            self._h_j, self._h_n, self._h_f = (
                slice_if(dims.N_j > 0, self._h, dims.joint_slice),
                slice_if(dims.N_c > 0, self._h, dims.normal_slice),
                slice_if(dims.N_c > 0, self._h, dims.friction_slice),
            )

            self._J_j_values, self._J_n_values, self._J_f_values = (
                slice_if(dims.N_j > 0, self._J_values, dims.joint_slice),
                slice_if(dims.N_c > 0, self._J_values, dims.normal_slice),
                slice_if(dims.N_c > 0, self._J_values, dims.friction_slice),
            )

            self._C_j_values, self._C_n_values, self._C_f_values = (
                slice_if(dims.N_j > 0, self._C_values, dims.joint_slice),
                slice_if(dims.N_c > 0, self._C_values, dims.normal_slice),
                slice_if(dims.N_c > 0, self._C_values, dims.friction_slice),
            )

            self._lambda_j, self._lambda_n, self._lambda_f = (
                slice_if(dims.N_j > 0, self._lambda, dims.joint_slice),
                slice_if(dims.N_c > 0, self._lambda, dims.normal_slice),
                slice_if(dims.N_c > 0, self._lambda, dims.friction_slice),
            )

            self._lambda_j_prev, self._lambda_n_prev, self._lambda_f_prev = (
                slice_if(dims.N_j > 0, self._lambda_prev, dims.joint_slice),
                slice_if(dims.N_c > 0, self._lambda_prev, dims.normal_slice),
                slice_if(dims.N_c > 0, self._lambda_prev, dims.friction_slice),
            )

            # --- Other working vectors ---
            self._JT_delta_lambda = _zeros(dims.N_b, wp.spatial_vector)
            self._delta_body_qd = _zeros(dims.dyn_dim)
            self._delta_body_qd_v = wp.array(
                self._delta_body_qd, shape=dims.N_b, dtype=wp.spatial_vector
            )
            self._delta_lambda = _zeros(dims.con_dim)
            self._b = _zeros(dims.con_dim)

            # --- Structures for dynamics ---
            self.gen_mass = wp.empty(self.N_b, dtype=GeneralizedMass)
            self.gen_inv_mass = wp.empty(self.N_b, dtype=GeneralizedMass)
            self._joint_interaction = wp.empty(self.N_j, dtype=JointInteraction)
            self._contact_interaction = wp.empty(self.N_c, dtype=ContactInteraction)

            # --- Dense matrices for logging (if enabled) ---
            if self.logger:
                self.Hinv_dense = _zeros((dims.dyn_dim, dims.dyn_dim))
                self.J_dense = _zeros((dims.con_dim, dims.dyn_dim))
                self.C_dense = _zeros((dims.con_dim, dims.con_dim))

    def _initialize_system_operators(self):
        """Initialize system matrix operators and preconditioner."""
        if self.config.matrixfree_representation:
            self.A_op = MatrixFreeSystemOperator(self)
        else:
            self.A_op = MatrixSystemOperator(self)

        self.preconditioner = JacobiPreconditioner(self)

    def _store_model_properties(self, model: Model):
        """Store references to model properties that don't change during simulation."""
        # Body properties
        self.body_com = model.body_com
        self.body_mass = model.body_mass
        self.body_inv_mass = model.body_inv_mass
        self.body_inertia = model.body_inertia
        self.body_inv_inertia = model.body_inv_inertia

        # Shape properties
        self.shape_geo = model.shape_geo
        self.shape_body = model.shape_body
        self.shape_materials = model.shape_materials

        # Joint properties
        self.joint_type = model.joint_type
        self.joint_enabled = model.joint_enabled
        self.joint_parent = model.joint_parent
        self.joint_child = model.joint_child
        self.joint_X_p = model.joint_X_p
        self.joint_X_c = model.joint_X_c
        self.joint_axis_start = model.joint_axis_start
        self.joint_axis_dim = model.joint_axis_dim
        self.joint_axis = model.joint_axis
        self.joint_linear_compliance = model.joint_linear_compliance
        self.joint_angular_compliance = model.joint_angular_compliance

        # Physics parameters
        self.gravity = model.gravity

        # Dynamic state references (will be updated each timestep)
        self._dt = 1e-3

        with wp.ScopedDevice(self.device):
            self.gen_mass = wp.empty(self.dims.N_b, dtype=GeneralizedMass)
            self.gen_inv_mass = wp.empty(self.dims.N_b, dtype=GeneralizedMass)
            self._joint_interaction = wp.empty(self.dims.N_j, dtype=JointInteraction)
            self._contact_interaction = wp.empty(
                self.dims.N_c, dtype=ContactInteraction
            )

        wp.launch(
            kernel=generalized_mass_kernel,
            dim=self.dims.N_b,
            inputs=[
                self.body_mass,
                self.body_inertia,
            ],
            outputs=[self.gen_mass],
            device=self.device,
        )

        wp.launch(
            kernel=generalized_mass_kernel,
            dim=self.dims.N_b,
            inputs=[
                self.body_inv_mass,
                self.body_inv_inertia,
            ],
            outputs=[self.gen_inv_mass],
            device=self.device,
        )

    def _clear_values(self):
        self._g.zero_()
        self._h.zero_()
        self._J_values.zero_()
        self._C_values.zero_()
        self._JT_delta_lambda.zero_()

        self._b.zero_()

    def update_state_variables(
        self, model: Model, state_in: State, state_out: State, dt: float
    ):
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
                self.body_com,
                self.shape_body,
                self.shape_geo,
                self.shape_materials,
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
                self.body_com,
                self.joint_type,
                self.joint_enabled,
                self.joint_parent,
                self.joint_child,
                self.joint_X_p,
                self.joint_X_c,
                self.joint_axis_start,
                self.joint_axis,
                self.joint_linear_compliance,
                self.joint_angular_compliance,
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
                self.shape_body,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                self.joint_parent,
                self.joint_child,
                self.N_j,
                self.N_c,
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
                self.gravity,
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
                self._contact_interaction,
                self._lambda_n,
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
                self._contact_interaction,
                self._lambda_f,
                self._lambda_n_prev,
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
        elif solver == "scipy":
            self.solve_scipy()
        else:
            raise ValueError(f"Invalid solver '{solver}'. Must be 'newton' or 'scipy'.")

        wp.copy(dest=state_out.body_qd, src=self._body_qd)
        wp.copy(dest=state_out.body_q, src=self._body_q)
