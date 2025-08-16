from typing import Literal
from typing import Optional

import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.constraints import contact_kinematics_kernel
from axion.constraints import frictional_constraint_kernel
from axion.constraints import joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.optim import JacobiPreconditioner
from axion.optim import MatrixFreeSystemOperator
from axion.optim import MatrixSystemOperator
from axion.utils import HDF5Logger
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

from .control import apply_control
from .logging_utils import LoggingMixin
from .newton_solver import MAX_BODIES
from .newton_solver import NewtonSolverMixin
from .newton_solver import RES_BUFFER_DIM
from .scipy_solver import ScipySolverMixin
from .utils import update_system_rhs_kernel


@wp.kernel
def update_constraint_body_idx_kernel(
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    # --- Parameters ---
    joint_count: wp.uint32,
    max_contact_count: wp.uint32,
    # --- Outputs ---
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
):
    constraint_idx = wp.tid()
    nj = wp.int32(joint_count)
    nc = wp.int32(max_contact_count)

    body_a = -1
    body_b = -1

    if constraint_idx < 5 * nj:
        joint_index = constraint_idx // 5
        body_a = joint_parent[joint_index]
        body_b = joint_child[joint_index]
    elif constraint_idx < 5 * nj + nc:
        offset = 5 * nj
        contact_index = (constraint_idx - offset) // 1
        body_a = contact_body_a[contact_index]
        body_b = contact_body_b[contact_index]
    else:
        offset = 5 * nj + nc
        contact_index = (constraint_idx - offset) // 2
        body_a = contact_body_a[contact_index]
        body_b = contact_body_b[contact_index]

    constraint_body_idx[constraint_idx, 0] = body_a
    constraint_body_idx[constraint_idx, 1] = body_b


class AxionEngine(Integrator, LoggingMixin, NewtonSolverMixin, ScipySolverMixin):
    def __init__(
        self,
        model: Model,
        newton_iters: int = 4,
        linear_iters: int = 4,
        joint_stabilization_factor: float = 0.01,
        contact_stabilization_factor: float = 0.1,
        contact_fb_alpha: float = 0.25,
        contact_fb_beta: float = 0.25,
        friction_fb_alpha: float = 0.25,
        friction_fb_beta: float = 0.25,
        matrixfree_representation: float = True,
        linesearch_alphas: tuple = (2.0, 1.0, 0.5, 0.25, 0.125),
        logger: Optional[HDF5Logger] = None,
    ):
        super().__init__()
        self.newton_iters = newton_iters
        self.linear_iters = linear_iters

        self.joint_stabilization_factor = joint_stabilization_factor
        self.contact_stabilization_factor = contact_stabilization_factor

        self.contact_fb_alpha = contact_fb_alpha
        self.contact_fb_beta = contact_fb_beta
        self.friction_fb_alpha = friction_fb_alpha
        self.friction_fb_beta = friction_fb_beta

        self.matrixfree_representation = matrixfree_representation

        self.logger = logger

        # Basic model properties
        self.device = model.device
        self.N_b = model.body_count
        self.N_c = model.rigid_contact_max
        self.N_j = model.joint_count
        self.N_alpha = len(linesearch_alphas)

        with wp.ScopedDevice(self.device):
            self.alphas = wp.array(linesearch_alphas, dtype=wp.float32)

        # Constraint dimensions
        self._setup_constraint_dimensions()

        # Initialize state vectors
        self._initialize_state_vectors()

        # Store model properties
        self._store_model_properties(model)

        # Initialize system operators
        self._initialize_system_operators()

    def _setup_constraint_dimensions(self):
        """Setup constraint dimensions and offsets."""
        nj = 5 * self.N_j  # joint constraints
        nn = self.N_c  # normal constraints
        nf = 2 * self.N_c  # friction constraints

        self.dyn_dim = self.N_b * 6  # Dynamic dimensions
        self.con_dim = nj + nn + nf  # Constraint dimensions
        self.res_dim = self.dyn_dim + self.con_dim

        assert (
            self.res_dim > self.dyn_dim
        ), "Exceeded number of maximum bodies in the scene"

        # Setup offsets for different constraint types
        self.h_j_offset = 0
        self.h_n_offset = nj
        self.h_f_offset = nj + nn

        self.J_j_offset = 0
        self.J_n_offset = nj
        self.J_f_offset = nj + nn

        self.C_j_offset = 0
        self.C_n_offset = nj
        self.C_f_offset = nj + nn

        self.lambda_j_offset = 0
        self.lambda_n_offset = nj
        self.lambda_f_offset = nj + nn

    def _initialize_state_vectors(self):
        """Initialize all state and working vectors."""

        def _zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype)

        with wp.ScopedDevice(self.device):
            self._res = _zeros(self.res_dim)

            # Residual vectors
            self._g = self._res[: self.dyn_dim]
            self._g_v = wp.array(self._g, shape=self.N_b, dtype=wp.spatial_vector)

            self._h = self._res[self.dyn_dim :]
            self._h_j = self._h[0 : self.h_n_offset] if self.N_j > 0 else None
            self._h_n = (
                self._h[self.h_n_offset : self.h_f_offset] if self.N_c > 0 else None
            )
            self._h_f = self._h[self.h_f_offset :] if self.N_c > 0 else None

            # System matrix components
            self._J_values = _zeros((self.con_dim, 2), wp.spatial_vector)
            self._J_j_values = (
                self._J_values[: self.J_n_offset] if self.N_j > 0 else None
            )
            self._J_n_values = (
                self._J_values[self.J_n_offset : self.J_f_offset]
                if self.N_c > 0
                else None
            )
            self._J_f_values = (
                self._J_values[self.J_f_offset :] if self.N_c > 0 else None
            )

            self._C_values = _zeros(self.con_dim)
            self._C_j_values = (
                self._C_values[: self.C_n_offset] if self.N_j > 0 else None
            )
            self._C_n_values = (
                self._C_values[self.C_n_offset : self.C_f_offset]
                if self.N_c > 0
                else None
            )
            self._C_f_values = (
                self._C_values[self.C_f_offset :] if self.N_c > 0 else None
            )

            # Working vectors
            self._JT_delta_lambda = _zeros(self.N_b, wp.spatial_vector)

            self._lambda = _zeros(self.con_dim)
            self._lambda_j = (
                self._lambda[: self.lambda_n_offset] if self.N_j > 0 else None
            )
            self._lambda_n = (
                self._lambda[self.lambda_n_offset : self.lambda_f_offset]
                if self.N_c > 0
                else None
            )
            self._lambda_f = (
                self._lambda[self.lambda_f_offset :] if self.N_c > 0 else None
            )

            self._lambda_prev = _zeros(self.con_dim)
            self._lambda_j_prev = (
                self._lambda_prev[: self.lambda_n_offset] if self.N_j > 0 else None
            )
            self._lambda_n_prev = (
                self._lambda_prev[self.lambda_n_offset : self.lambda_f_offset]
                if self.N_c > 0
                else None
            )
            self._lambda_f_prev = (
                self._lambda_prev[self.lambda_f_offset :] if self.N_c > 0 else None
            )

            self._delta_body_qd = _zeros(self.dyn_dim)
            self._delta_body_qd_v = wp.array(
                self._delta_body_qd, shape=self.N_b, dtype=wp.spatial_vector
            )
            self._delta_lambda = _zeros(self.con_dim)
            self._b = _zeros(self.con_dim)

            # Contact-specific arrays
            self._contact_gap = _zeros(self.N_c)
            self._J_contact_a = _zeros((self.N_c, 3), wp.spatial_vector)
            self._J_contact_b = _zeros((self.N_c, 3), wp.spatial_vector)
            self._contact_body_a = _zeros(self.N_c, wp.int32)
            self._contact_body_b = _zeros(self.N_c, wp.int32)
            self._contact_restitution_coeff = _zeros(self.N_c)
            self._contact_friction_coeff = _zeros(self.N_c)

            self._constraint_body_idx = _zeros((self.con_dim, 2), wp.int32)

            # Dense matrices for logging
            if self.logger:
                self.Hinv_dense = _zeros((self.dyn_dim, self.dyn_dim))
                self.J_dense = _zeros((self.con_dim, self.dyn_dim))
                self.C_dense = _zeros((self.con_dim, self.con_dim))

            self._res_buffer = _zeros((self.N_alpha, RES_BUFFER_DIM))
            self._res_buffer_v = wp.array(
                self._res_buffer,
                shape=self.N_alpha,
                dtype=wp.types.vector(length=RES_BUFFER_DIM, dtype=wp.float32),
            )  # Zero-copy of the self._res_buffer array

            self._res_norm_sq = _zeros((1, self.N_alpha))
            self._res_norm_sq_v = wp.array(
                self._res_norm_sq,
                shape=1,
                dtype=wp.types.vector(length=self.N_alpha, dtype=wp.float32),
            )

            self._best_alpha_idx = _zeros((1,), wp.uint32)

    def _initialize_system_operators(self):
        """Initialize system matrix operators and preconditioner."""
        if self.matrixfree_representation:
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
        self._rigid_contact_count = model.rigid_contact_count
        self._rigid_contact_shape0 = model.rigid_contact_shape0
        self._rigid_contact_shape1 = model.rigid_contact_shape1
        self._rigid_contact_normal = model.rigid_contact_normal
        self._rigid_contact_point0 = model.rigid_contact_point0
        self._rigid_contact_point1 = model.rigid_contact_point1

        wp.launch(
            kernel=contact_kinematics_kernel,
            dim=self.N_c,
            inputs=[
                self._body_q,
                self.body_com,
                self.shape_body,
                self.shape_geo,
                self.shape_materials,
                self._rigid_contact_count,
                self._rigid_contact_point0,
                self._rigid_contact_point1,
                self._rigid_contact_normal,
                self._rigid_contact_shape0,
                self._rigid_contact_shape1,
            ],
            outputs=[
                self._contact_gap,
                self._J_contact_a,
                self._J_contact_b,
                self._contact_body_a,
                self._contact_body_b,
                self._contact_restitution_coeff,
                self._contact_friction_coeff,
            ],
        )

        wp.launch(
            kernel=update_constraint_body_idx_kernel,
            dim=self.con_dim,
            inputs=[
                self._contact_body_a,
                self._contact_body_b,
                self.joint_parent,
                self.joint_child,
                self.N_j,
                self.N_c,
            ],
            outputs=[
                self._constraint_body_idx,
            ],
        )

    def update_system_values(self):
        self._clear_values()

        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=self.N_b,
            inputs=[
                self._body_qd,
                self._body_qd_prev,
                self._body_f,
                self.body_mass,
                self.body_inertia,
                self._dt,
                self.gravity,
            ],
            outputs=[self._g_v],
            device=self.device,
        )
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=self.N_c,
            inputs=[
                self._body_qd,
                self._body_qd_prev,
                self._contact_gap,
                self._J_contact_a,
                self._J_contact_b,
                self._contact_body_a,
                self._contact_body_b,
                self._contact_restitution_coeff,
                # Velocity impulse variables
                self._lambda_n,
                # Parameters
                self._dt,
                self.contact_stabilization_factor,
                self.contact_fb_alpha,
                self.contact_fb_beta,
            ],
            outputs=[
                self._g_v,
                self._h_n,
                self._J_n_values,
                self._C_n_values,
            ],
        )
        wp.launch(
            kernel=joint_constraint_kernel,
            dim=self.N_j,
            inputs=[
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
                self._lambda_j,
                # Parameters
                self._dt,
                self.joint_stabilization_factor,
            ],
            outputs=[
                self._g_v,
                self._h_j,
                self._J_j_values,
                self._C_j_values,
            ],
        )

        wp.launch(
            kernel=frictional_constraint_kernel,
            dim=self.N_c,
            inputs=[
                self._body_qd,
                self._contact_gap,
                self._J_contact_a,
                self._J_contact_b,
                self._contact_body_a,
                self._contact_body_b,
                self._contact_friction_coeff,
                # Velocity impulse variables
                self._lambda_f,
                self._lambda_n_prev,
                # Parameters
                self.friction_fb_alpha,
                self.friction_fb_beta,
            ],
            outputs=[
                self._g_v,
                self._h_f,
                self._J_f_values,
                self._C_f_values,
            ],
        )

        if not self.matrixfree_representation:
            self.A_op.update()

        self.preconditioner.update()  # Update the preconditioner with new values

        wp.launch(
            kernel=update_system_rhs_kernel,
            dim=(self.con_dim,),
            inputs=[
                self.body_inv_mass,
                self.body_inv_inertia,
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
            contact_count = self._rigid_contact_count.numpy().item()
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
