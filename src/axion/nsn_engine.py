import warp as wp
from axion.contact_constraint import contact_constraint_kernel
from axion.dynamics_constraint import unconstrained_dynamics_kernel
from axion.frictional_constraint import frictional_constraint_kernel
from axion.joint_constraint import joint_constraint_kernel
from axion.optim import JacobiPreconditioner
from axion.optim.cr import cr_solver
from axion.optim.system_operator import SystemOperator
from axion.utils import add_inplace
from axion.utils import apply_joint_actions_kernel
from axion.utils import contact_kinematics_kernel
from axion.utils import get_constraint_body_index
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

# --- Constants remain the same ---
JOINT_CONSTRAINT_STABILIZATION = 0.01
CONTACT_CONSTRAINT_STABILIZATION = 0.1
CONTACT_FB_ALPHA = 0.25
CONTACT_FB_BETA = 0.25
FRICTION_FB_ALPHA = 0.25
FRICTION_FB_BETA = 0.25


@wp.func
def _compute_JHinvG_i(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    body_idx: int,
    J: wp.spatial_vector,
    g: wp.array(dtype=wp.float32),
):
    # This function is used by update_system_rhs_kernel and remains unchanged
    if body_idx < 0:
        return 0.0
    g_ang_body = wp.vec3(g[body_idx * 6 + 0], g[body_idx * 6 + 1], g[body_idx * 6 + 2])
    Hinv_g_ang = body_inertia_inv[body_idx] @ g_ang_body
    g_lin_body = wp.vec3(g[body_idx * 6 + 3], g[body_idx * 6 + 4], g[body_idx * 6 + 5])
    Hinv_g_lin = body_mass_inv[body_idx] * g_lin_body
    Hinv_g = wp.spatial_vector(Hinv_g_ang, Hinv_g_lin)
    return wp.dot(J, Hinv_g)


@wp.kernel
def update_system_rhs_kernel(
    body_mass_inv: wp.array(dtype=wp.float32),
    body_inertia_inv: wp.array(dtype=wp.mat33),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: wp.int32,
    J_n_offset: wp.int32,
    J_f_offset: wp.int32,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    g: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    # This kernel computes the right-hand-side 'b' and remains unchanged.
    tid = wp.tid()
    body_a, body_b = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        tid,
    )
    J_ia = J_values[tid, 0]
    J_ib = J_values[tid, 1]
    JHinvG = _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_a, J_ia, g)
    JHinvG += _compute_JHinvG_i(body_mass_inv, body_inertia_inv, body_b, J_ib, g)
    b[tid] = JHinvG - h[tid]


@wp.kernel
def compute_JT_delta_lambda_kernel(
    # Constraint layout information
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    contact_body_a: wp.array(dtype=wp.int32),
    contact_body_b: wp.array(dtype=wp.int32),
    J_j_offset: int,
    J_n_offset: int,
    J_f_offset: int,
    # Jacobian and vector data
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    delta_lambda: wp.array(dtype=wp.float32),
    # Output array
    JT_delta_lambda: wp.array(dtype=wp.float32),
):
    constraint_idx = wp.tid()

    body_a, body_b = get_constraint_body_index(
        joint_parent,
        joint_child,
        contact_body_a,
        contact_body_b,
        J_j_offset,
        J_n_offset,
        J_f_offset,
        constraint_idx,
    )
    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]
    dl = delta_lambda[constraint_idx]

    # Compute branchless multipliers (because of bodies that are -1)
    multiplier_a = wp.where(body_a >= 0, 1.0, 0.0)
    multiplier_b = wp.where(body_b >= 0, 1.0, 0.0)

    for i in range(wp.static(6)):
        st_i = wp.static(i)

        # For body_a
        index_a = wp.where(body_a >= 0, body_a * 6 + st_i, st_i)
        wp.atomic_add(JT_delta_lambda, index_a, J_ia[st_i] * dl * multiplier_a)

        # For body_b
        index_b = wp.where(body_b >= 0, body_b * 6 + st_i, st_i)
        wp.atomic_add(JT_delta_lambda, index_b, J_ib[st_i] * dl * multiplier_b)


@wp.kernel
def compute_delta_body_qd_kernel(
    # This kernel is also still needed to update body velocities after the solve
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    JT_delta_lambda: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    delta_body_qd: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    body_idx = tid // 6
    body_dim = tid % 6
    if body_idx >= body_inv_mass.shape[0]:
        return
    if body_dim < 3:
        tmp_ang_x = JT_delta_lambda[body_idx * 6 + 0] - g[body_idx * 6 + 0]
        tmp_ang_y = JT_delta_lambda[body_idx * 6 + 1] - g[body_idx * 6 + 1]
        tmp_ang_z = JT_delta_lambda[body_idx * 6 + 2] - g[body_idx * 6 + 2]
        inertia_inv = body_inv_inertia[body_idx]
        delta_body_qd[tid] = (
            inertia_inv[body_dim, 0] * tmp_ang_x
            + inertia_inv[body_dim, 1] * tmp_ang_y
            + inertia_inv[body_dim, 2] * tmp_ang_z
        )
    else:
        tmp_lin_component = JT_delta_lambda[tid] - g[tid]
        mass_inv = body_inv_mass[body_idx]
        delta_body_qd[tid] = mass_inv * tmp_lin_component


@wp.kernel
def fused_reset_kernel(
    h: wp.array(dtype=wp.float32),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    C_values: wp.array(dtype=wp.float32),
):
    """A single kernel to reset arrays that are built in slices."""
    tid = wp.tid()
    h[tid] = 0.0
    J_values[tid, 0] = wp.spatial_vector()
    J_values[tid, 1] = wp.spatial_vector()
    C_values[tid] = 0.0


class NSNEngine(Integrator):
    def __init__(
        self,
        model: Model,
        newton_iters: int = 4,
        linear_iters: int = 5,
    ):
        super().__init__()
        self.newton_iters = newton_iters
        self.linear_iters = linear_iters

        self.device = model.device
        self.N_b = model.body_count
        self.N_c = model.rigid_contact_max
        self.N_j = model.joint_count

        num_j_constraints = 5 * self.N_j  # Number of joint constraints
        num_n_constraints = self.N_c  # Number of normal contact constraints
        num_f_constraints = 2 * self.N_c  # Number of frictional constraints

        con_dim = (
            num_j_constraints + num_n_constraints + num_f_constraints
        )  # Total number of constraints
        dyn_dim = self.N_b * 6  # Number of dynamics equations (6 per body)

        self.con_dim = con_dim
        self.dyn_dim = dyn_dim

        # ============== RESIDUAL VECTORS ==============
        # Momentum balance vector - this is the residual of the dynamics
        # of the system
        self._g = wp.zeros((dyn_dim,), dtype=wp.float32, device=self.device)

        # The vector of the constraint errors
        self._h = wp.zeros((con_dim,), dtype=wp.float32, device=self.device)
        self.h_j_offset = 0
        self.h_n_offset = num_j_constraints
        self.h_f_offset = num_j_constraints + num_n_constraints

        self._J_values = wp.zeros(
            (con_dim, 2), dtype=wp.spatial_vector, device=self.device
        )  # Non-zero values of the constraint Jacobian
        self.J_j_offset = 0
        self.J_n_offset = num_j_constraints
        self.J_f_offset = num_j_constraints + num_n_constraints

        self._C_values = wp.zeros((con_dim,), dtype=wp.float32, device=self.device)
        self.C_j_offset = 0
        self.C_n_offset = num_j_constraints
        self.C_f_offset = num_j_constraints + num_n_constraints

        self._JT_delta_lambda = wp.zeros(
            (dyn_dim,), dtype=wp.float32, device=self.device
        )
        # Contact impulse vector
        self._lambda = wp.zeros((con_dim,), dtype=wp.float32, device=self.device)
        self._lambda_prev = wp.zeros((con_dim,), dtype=wp.float32, device=self.device)
        self.lambda_j_offset = 0
        self.lambda_n_offset = num_j_constraints
        self.lambda_f_offset = num_j_constraints + num_n_constraints

        # The differences of the variables
        self._delta_body_qd = wp.zeros(dyn_dim, dtype=wp.float32, device=self.device)
        self._delta_lambda = wp.zeros(con_dim, dtype=wp.float32, device=self.device)

        # Model attributes that doesn't change during the simulation
        self.body_com = model.body_com
        self.body_mass = model.body_mass
        self.body_inv_mass = model.body_inv_mass
        self.body_inertia = model.body_inertia
        self.body_inv_inertia = model.body_inv_inertia

        self.shape_geo = model.shape_geo
        self.shape_body = model.shape_body
        self.shape_materials = model.shape_materials

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

        self.gravity = model.gravity

        # Modifiable variables
        self._body_q = model.body_q
        self._body_qd = model.body_qd
        self._body_qd_prev = model.body_qd
        self._body_f = model.body_qd
        self._dt = 1e-3
        self._rigid_contact_count = model.rigid_contact_count
        self._rigid_contact_shape0 = model.rigid_contact_shape0
        self._rigid_contact_shape1 = model.rigid_contact_shape1
        self._rigid_contact_normal = model.rigid_contact_normal
        self._rigid_contact_point0 = model.rigid_contact_point0
        self._rigid_contact_point1 = model.rigid_contact_point1

        self._contact_gap = wp.zeros(self.N_c, dtype=wp.float32, device=self.device)
        self._J_contact_a = wp.zeros(
            (self.N_c, 3), dtype=wp.spatial_vector, device=self.device
        )
        self._J_contact_b = wp.zeros(
            (self.N_c, 3), dtype=wp.spatial_vector, device=self.device
        )
        self._contact_body_a = wp.zeros(self.N_c, dtype=wp.int32, device=self.device)
        self._contact_body_b = wp.zeros(self.N_c, dtype=wp.int32, device=self.device)
        self._contact_restitution_coeff = wp.zeros(
            self.N_c, dtype=wp.float32, device=self.device
        )
        self._contact_friction_coeff = wp.zeros(
            self.N_c, dtype=wp.float32, device=self.device
        )

        # --- System matrix A and Right-hand side vector b ---
        self.A_op = SystemOperator(self)
        self._b = wp.zeros((con_dim,), dtype=wp.float32, device=self.device)
        self.preconditioner = JacobiPreconditioner(self)

        # self.stream1 = wp.Stream(device=self.device)
        # self.stream2 = wp.Stream(device=self.device)
        # self.stream3 = wp.Stream(device=self.device)

    def _clear_values(self):
        """Resets only the necessary arrays for the next Newton iteration."""
        wp.launch(
            kernel=fused_reset_kernel,
            dim=self.con_dim,
            inputs=[self._h, self._J_values, self._C_values],
            device=self.device,
        )

    def compute_contact_kinematics(self):
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
            outputs=[self._g],
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
                self.lambda_n_offset,
                self._lambda,
                # Parameters
                self._dt,
                CONTACT_CONSTRAINT_STABILIZATION,
                CONTACT_FB_ALPHA,
                CONTACT_FB_BETA,
                # Offsets for output arrays
                self.h_n_offset,
                self.J_n_offset,
                self.C_n_offset,
            ],
            outputs=[
                self._g,
                self._h,
                self._J_values,
                self._C_values,
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
                self.lambda_j_offset,
                self._lambda,
                # Parameters
                self._dt,
                JOINT_CONSTRAINT_STABILIZATION,
                # Offsets for output arrays
                self.h_j_offset,
                self.J_j_offset,
                self.C_j_offset,
            ],
            outputs=[
                self._g,
                self._h,
                self._J_values,
                self._C_values,
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
                self.lambda_n_offset,
                self.lambda_f_offset,
                self._lambda,
                self._lambda_prev,
                # Parameters
                FRICTION_FB_ALPHA,
                FRICTION_FB_BETA,
                # Offsets for output arrays
                self.h_f_offset,
                self.J_f_offset,
                self.C_f_offset,
            ],
            outputs=[
                self._g,
                self._h,
                self._J_values,
                self._C_values,
            ],
        )

        self.preconditioner.update()  # Update the preconditioner with new values
        wp.launch(
            kernel=update_system_rhs_kernel,
            dim=(self.con_dim,),
            inputs=[
                self.body_inv_mass,
                self.body_inv_inertia,
                self.joint_parent,
                self.joint_child,
                self._contact_body_a,
                self._contact_body_b,
                self.J_j_offset,
                self.J_n_offset,
                self.J_f_offset,
                self._J_values,
                self._g,
                self._h,
            ],
            outputs=[self._b],
        )

    def solve_system(self):
        cr_solver(
            A=self.A_op,
            b=self._b,
            x=self._delta_lambda,
            iters=self.linear_iters,
            preconditioner=self.preconditioner,
        )

        # The post-solve steps are the same: apply the computed impulses.
        self._JT_delta_lambda.zero_()  # Must zero before atomic add kernel
        wp.launch(
            kernel=compute_JT_delta_lambda_kernel,
            dim=self.con_dim,
            inputs=[
                self.joint_parent,
                self.joint_child,
                self._contact_body_a,
                self._contact_body_b,
                self.J_j_offset,
                self.J_n_offset,
                self.J_f_offset,
                self._J_values,
                self._delta_lambda,
            ],
            outputs=[self._JT_delta_lambda],
        )
        wp.launch(
            kernel=compute_delta_body_qd_kernel,
            dim=self.dyn_dim,
            inputs=[
                self.body_inv_mass,
                self.body_inv_inertia,
                self._JT_delta_lambda,
                self._g,
            ],
            outputs=[self._delta_body_qd],
        )

        # Add the changes to the state variables.
        add_inplace(self._body_qd, self._delta_body_qd, 0, 0, self.N_b)
        add_inplace(self._lambda, self._delta_lambda, 0, 0, self.con_dim)

    def update_state_variables(
        self, model: Model, state_in: State, state_out: State, dt: float
    ):
        # This method remains unchanged
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

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
    ):
        # The main simulation loop logic remains unchanged
        wp.sim.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
        if control is not None:
            wp.launch(
                kernel=apply_joint_actions_kernel,
                dim=(model.joint_count,),
                inputs=[
                    state_in.body_q,
                    model.body_com,
                    state_in.joint_q,
                    state_in.joint_qd,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis_start,
                    model.joint_axis_dim,
                    model.joint_axis,
                    model.joint_axis_mode,
                    control.joint_act,
                ],
                outputs=[state_in.body_f],
            )

        self.integrate_bodies(model, state_in, state_out, dt)
        self.update_state_variables(model, state_in, state_out, dt)

        self.compute_contact_kinematics()
        self._lambda.zero_()

        # The Newton iteration loop now calls the streamlined methods.
        for _ in range(self.newton_iters):
            wp.copy(dest=self._lambda_prev, src=self._lambda)
            self.update_system_values()
            self.solve_system()

        wp.copy(dest=state_out.body_qd, src=self._body_qd)
        wp.copy(dest=state_out.body_q, src=self._body_q)
