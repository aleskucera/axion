import numpy as np
import warp as wp
from axion.constraints import fill_control_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.tiled import TiledSqNorm
from axion.types.spatial_inertia import SpatialInertia

from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_dims import EngineDimensions
from .model import AxionModel


def _compute_linesearch_step_size_array(config: EngineConfig) -> wp.array:
    # --- 1. Conservative Steps (Logarithmic) ---
    # "I don't trust the solver, let's try tiny steps."
    steps_conservative = np.logspace(
        np.log10(config.linesearch_min_step),
        np.log10(config.linesearch_conservative_upper_bound),
        config.linesearch_conservative_step_count,
    )

    # --- 2. Optimistic Steps (Linear) ---
    # "I trust the Newton direction, let's check around 1.0."
    half_window = config.linesearch_optimistic_window / 2.0
    steps_optimistic = np.linspace(
        1.0 - half_window, 1.0 + half_window, config.linesearch_optimistic_step_count
    )

    # --- 3. Combine & Sort ---
    ls_steps_np = np.concatenate([steps_conservative, steps_optimistic])
    ls_steps_np.sort()

    # Force exact 1.0 to ensure the standard Newton step is tested
    closest_idx = np.argmin(np.abs(ls_steps_np - 1.0))
    ls_steps_np[closest_idx] = 1.0

    return wp.from_numpy(ls_steps_np, dtype=wp.float32)


class EngineData:
    def __init__(
        self,
        model: AxionModel,
        dims: EngineDimensions,
        config: EngineConfig,
        device: wp.Device,
        allocate_history: bool = False,
        allocate_grad: bool = True,
    ):
        self.device = device

        self.dt: float = None

        # --- Helper for concise allocation ---
        def _alloc(shape, dtype, requires_grad=False):
            assert isinstance(shape, tuple)

            batched_shape = (dims.num_worlds,) + shape
            return wp.zeros(batched_shape, dtype=dtype, device=device, requires_grad=requires_grad)

        def _alloc_buffer(buffer_size, array):
            return wp.zeros((buffer_size,) + array.shape, dtype=array.dtype, device=device)

        # =========================================================================
        # Body State Arrays
        # =========================================================================
        # External force
        self.ext_force = _alloc((dims.body_count,), wp.spatial_vector, allocate_grad)

        # State of bodies (q - position, u - velocity)
        self.body_pose = _alloc((dims.body_count,), wp.transform)
        self.body_vel = _alloc((dims.body_count,), wp.spatial_vector)

        # State at previous timestep
        self.body_pose_prev = _alloc((dims.body_count,), wp.transform, allocate_grad)
        self.body_vel_prev = _alloc((dims.body_count,), wp.spatial_vector, allocate_grad)

        # Actuation
        self.joint_target_pos = _alloc((dims.joint_dof_count,), wp.float32)
        self.joint_target_vel = _alloc((dims.joint_dof_count,), wp.float32)

        # =========================================================================
        # Constraint Arrays
        # =========================================================================
        self._constr_force = _alloc((dims.num_constraints,), wp.float32)
        self._constr_force_prev_iter = _alloc((dims.num_constraints,), wp.float32)
        self._constr_body_idx = _alloc((dims.num_constraints, 2), wp.int32)
        self._constr_active_mask = _alloc((dims.num_constraints,), wp.float32)

        self.constr_force = ConstraintView(self._constr_force, dims)
        self.constr_force_prev_iter = ConstraintView(self._constr_force_prev_iter, dims)
        self.constr_body_idx = ConstraintView(self._constr_body_idx, dims, axis=-2)
        self.constr_active_mask = ConstraintView(self._constr_active_mask, dims)

        # Contact information
        self.contact_body_a = _alloc((dims.contact_count,), wp.int32)
        self.contact_body_b = _alloc((dims.contact_count,), wp.int32)
        self.contact_point_a = _alloc((dims.contact_count,), wp.vec3)
        self.contact_point_b = _alloc((dims.contact_count,), wp.vec3)
        self.contact_thickness_a = _alloc((dims.contact_count,), wp.float32)
        self.contact_thickness_b = _alloc((dims.contact_count,), wp.float32)
        self.contact_dist = _alloc((dims.contact_count,), wp.float32)
        self.contact_friction_coeff = _alloc((dims.contact_count,), wp.float32)
        self.contact_restitution_coeff = _alloc((dims.contact_count,), wp.float32)
        self.contact_basis_n_a = _alloc((dims.contact_count,), wp.spatial_vector)
        self.contact_basis_t1_a = _alloc((dims.contact_count,), wp.spatial_vector)
        self.contact_basis_t2_a = _alloc((dims.contact_count,), wp.spatial_vector)
        self.contact_basis_n_b = _alloc((dims.contact_count,), wp.spatial_vector)
        self.contact_basis_t1_b = _alloc((dims.contact_count,), wp.spatial_vector)
        self.contact_basis_t2_b = _alloc((dims.contact_count,), wp.spatial_vector)

        # =========================================================================
        # Linear System Arrays
        # =========================================================================
        # Residual
        self._res = _alloc((dims.N_u + dims.num_constraints,), wp.float32, allocate_grad)
        self._res_spatial = _alloc((dims.body_count,), wp.spatial_vector, allocate_grad)

        self.res = SystemView(self._res, dims, self._res_spatial)

        # Efficiently stored values of sparse system matrix
        self.world_M = _alloc((dims.body_count,), SpatialInertia)
        self.world_M_inv = _alloc((dims.body_count,), SpatialInertia)
        self._J_values = _alloc((dims.num_constraints, 2), wp.spatial_vector)
        self._C_values = _alloc((dims.num_constraints,), wp.float32)

        self.J_values = ConstraintView(self._J_values, dims, axis=-2)
        self.C_values = ConstraintView(self._C_values, dims)

        # Intermediate array for linearization
        self.JT_dconstr_force = _alloc((dims.body_count,), wp.spatial_vector)

        # The unknown arrays for the linear solve
        self.dbody_vel = _alloc((dims.body_count,), wp.spatial_vector)
        self._dconstr_force = _alloc((dims.num_constraints,), wp.float32)

        self.dconstr_force = ConstraintView(self._dconstr_force, dims)

        # The right-hand side of the Schur-Complement
        self.rhs = _alloc((dims.num_constraints,), wp.float32)

        # =========================================================================
        # Adjoint Arrays
        # =========================================================================
        if allocate_grad:
            # The adjoint vector
            self._w = _alloc((dims.N_u + dims.num_constraints,), wp.float32)
            self._w_spatial = _alloc((dims.body_count,), wp.spatial_vector)

            self.w = SystemView(self._w, dims, self._w_spatial)

            # The right-hand side of the Schur-Complement of the adjoint linear system
            self.adjoint_rhs = _alloc((dims.num_constraints,), wp.float32)

            self.body_pose_grad = _alloc((dims.body_count,), wp.float32)
            self.body_vel_grad = _alloc((dims.body_count,), wp.float32)

        # =========================================================================
        # Linesearch Arrays
        # =========================================================================
        if config.enable_linesearch:
            step_count = config.linesearch_conservative_step_count
            step_count += config.linesearch_optimistic_step_count

            self.linesearch_step_size = _compute_linesearch_step_size_array(config)

            self.linesearch_body_pose = _alloc_buffer(step_count, self.body_pose)
            self.linesearch_body_vel = _alloc_buffer(step_count, self.body_vel)

            self._linesearch_constr_force = _alloc_buffer(step_count, self._constr_force)
            self.linesearch_constr_force = ConstraintView(self._linesearch_constr_force, dims)

            self._linesearch_res = _alloc_buffer(step_count, self._res)
            self._linesearch_res_spatial = _alloc_buffer(step_count, self._res_spatial)

            self.linesearch_res = SystemView(
                self._linesearch_res, dims, self._linesearch_res_spatial
            )

            self.linesearch_res_norm_sq = wp.zeros((step_count, dims.num_worlds), wp.float32)
            self.linesearch_minimal_index = wp.zeros((dims.num_worlds,), wp.int32)

            # Class for computing squared norm efficiently
            self.linesearch_tiled_res_sq_norm = TiledSqNorm(
                shape=self._linesearch_res.shape,
                dtype=wp.float32,
                device=device,
            )

        # ----- REMOVE -----
        # =========================================================================
        # LOGGING: Newton-Raphson (NR) History Arrays
        # =========================================================================
        if allocate_history:
            buffer_size = config.max_newton_iters + 1
            self.nr_history_body_pose = _alloc_buffer(buffer_size, self.body_pose)
            self.nr_history_body_vel = _alloc_buffer(buffer_size, self.body_vel)

            self._nr_history_constr_force = _alloc_buffer(buffer_size, self._constr_force)
            self.nr_history_constr_force = ConstraintView(self._nr_history_constr_force, dims)

            self._nr_history_res = _alloc_buffer(buffer_size, self._res)
            self._nr_history_res_spatial = _alloc_buffer(buffer_size, self._res_spatial)

            self.nr_history_res = SystemView(
                self._nr_history_res, dims, self._nr_history_res_spatial
            )

        # =========================================================================
        # LOGGING: Preconditioned Conjugate Residual (PCR) History Arrays
        # =========================================================================
        if allocate_history:
            buffer_size = config.max_linear_iters + 1
            self.pcr_history_body_pose = _alloc_buffer(buffer_size, self.body_pose)
            self.pcr_history_body_vel = _alloc_buffer(buffer_size, self.body_vel)

            self._pcr_history_constr_force = _alloc_buffer(buffer_size, self._constr_force)
            self.pcr_history_constr_force = ConstraintView(self._pcr_history_constr_force, dims)

            self._pcr_history_res = _alloc_buffer(buffer_size, self._res)
            self._pcr_history_res_spatial = _alloc_buffer(buffer_size, self._res_spatial)

            self.pcr_history_res = SystemView(
                self._pcr_history_res, dims, self._pcr_history_res_spatial
            )

        # =========================================================================
        # Init Kernels
        # =========================================================================
        wp.launch(
            kernel=fill_joint_constraint_body_idx_kernel,
            dim=(dims.num_worlds, dims.joint_count),
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_constraint_offsets,
            ],
            outputs=[
                self.constr_body_idx.j,
            ],
            device=device,
        )

        wp.launch(
            kernel=fill_control_constraint_body_idx_kernel,
            dim=(dims.num_worlds, dims.joint_count),
            inputs=[
                model.joint_parent,
                model.joint_child,
                model.joint_type,
                model.joint_dof_mode,
                model.joint_qd_start,
                model.control_constraint_offsets,
            ],
            outputs=[
                self.constr_body_idx.ctrl,
            ],
            device=device,
        )

    def zero_gradients(self):
        self.ext_force.grad.zero_()
        self.body_pose_prev.grad.zero_()
        self.body_vel_prev.grad.zero_()

        self._res.grad.zero_()
        self._res_spatial.grad.zero_()

        self.joint_target_pos.grad.zero_()
        self.joint_target_vel.grad.zero_()
