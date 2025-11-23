from dataclasses import dataclass
from functools import cached_property

import numpy as np
import warp as wp
import warp.context as wpc
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.types import contact_interaction_kernel
from axion.types import ContactInteraction
from axion.types import joint_constraint_data_kernel
from axion.types import JointConstraintData
from axion.types import SpatialInertia
from axion.types import world_spatial_inertia_kernel
from newton import Contacts
from newton import Model
from newton import State

from .batched_model import BatchedContacts
from .batched_model import BatchedModel
from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_dims import EngineDimensions


@dataclass(frozen=True)
class EngineArrays:
    """
    Manages all Warp arrays (vectors and matrices) used by the AxionEngine.

    This class centralizes the allocation and provides convenient sliced views
    for the various components of the simulation state and constraint system.

    Use `create_engine_arrays()` to create instances.
    """

    dims: EngineDimensions
    device: wpc.Device

    dt: float

    # --- Primary allocated arrays ---
    _h: wp.array  # Residual
    _J_values: wp.array  # Jacobian values for sparse construction
    _C_values: wp.array  # Compliance values

    _body_lambda: wp.array  # Constraint impulses
    _body_lambda_prev: wp.array  # Constraint impulses at previous newton step
    _dbody_lambda: wp.array  # Change in constraint impulses

    _constraint_body_idx: wp.array  # Indices of bodies involved in constraints

    body_f: wp.array  # External forces
    body_q: wp.array  # Positions
    body_q_prev: wp.array  # Positions at previous time step
    body_u: wp.array  # Velocities
    body_u_prev: wp.array  # Velocities at previous time step
    dbody_u: wp.array  # Change in body velocities

    s_n: wp.array  # Scale for normal impulse
    s_n_prev: wp.array  # Scale for normal impulse at previous newton step

    JT_delta_lambda: wp.array  # J^T * delta_body_lambda
    b: wp.array  # RHS vector for linear system

    body_M: wp.array
    body_M_inv: wp.array
    world_M: wp.array
    world_M_inv: wp.array
    joint_constraint_data: wp.array
    joint_constraint_offsets: wp.array
    contact_interaction: wp.array

    linesearch_steps: wp.array = None
    linesearch_batch_body_u: wp.array = None
    _linesearch_batch_body_lambda: wp.array = None
    _linesearch_batch_h: wp.array = None
    linesearch_batch_h_norm_sq: wp.array = None
    linesearch_minimal_index: wp.array = None

    M_inv_dense: wp.array = None
    J_dense: wp.array = None
    C_dense: wp.array = None

    pca_batch_body_u: wp.array = None
    _pca_batch_body_lambda: wp.array = None
    _pca_batch_h: wp.array = None
    pca_batch_h_norm: wp.array = None

    optim_h: wp.array = None
    optim_trajectory: wp.array = None

    g_accel: wp.array = None

    # 1. System Views (Combined Dynamics + Constraints)
    #    Access pattern: sys.d (dynamics), sys.c.j (joint constraints), etc.

    @cached_property
    def h(self) -> SystemView:
        """Residual vector [h_d, h_c]."""
        return SystemView(self._h, self.dims)

    # 2. Constraint Views (Constraints Only)
    #    Access pattern: const.j, const.n, const.f

    @cached_property
    def J_values(self) -> ConstraintView:
        """Jacobian values view."""
        return ConstraintView(self._J_values, self.dims, axis=-2)

    @cached_property
    def C_values(self) -> ConstraintView:
        """Compliance values view."""
        return ConstraintView(self._C_values, self.dims)

    @cached_property
    def body_lambda(self) -> ConstraintView:
        """Lagrange multipliers view."""
        return ConstraintView(self._body_lambda, self.dims)

    @cached_property
    def body_lambda_prev(self) -> ConstraintView:
        return ConstraintView(self._body_lambda_prev, self.dims)

    @cached_property
    def dbody_lambda(self) -> ConstraintView:
        """Delta Lambda (Newton step) view."""
        return ConstraintView(self._dbody_lambda, self.dims)

    @cached_property
    def constraint_body_idx(self) -> ConstraintView:
        return ConstraintView(self._constraint_body_idx, self.dims, axis=-2)

    # 3. Linesearch Batch Views

    @cached_property
    def linesearch_batch_h(self) -> SystemView:
        return SystemView(self._linesearch_batch_h, self.dims)

    @cached_property
    def linesearch_batch_body_lambda(self) -> ConstraintView:
        return ConstraintView(self._linesearch_batch_body_lambda, self.dims)

    # 4. PCA Batch Views

    @cached_property
    def pca_batch_h(self) -> SystemView:
        return SystemView(self._pca_batch_h, self.dims)

    @cached_property
    def pca_batch_body_lambda(self) -> ConstraintView:
        return ConstraintView(self._pca_batch_body_lambda, self.dims)

    @property
    def has_linesearch(self) -> bool:
        """Returns True if linesearch arrays are allocated (N_alpha > 0)."""
        return self.dims.N_alpha > 0

    @property
    def allocated_pca_arrays(self) -> bool:
        return self.pca_batch_body_u is not None

    def set_g_accel(self, model: Model):
        assert (
            self.g_accel is None
        ), "Setting gravitational acceleration more than once is forbidden"
        object.__setattr__(self, "g_accel", model.gravity)

    def set_dt(self, dt: float):
        object.__setattr__(self, "dt", dt)

    def update_state_data(
        self,
        model: Model,
        batched_model: BatchedModel,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        batched_contacts: BatchedContacts,
        dt: float,
    ):
        self.set_dt(dt)

        wp.copy(dest=self.body_f, src=state_in.body_f)
        wp.copy(dest=self.body_q, src=state_in.body_q)
        wp.copy(dest=self.body_q_prev, src=state_in.body_q)
        wp.copy(dest=self.body_u, src=state_out.body_qd)
        wp.copy(dest=self.body_u_prev, src=state_in.body_qd)

        wp.launch(
            kernel=world_spatial_inertia_kernel,
            dim=(batched_model.num_worlds, batched_model.body_count),
            inputs=[
                self.body_q,
                batched_model.body_mass,
                batched_model.body_inertia,
            ],
            outputs=[
                self.world_M,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=world_spatial_inertia_kernel,
            dim=(batched_model.num_worlds, batched_model.body_count),
            inputs=[
                self.body_q,
                batched_model.body_inv_mass,
                batched_model.body_inv_inertia,
            ],
            outputs=[
                self.world_M_inv,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=(batched_model.num_worlds, batched_contacts.max_contacts),
            inputs=[
                self.body_q,
                batched_model.body_com,
                batched_model.shape_body,
                batched_model.shape_thickness,
                batched_model.shape_material_mu,
                batched_model.shape_material_restitution,
                batched_contacts.contact_count,
                batched_contacts.contact_point0,
                batched_contacts.contact_point1,
                batched_contacts.contact_normal,
                batched_contacts.contact_shape0,
                batched_contacts.contact_shape1,
                batched_contacts.contact_thickness0,
                batched_contacts.contact_thickness1,
            ],
            outputs=[
                self.contact_interaction,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=joint_constraint_data_kernel,
            dim=(batched_model.num_worlds, batched_model.joint_count),
            inputs=[
                self.body_q,
                batched_model.body_com,
                batched_model.joint_type,
                batched_model.joint_enabled,
                batched_model.joint_parent,
                batched_model.joint_child,
                batched_model.joint_X_p,
                batched_model.joint_X_c,
                batched_model.joint_qd_start,
                batched_model.joint_axis,
                self.joint_constraint_offsets,
            ],
            outputs=[
                self.joint_constraint_data,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_joint_constraint_body_idx_kernel,
            dim=(batched_model.num_worlds, self.dims.N_j),
            inputs=[
                self.joint_constraint_data,
            ],
            outputs=[
                self.constraint_body_idx.j,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_contact_constraint_body_idx_kernel,
            dim=(batched_model.num_worlds, self.dims.N_n),
            inputs=[
                self.contact_interaction,
            ],
            outputs=[
                self.constraint_body_idx.n,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_friction_constraint_body_idx_kernel,
            dim=(batched_model.num_worlds, self.dims.N_f),
            inputs=[
                self.contact_interaction,
            ],
            outputs=[
                self.constraint_body_idx.f,
            ],
            device=self.device,
        )


def create_engine_arrays(
    dims: EngineDimensions,
    config: EngineConfig,
    joint_constraint_offsets: wp.array,
    device: wpc.Device,
    allocate_dense: bool = False,
    allocate_pca: bool = False,
    pca_grid_res: int = 100,
) -> EngineArrays:
    """
    Factory function to create and initialize EngineArrays.

    Args:
        dims: Constraint dimensions defining array sizes
        device: Warp device for array allocation
        N_alpha: Number of alpha values for linesearch (0 to disable linesearch)
        alphas: Tuple of alpha values for linesearch (ignored if N_alpha=0)

    Returns:
        Fully initialized EngineArrays instance
    """

    def _zeros(shape, dtype=wp.float32, ndim=None):
        return wp.zeros(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

    def _ones(shape, dtype=wp.float32, ndim=None):
        return wp.ones(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

    def _empty(shape, dtype, device=device):
        return wp.empty(shape, dtype=dtype, device=device).contiguous()

    # ---- Core arrays using tuple shapes ----
    h = _zeros((dims.N_w, dims.N_u + dims.N_c))
    J_values = _zeros((dims.N_w, dims.N_c, 2), wp.spatial_vector)
    C_values = _zeros((dims.N_w, dims.N_c))

    body_f = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
    body_q = _zeros((dims.N_w, dims.N_b), wp.transform)
    body_q_prev = _zeros((dims.N_w, dims.N_b), wp.transform)
    body_u = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
    body_u_prev = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)

    body_lambda = _zeros((dims.N_w, dims.N_c))
    body_lambda_prev = _zeros((dims.N_w, dims.N_c))

    s_n = _zeros((dims.N_w, dims.N_n))
    s_n_prev = _zeros((dims.N_w, dims.N_n))

    constraint_body_idx = _zeros((dims.N_w, dims.N_c, 2), wp.int32)

    JT_delta_lambda = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
    dbody_u = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
    dbody_lambda = _zeros((dims.N_w, dims.N_c))

    b = _zeros((dims.N_w, dims.N_c))

    body_M = _empty((dims.N_w, dims.N_b), SpatialInertia)
    body_M_inv = _empty((dims.N_w, dims.N_b), SpatialInertia)
    world_M = _empty((dims.N_w, dims.N_b), SpatialInertia)
    world_M_inv = _empty((dims.N_w, dims.N_b), SpatialInertia)

    joint_constraint_data = _empty((dims.N_w, dims.N_j), JointConstraintData)
    contact_interaction = _empty((dims.N_w, dims.N_n), ContactInteraction)

    # ---- Linesearch Arrays ----
    linesearch_steps = None
    linesearch_batch_body_u = None
    linesearch_batch_body_lambda = None
    linesearch_batch_h = None
    linesearch_batch_h_norm_sq = None
    linesearch_minimal_index = None

    if config.enable_linesearch:
        step_count = config.linesearch_step_count

        log_step_min = np.log10(config.linesearch_step_min)
        log_step_max = np.log10(config.linesearch_step_max)
        ls_steps_np = np.logspace(log_step_min, log_step_max, step_count)

        # force one step to be 1.0
        closest = np.argmin(np.abs(ls_steps_np - 1.0))
        ls_steps_np[closest] = 1.0
        linesearch_steps = wp.from_numpy(ls_steps_np, dtype=wp.float32)

        linesearch_batch_body_u = _zeros((step_count, dims.N_w, dims.N_b), wp.spatial_vector)
        linesearch_batch_body_lambda = _zeros((step_count, dims.N_w, dims.N_c))
        linesearch_batch_h = _zeros((step_count, dims.N_w, dims.N_u + dims.N_c))
        linesearch_batch_h_norm_sq = _zeros((step_count, dims.N_w))
        linesearch_minimal_index = _zeros((dims.N_w,), wp.int32)

    # ---- Dense Representation ----
    M_inv_dense = None
    J_dense = None
    C_dense = None
    if allocate_dense:
        M_inv_dense = _zeros((dims.N_w, dims.N_u, dims.N_u))
        J_dense = _zeros((dims.N_w, dims.N_c, dims.N_u))
        C_dense = _zeros((dims.N_w, dims.N_c, dims.N_c))

    # ---- PCA Storage Buffers ----
    (
        optim_h,
        optim_trajectory,
        pca_batch_body_u,
        pca_batch_body_lambda,
        pca_batch_h,
        pca_batch_h_norm,
    ) = (None, None, None, None, None, None)

    if allocate_pca:
        pca_batch_size = pca_grid_res * pca_grid_res

        optim_h = _zeros((config.newton_iters, dims.N_w, dims.N_u + dims.N_c))
        optim_trajectory = _zeros((config.newton_iters, dims.N_w, dims.N_u + dims.N_c))

        pca_batch_body_u = _zeros((pca_batch_size, dims.N_w, dims.N_b), wp.spatial_vector)
        pca_batch_body_lambda = _zeros((pca_batch_size, dims.N_w, dims.N_c))
        pca_batch_h = _zeros((pca_batch_size, dims.N_w, dims.N_u + dims.N_c))
        pca_batch_h_norm = _zeros((pca_batch_size, dims.N_w))

    return EngineArrays(
        dims=dims,
        device=device,
        dt=None,
        _h=h,
        _J_values=J_values,
        _C_values=C_values,
        body_f=body_f,
        body_q=body_q,
        body_q_prev=body_q_prev,
        body_u=body_u,
        body_u_prev=body_u_prev,
        dbody_u=dbody_u,
        _body_lambda=body_lambda,
        _body_lambda_prev=body_lambda_prev,
        s_n=s_n,
        s_n_prev=s_n_prev,
        _constraint_body_idx=constraint_body_idx,
        JT_delta_lambda=JT_delta_lambda,
        _dbody_lambda=dbody_lambda,
        b=b,
        body_M=body_M,
        body_M_inv=body_M_inv,
        world_M=world_M,
        world_M_inv=world_M_inv,
        joint_constraint_data=joint_constraint_data,
        joint_constraint_offsets=joint_constraint_offsets,
        contact_interaction=contact_interaction,
        linesearch_steps=linesearch_steps,
        linesearch_batch_body_u=linesearch_batch_body_u,
        _linesearch_batch_body_lambda=linesearch_batch_body_lambda,
        _linesearch_batch_h=linesearch_batch_h,
        linesearch_batch_h_norm_sq=linesearch_batch_h_norm_sq,
        linesearch_minimal_index=linesearch_minimal_index,
        M_inv_dense=M_inv_dense,
        J_dense=J_dense,
        C_dense=C_dense,
        pca_batch_body_u=pca_batch_body_u,
        _pca_batch_body_lambda=pca_batch_body_lambda,
        _pca_batch_h=pca_batch_h,
        pca_batch_h_norm=pca_batch_h_norm,
        optim_h=optim_h,
        optim_trajectory=optim_trajectory,
    )
