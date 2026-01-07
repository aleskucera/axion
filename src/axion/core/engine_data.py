from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import warp as wp
import warp.context as wpc
from axion.types import ContactInteraction
from axion.types import SpatialInertia
from axion.tiled.tiled_utils import TiledSqNorm
from newton import Model

from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_dims import EngineDimensions


@dataclass(frozen=True)
class LinesearchData:
    dims: EngineDimensions
    steps: wp.array
    batch_body_u: wp.array
    batch_body_q: wp.array
    _batch_body_lambda: wp.array
    _batch_h: wp.array
    batch_h_norm_sq: wp.array
    minimal_index: wp.array
    tiled_sq_norm: Optional[TiledSqNorm] = None
    _batch_h_spatial: Optional[wp.array] = None

    @cached_property
    def batch_h(self) -> SystemView:
        # return SystemView(self._batch_h, self.dims, _d_spatial=self._batch_h_spatial)
        return SystemView(self._batch_h, self.dims)

    @cached_property
    def batch_body_lambda(self) -> ConstraintView:
        return ConstraintView(self._batch_body_lambda, self.dims)


@dataclass(frozen=True)
class HistoryData:
    dims: EngineDimensions
    _h_history: wp.array
    body_q_history: wp.array
    body_u_history: wp.array
    _body_lambda_history: wp.array

    pca_batch_body_q: wp.array
    pca_batch_body_u: wp.array
    _pca_batch_body_lambda: wp.array
    _pca_batch_h: wp.array
    pca_batch_h_norm: wp.array

    _h_history_spatial: Optional[wp.array] = None
    _pca_batch_h_spatial: Optional[wp.array] = None

    @cached_property
    def h_history(self) -> SystemView:
        # return SystemView(self._h_history, self.dims, _d_spatial=self._h_history_spatial)
        return SystemView(self._h_history, self.dims)

    @cached_property
    def body_lambda_history(self) -> ConstraintView:
        return ConstraintView(self._body_lambda_history, self.dims)

    @cached_property
    def pca_batch_h(self) -> SystemView:
        # return SystemView(self._pca_batch_h, self.dims, _d_spatial=self._pca_batch_h_spatial)
        return SystemView(self._pca_batch_h, self.dims)

    @cached_property
    def pca_batch_body_lambda(self) -> ConstraintView:
        return ConstraintView(self._pca_batch_body_lambda, self.dims)


@dataclass(frozen=True)
class EngineData:
    """
    Manages all Warp arrays (vectors and matrices) used by the AxionEngine.

    This class centralizes the allocation and provides convenient sliced views
    for the various components of the simulation state and constraint system.

    Use `EngineData.create()` to create instances.
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
    _constraint_active_mask: wp.array

    body_f: wp.array  # External forces
    body_q: wp.array  # Positions
    body_q_prev: wp.array  # Positions at previous time step
    body_u: wp.array  # Velocities
    body_u_prev: wp.array  # Velocities at previous time step
    dbody_u: wp.array  # Change in body velocities

    s_n: wp.array  # Scale for normal impulse
    s_n_prev: wp.array  # Scale for normal impulse at previous newton step

    JT_delta_lambda: wp.array  # J^T * delta_body_lambda
    system_diag: wp.array  # diagonal of J @ M^{-1} @ J^T
    b: wp.array  # RHS vector for linear system

    world_M: wp.array
    world_M_inv: wp.array
    joint_constraint_offsets: wp.array
    control_constraint_offsets: wp.array
    contact_interaction: wp.array
    joint_target: wp.array

    linesearch: Optional[LinesearchData] = None
    history: Optional[HistoryData] = None

    g_accel: wp.array = None
    _h_spatial: Optional[wp.array] = None

    # 1. System Views (Combined Dynamics + Constraints)
    #    Access pattern: sys.d (dynamics), sys.c.j (joint constraints), etc.

    @cached_property
    def h(self) -> SystemView:
        """Residual vector [h_d, h_c]."""
        # return SystemView(self._h, self.dims, _d_spatial=self._h_spatial)
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

    @cached_property
    def constraint_active_mask(self) -> ConstraintView:
        return ConstraintView(self._constraint_active_mask, self.dims)

    @property
    def has_linesearch(self) -> bool:
        """Returns True if linesearch arrays are allocated (N_alpha > 0)."""
        return self.linesearch is not None

    @property
    def allocated_pca_arrays(self) -> bool:
        return self.history is not None

    def set_g_accel(self, model: Model):
        assert (
            self.g_accel is None
        ), "Setting gravitational acceleration more than once is forbidden"
        object.__setattr__(self, "g_accel", model.gravity)

    def set_dt(self, dt: float):
        object.__setattr__(self, "dt", dt)

    @staticmethod
    def create(
        dims: EngineDimensions,
        config: EngineConfig,
        joint_constraint_offsets: wp.array,
        control_constraint_offsets: wp.array,
        dof_count: int,
        device: wpc.Device,
        allocate_pca: bool = False,
        pca_grid_res: int = 100,
    ) -> EngineData:
        """
        Factory function to create and initialize EngineData.

        Args:
            dims: Constraint dimensions defining array sizes
            device: Warp device for array allocation
            config: Engine configuration
            allocate_pca: Whether to allocate history/PCA arrays

        Returns:
            Fully initialized EngineData instance
        """

        def _zeros(shape, dtype=wp.float32, ndim=None):
            return wp.zeros(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

        def _ones(shape, dtype=wp.float32, ndim=None):
            return wp.ones(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

        def _empty(shape, dtype, device=device):
            return wp.empty(shape, dtype=dtype, device=device).contiguous()

        def _make_spatial_view(src, n_b):
            # src is (..., N_u + N_c)
            # we want (..., n_b) spatial
            new_shape = src.shape[:-1] + (n_b,)
            new_strides = src.strides[:-1] + (24,)
            return wp.array(
                ptr=src.ptr,
                shape=new_shape,
                dtype=wp.spatial_vector,
                strides=new_strides,
                device=src.device,
            )

        # ---- Core arrays using tuple shapes ----
        h = _zeros((dims.N_w, dims.N_u + dims.N_c))
        h_spatial = _make_spatial_view(h, dims.N_b)
        
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
        constraint_active_mask = _zeros((dims.N_w, dims.N_c), wp.float32)

        JT_delta_lambda = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
        system_diag = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
        dbody_u = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
        dbody_lambda = _zeros((dims.N_w, dims.N_c))

        b = _zeros((dims.N_w, dims.N_c))

        world_M = _empty((dims.N_w, dims.N_b), SpatialInertia)
        world_M_inv = _empty((dims.N_w, dims.N_b), SpatialInertia)

        contact_interaction = _empty((dims.N_w, dims.N_n), ContactInteraction)

        joint_target = _zeros((dims.N_w, dof_count))

        # ---- Linesearch Arrays ----
        linesearch_data = None

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
            linesearch_batch_body_q = _zeros((step_count, dims.N_w, dims.N_b), wp.transform)
            linesearch_batch_body_lambda = _zeros((step_count, dims.N_w, dims.N_c))
            linesearch_batch_h = _zeros((step_count, dims.N_w, dims.N_u + dims.N_c))
            ls_h_spatial = _make_spatial_view(linesearch_batch_h, dims.N_b)

            linesearch_batch_h_norm_sq = _zeros((step_count, dims.N_w))
            linesearch_minimal_index = _zeros((dims.N_w,), wp.int32)

            tiled_sq_norm = TiledSqNorm(
                shape=linesearch_batch_h.shape,
                dtype=wp.float32,
                device=device,
            )

            linesearch_data = LinesearchData(
                dims=dims,
                steps=linesearch_steps,
                batch_body_u=linesearch_batch_body_u,
                batch_body_q=linesearch_batch_body_q,
                _batch_body_lambda=linesearch_batch_body_lambda,
                _batch_h=linesearch_batch_h,
                batch_h_norm_sq=linesearch_batch_h_norm_sq,
                minimal_index=linesearch_minimal_index,
                tiled_sq_norm=tiled_sq_norm,
                _batch_h_spatial=ls_h_spatial,
            )

        # ---- PCA Storage Buffers ----
        history_data = None

        if allocate_pca:
            pca_batch_size = pca_grid_res * pca_grid_res

            h_history = _zeros((config.newton_iters + 1, dims.N_w, dims.N_u + dims.N_c))
            h_history_spatial = _make_spatial_view(h_history, dims.N_b)
            
            body_q_history = _zeros(
                (config.newton_iters + 1, dims.N_w, dims.N_b), dtype=wp.transform
            )
            body_u_history = _zeros(
                (config.newton_iters + 1, dims.N_w, dims.N_b), dtype=wp.spatial_vector
            )
            body_lambda_history = _zeros((config.newton_iters + 1, dims.N_w, dims.N_c))

            pca_batch_body_q = _zeros((pca_batch_size, dims.N_w, dims.N_b), wp.transform)
            pca_batch_body_u = _zeros((pca_batch_size, dims.N_w, dims.N_b), wp.spatial_vector)
            pca_batch_body_lambda = _zeros((pca_batch_size, dims.N_w, dims.N_c))
            pca_batch_h = _zeros((pca_batch_size, dims.N_w, dims.N_u + dims.N_c))
            pca_batch_h_spatial = _make_spatial_view(pca_batch_h, dims.N_b)
            
            pca_batch_h_norm = _zeros((pca_batch_size, dims.N_w))

            history_data = HistoryData(
                dims=dims,
                _h_history=h_history,
                body_q_history=body_q_history,
                body_u_history=body_u_history,
                _body_lambda_history=body_lambda_history,
                pca_batch_body_q=pca_batch_body_q,
                pca_batch_body_u=pca_batch_body_u,
                _pca_batch_body_lambda=pca_batch_body_lambda,
                _pca_batch_h=pca_batch_h,
                pca_batch_h_norm=pca_batch_h_norm,
                _h_history_spatial=h_history_spatial,
                _pca_batch_h_spatial=pca_batch_h_spatial,
            )

        return EngineData(
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
            _body_lambda=body_lambda,
            _body_lambda_prev=body_lambda_prev,
            _dbody_lambda=dbody_lambda,
            dbody_u=dbody_u,
            JT_delta_lambda=JT_delta_lambda,
            system_diag=system_diag,
            b=b,
            world_M=world_M,
            world_M_inv=world_M_inv,
            _constraint_body_idx=constraint_body_idx,
            _constraint_active_mask=constraint_active_mask,
            s_n=s_n,
            s_n_prev=s_n_prev,
            contact_interaction=contact_interaction,
            joint_constraint_offsets=joint_constraint_offsets,
            control_constraint_offsets=control_constraint_offsets,
            joint_target=joint_target,
            linesearch=linesearch_data,
            history=history_data,
            _h_spatial=h_spatial,
        )
