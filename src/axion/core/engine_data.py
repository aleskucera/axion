from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import warp as wp
import warp.context as wpc
from axion.tiled.tiled_utils import TiledSqNorm
from axion.types import SpatialInertia
from newton import Model

from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_dims import EngineDimensions


@dataclass
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
        return SystemView(self._batch_h, self.dims, _d_spatial=self._batch_h_spatial)

    @cached_property
    def batch_body_lambda(self) -> ConstraintView:
        return ConstraintView(self._batch_body_lambda, self.dims)


@dataclass
class NewtonHistoryData:
    dims: EngineDimensions
    _h_history: wp.array
    body_q_history: wp.array
    body_u_history: wp.array
    _body_lambda_history: wp.array

    _h_history_spatial: Optional[wp.array] = None

    @cached_property
    def h_history(self) -> SystemView:
        return SystemView(self._h_history, self.dims, _d_spatial=self._h_history_spatial)

    @cached_property
    def body_lambda_history(self) -> ConstraintView:
        return ConstraintView(self._body_lambda_history, self.dims)


@dataclass
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
    _h: wp.array  # Residual (Padded)
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
    joint_target: wp.array

    # --- Contact Interaction SOA ---
    contact_body_a: wp.array
    contact_body_b: wp.array
    contact_point_a: wp.array
    contact_point_b: wp.array
    contact_thickness_a: wp.array
    contact_thickness_b: wp.array
    contact_dist: wp.array
    contact_friction_coeff: wp.array
    contact_restitution_coeff: wp.array
    contact_basis_n_a: wp.array
    contact_basis_t1_a: wp.array
    contact_basis_t2_a: wp.array
    contact_basis_n_b: wp.array
    contact_basis_t1_b: wp.array
    contact_basis_t2_b: wp.array

    linesearch: Optional[LinesearchData] = None
    newton_history: Optional[NewtonHistoryData] = None

    g_accel: wp.array = None
    _h_spatial: Optional[wp.array] = None

    # 1. System Views (Combined Dynamics + Constraints)
    #    Access pattern: sys.d (dynamics), sys.c.j (joint constraints), etc.

    @cached_property
    def h(self) -> SystemView:
        """Residual vector [h_d, h_c]."""
        return SystemView(self._h, self.dims, _d_spatial=self._h_spatial)

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
    def allocated_history_arrays(self) -> bool:
        return self.newton_history is not None

    def set_g_accel(self, model: Model):
        assert (
            self.g_accel is None
        ), "Setting gravitational acceleration more than once is forbidden"
        object.__setattr__(self, "g_accel", model.gravity)

    def set_dt(self, dt: float):
        object.__setattr__(self, "dt", dt)

    def _serialize_to_numpy(self, data: Any) -> Any:
        """
        Recursively converts Warp arrays (and Views) into pure NumPy objects.
        - wp.array -> np.ndarray
        - Struct array -> dict of np.ndarrays (expanded fields)
        - Scalar -> scalar
        """
        if data is None:
            return None

        # Handle Views or Arrays
        if hasattr(data, "numpy"):
            np_data = data.numpy()

            # If it is a structured array (from a Warp Struct), expand it to a dict
            if np_data.dtype.names:
                return {
                    name: self._serialize_to_numpy(np_data[name]) for name in np_data.dtype.names
                }

            return np_data

        return data

    def _get_norm_sq(self, data: Any) -> Optional[np.ndarray]:
        """Serializes data to numpy and computes the squared norm along the last axis."""
        np_data = self._serialize_to_numpy(data)
        if np_data is None:
            return None
        return np.sum(np.square(np_data), axis=-1)

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Returns a dictionary view of the current physics state.
        Decouples logging from internal data structure.
        All data is converted to NumPy arrays or standard Python types.
        """
        snapshot = {}

        # 1. Dynamics
        snapshot["dynamics"] = {
            "h_d": self._serialize_to_numpy(self.h.d),
            "body_q": self._serialize_to_numpy(self.body_q),
            "body_u": self._serialize_to_numpy(self.body_u),
            "body_u_prev": self._serialize_to_numpy(self.body_u_prev),
            "body_f": self._serialize_to_numpy(self.body_f),
            "world_M": self._serialize_to_numpy(self.world_M),
            "world_M_inv": self._serialize_to_numpy(self.world_M_inv),
        }

        # 2. Constraints
        constraints = {}

        if self.dims.N_j > 0:
            constraints["Joint constraint data"] = {
                "h": self._serialize_to_numpy(self.h.c.j),
                "J_values": self._serialize_to_numpy(self.J_values.j),
                "C_values": self._serialize_to_numpy(self.C_values.j),
                "body_lambda": self._serialize_to_numpy(self.body_lambda.j),
            }

        if self.dims.N_ctrl > 0:
            constraints["Control constraint data"] = {
                "h": self._serialize_to_numpy(self.h.c.ctrl),
                "J_values": self._serialize_to_numpy(self.J_values.ctrl),
                "C_values": self._serialize_to_numpy(self.C_values.ctrl),
                "body_lambda": self._serialize_to_numpy(self.body_lambda.ctrl),
            }

        if self.dims.N_n > 0:
            constraints["Contact constraint data"] = {
                "h": self._serialize_to_numpy(self.h.c.n),
                "s": self._serialize_to_numpy(self.s_n),
                "body_lambda": self._serialize_to_numpy(self.body_lambda.n),
                "contact_body_a": self._serialize_to_numpy(self.contact_body_a),
                "contact_body_b": self._serialize_to_numpy(self.contact_body_b),
                "contact_dist": self._serialize_to_numpy(self.contact_dist),
                "contact_friction_coeff": self._serialize_to_numpy(self.contact_friction_coeff),
                "contact_restitution_coeff": self._serialize_to_numpy(
                    self.contact_restitution_coeff
                ),
            }
            constraints["Friction constraint data"] = {
                "h": self._serialize_to_numpy(self.h.c.f),
                "body_lambda": self._serialize_to_numpy(self.body_lambda.f),
            }

        snapshot["constraints"] = constraints

        # 3. Linear System
        snapshot["linear_system"] = {
            "b": self._serialize_to_numpy(self.b),
            "system_diag": self._serialize_to_numpy(self.system_diag),
            "dbody_u": self._serialize_to_numpy(self.dbody_u),
            "dbody_u_norm_sq": self._get_norm_sq(
                self.dbody_u.numpy().reshape(self.dbody_u.shape[0], -1)
            ),
            "dbody_lambda": self._serialize_to_numpy(self.dbody_lambda.full),
            "dbody_lambda_norm_sq": self._get_norm_sq(self.dbody_lambda.full),
        }

        # 4. Linesearch (if allocated)
        if self.linesearch:
            snapshot["linesearch"] = {
                "steps": self._serialize_to_numpy(self.linesearch.steps),
                "batch_h_norm_sq": self._serialize_to_numpy(self.linesearch.batch_h_norm_sq),
                "minimal_index": self._serialize_to_numpy(self.linesearch.minimal_index),
                "batch_h_d_norm_sq": self._get_norm_sq(self.linesearch.batch_h.d),
                "batch_h_c_j_norm_sq": self._get_norm_sq(self.linesearch.batch_h.c.j),
                "batch_h_c_n_norm_sq": self._get_norm_sq(self.linesearch.batch_h.c.n),
                "batch_h_c_f_norm_sq": self._get_norm_sq(self.linesearch.batch_h.c.f),
                "batch_h_c_ctrl_norm_sq": self._get_norm_sq(self.linesearch.batch_h.c.ctrl),
            }

        # 5. History (if allocated)
        if self.newton_history:
            history_snap = {}

            # Dynamics History
            history_snap["dynamics"] = {
                "h_d": self._serialize_to_numpy(self.newton_history.h_history.d),
                "body_q": self._serialize_to_numpy(self.newton_history.body_q_history),
                "body_u": self._serialize_to_numpy(self.newton_history.body_u_history),
            }

            # Constraints History
            constraints_hist = {}
            if self.dims.N_j > 0:
                constraints_hist["Joint constraint data"] = {
                    "h": self._serialize_to_numpy(self.newton_history.h_history.c.j),
                    "body_lambda": self._serialize_to_numpy(self.newton_history.body_lambda_history.j),
                }

            if self.dims.N_ctrl > 0:
                constraints_hist["Control constraint data"] = {
                    "h": self._serialize_to_numpy(self.newton_history.h_history.c.ctrl),
                    "body_lambda": self._serialize_to_numpy(self.newton_history.body_lambda_history.ctrl),
                }

            if self.dims.N_n > 0:
                constraints_hist["Contact constraint data"] = {
                    "h": self._serialize_to_numpy(self.newton_history.h_history.c.n),
                    "body_lambda": self._serialize_to_numpy(self.newton_history.body_lambda_history.n),
                }
                constraints_hist["Friction constraint data"] = {
                    "h": self._serialize_to_numpy(self.newton_history.h_history.c.f),
                    "body_lambda": self._serialize_to_numpy(self.newton_history.body_lambda_history.f),
                }

            history_snap["constraints"] = constraints_hist
            snapshot["newton_history"] = history_snap

        return snapshot

    @staticmethod
    def create(
        dims: EngineDimensions,
        config: EngineConfig,
        joint_constraint_offsets: wp.array,
        control_constraint_offsets: wp.array,
        dof_count: int,
        device: wpc.Device,
        allocate_history: bool = False,
    ) -> EngineData:

        def _zeros(shape, dtype=wp.float32, ndim=None):
            return wp.zeros(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

        def _ones(shape, dtype=wp.float32, ndim=None):
            return wp.ones(shape, dtype=dtype, device=device, ndim=ndim).contiguous()

        def _empty(shape, dtype, device=device):
            return wp.empty(shape, dtype=dtype, device=device).contiguous()

        # ---- Core arrays ----

        # 1. Standard Float Allocation (No Padding)
        h = _zeros((dims.N_w, dims.N_u + dims.N_c))

        # 2. Separate Spatial Buffer Allocation
        h_spatial = _zeros((dims.N_w, dims.N_b), dtype=wp.spatial_vector)

        J_values = _zeros((dims.N_w, dims.N_c, 2), wp.spatial_vector)
        C_values = _zeros((dims.N_w, dims.N_c))

        # ... (Other allocations: body_f, body_q, etc. remain the same) ...
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
        system_diag = _zeros((dims.N_w, dims.N_c), wp.float32)
        dbody_u = _zeros((dims.N_w, dims.N_b), wp.spatial_vector)
        dbody_lambda = _zeros((dims.N_w, dims.N_c))

        b = _zeros((dims.N_w, dims.N_c))

        world_M = _empty((dims.N_w, dims.N_b), SpatialInertia)
        world_M_inv = _empty((dims.N_w, dims.N_b), SpatialInertia)

        contact_body_a = _zeros((dims.N_w, dims.N_n), wp.int32)
        contact_body_b = _zeros((dims.N_w, dims.N_n), wp.int32)
        contact_point_a = _zeros((dims.N_w, dims.N_n), wp.vec3)
        contact_point_b = _zeros((dims.N_w, dims.N_n), wp.vec3)
        contact_thickness_a = _zeros((dims.N_w, dims.N_n), wp.float32)
        contact_thickness_b = _zeros((dims.N_w, dims.N_n), wp.float32)
        contact_dist = _zeros((dims.N_w, dims.N_n), wp.float32)
        contact_friction_coeff = _zeros((dims.N_w, dims.N_n), wp.float32)
        contact_restitution_coeff = _zeros((dims.N_w, dims.N_n), wp.float32)
        contact_basis_n_a = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)
        contact_basis_t1_a = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)
        contact_basis_t2_a = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)
        contact_basis_n_b = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)
        contact_basis_t1_b = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)
        contact_basis_t2_b = _zeros((dims.N_w, dims.N_n), wp.spatial_vector)

        joint_target = _zeros((dims.N_w, dof_count))

        # ---- Linesearch Arrays ----
        linesearch_data = None
        if config.enable_linesearch:
            # Calculate total budget
            step_count = (
                config.linesearch_conservative_step_count + config.linesearch_optimistic_step_count
            )

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

            # Crucial: Force exact 1.0 to ensure the standard Newton step is tested
            closest_idx = np.argmin(np.abs(ls_steps_np - 1.0))
            ls_steps_np[closest_idx] = 1.0

            linesearch_steps = wp.from_numpy(ls_steps_np, dtype=wp.float32)

            linesearch_batch_body_u = _zeros((step_count, dims.N_w, dims.N_b), wp.spatial_vector)
            linesearch_batch_body_q = _zeros((step_count, dims.N_w, dims.N_b), wp.transform)
            linesearch_batch_body_lambda = _zeros((step_count, dims.N_w, dims.N_c))

            # Separate Allocations
            linesearch_batch_h = _zeros((step_count, dims.N_w, dims.N_u + dims.N_c))
            ls_h_spatial = _zeros((step_count, dims.N_w, dims.N_b), dtype=wp.spatial_vector)

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
        newton_history_data = None
        if allocate_history:
            # Separate Allocations
            h_history = _zeros((config.max_newton_iters + 1, dims.N_w, dims.N_u + dims.N_c))
            h_history_spatial = _zeros(
                (config.max_newton_iters + 1, dims.N_w, dims.N_b), dtype=wp.spatial_vector
            )

            body_q_history = _zeros(
                (config.max_newton_iters + 1, dims.N_w, dims.N_b), dtype=wp.transform
            )
            body_u_history = _zeros(
                (config.max_newton_iters + 1, dims.N_w, dims.N_b), dtype=wp.spatial_vector
            )
            body_lambda_history = _zeros((config.max_newton_iters + 1, dims.N_w, dims.N_c))

            newton_history_data = NewtonHistoryData(
                dims=dims,
                _h_history=h_history,
                body_q_history=body_q_history,
                body_u_history=body_u_history,
                _body_lambda_history=body_lambda_history,
                _h_history_spatial=h_history_spatial,
            )

        return EngineData(
            dims=dims,
            device=device,
            dt=None,
            _h=h,
            _h_spatial=h_spatial,  # Pass buffer
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
            contact_body_a=contact_body_a,
            contact_body_b=contact_body_b,
            contact_point_a=contact_point_a,
            contact_point_b=contact_point_b,
            contact_thickness_a=contact_thickness_a,
            contact_thickness_b=contact_thickness_b,
            contact_dist=contact_dist,
            contact_friction_coeff=contact_friction_coeff,
            contact_restitution_coeff=contact_restitution_coeff,
            contact_basis_n_a=contact_basis_n_a,
            contact_basis_t1_a=contact_basis_t1_a,
            contact_basis_t2_a=contact_basis_t2_a,
            contact_basis_n_b=contact_basis_n_b,
            contact_basis_t1_b=contact_basis_t1_b,
            contact_basis_t2_b=contact_basis_t2_b,
            joint_constraint_offsets=joint_constraint_offsets,
            control_constraint_offsets=control_constraint_offsets,
            joint_target=joint_target,
            linesearch=linesearch_data,
            newton_history=newton_history_data,
        )
