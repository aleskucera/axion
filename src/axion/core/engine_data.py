from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from typing import Tuple

import warp as wp
import warp.context as wpc
from axion.types import ContactInteraction
from axion.types import GeneralizedMass
from axion.types import JointInteraction

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

    # --- Primary allocated arrays ---
    res: wp.array  # Residual: [g, h]
    J_values: wp.array  # Jacobian values for sparse construction
    C_values: wp.array  # Compliance values
    lambda_vec: wp.array  # Constraint impulses
    lambda_prev: wp.array  # Previous constraint impulses
    constraint_body_idx: wp.array  # Indices of bodies involved in constraints

    JT_delta_lambda: wp.array  # J^T * delta_lambda
    delta_body_qd: wp.array  # Change in body velocities (full vector)
    delta_lambda: wp.array  # Change in constraint impulses
    b: wp.array  # RHS vector for linear system

    gen_mass: wp.array  # Generalized mass properties
    gen_inv_mass: wp.array  # Inverse generalized mass properties
    joint_interaction: wp.array  # Joint interaction data
    contact_interaction: wp.array  # Contact interaction data

    # --- Linesearch specific arrays (always allocated, may be empty) ---
    N_alpha: int = 0
    res_alpha: wp.array = None  # Residuals for different alphas
    res_alpha_norm_sq: wp.array = None  # Squared norms of linesearch residuals
    best_alpha_idx: wp.array = None  # Index of best alpha
    alphas: wp.array = None  # Alpha values for linesearch

    # --- Core views ---
    @cached_property
    def g(self) -> wp.array:
        """Residual vector g (dynamics part)."""
        return self.res[: self.dims.dyn_dim]

    @cached_property
    def g_v(self) -> wp.array:
        """Residual vector g as spatial vectors."""
        return wp.array(self.g, shape=self.dims.N_b, dtype=wp.spatial_vector)

    @cached_property
    def h(self) -> wp.array:
        """Residual vector h (constraints part)."""
        return self.res[self.dims.dyn_dim :]

    @cached_property
    def delta_body_qd_v(self) -> wp.array:
        """Change in body velocities as spatial vectors."""
        return wp.array(self.delta_body_qd, shape=self.dims.N_b, dtype=wp.spatial_vector)

    # --- Constraint-specific views ---
    @cached_property
    def h_j(self) -> Optional[wp.array]:
        """Joint constraint residuals."""
        return self.h[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def h_n(self) -> Optional[wp.array]:
        """Normal contact constraint residuals."""
        return self.h[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def h_f(self) -> Optional[wp.array]:
        """Friction constraint residuals."""
        return self.h[self.dims.friction_slice] if self.dims.N_c > 0 else None

    @cached_property
    def J_j_values(self) -> Optional[wp.array]:
        """Joint Jacobian values."""
        return self.J_values[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def J_n_values(self) -> Optional[wp.array]:
        """Normal contact Jacobian values."""
        return self.J_values[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def J_f_values(self) -> Optional[wp.array]:
        """Friction Jacobian values."""
        return self.J_values[self.dims.friction_slice] if self.dims.N_c > 0 else None

    @cached_property
    def C_j_values(self) -> Optional[wp.array]:
        """Joint compliance values."""
        return self.C_values[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def C_n_values(self) -> Optional[wp.array]:
        """Normal contact compliance values."""
        return self.C_values[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def C_f_values(self) -> Optional[wp.array]:
        """Friction compliance values."""
        return self.C_values[self.dims.friction_slice] if self.dims.N_c > 0 else None

    @cached_property
    def lambda_j(self) -> Optional[wp.array]:
        """Joint constraint impulses."""
        return self.lambda_vec[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def lambda_n(self) -> Optional[wp.array]:
        """Normal contact impulses."""
        return self.lambda_vec[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def lambda_f(self) -> Optional[wp.array]:
        """Friction impulses."""
        return self.lambda_vec[self.dims.friction_slice] if self.dims.N_c > 0 else None

    @cached_property
    def lambda_j_prev(self) -> Optional[wp.array]:
        """Previous joint constraint impulses."""
        return self.lambda_prev[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def lambda_n_prev(self) -> Optional[wp.array]:
        """Previous normal contact impulses."""
        return self.lambda_prev[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def lambda_f_prev(self) -> Optional[wp.array]:
        """Previous friction impulses."""
        return self.lambda_prev[self.dims.friction_slice] if self.dims.N_c > 0 else None

    @cached_property
    def delta_lambda_j(self) -> Optional[wp.array]:
        """Change in joint impulses."""
        return self.delta_lambda[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def delta_lambda_n(self) -> Optional[wp.array]:
        """Change in normal contact impulses."""
        return self.delta_lambda[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def delta_lambda_f(self) -> Optional[wp.array]:
        """Change in friction impulses."""
        return self.delta_lambda[self.dims.friction_slice] if self.dims.N_c > 0 else None

    # --- Linesearch views (always available, may be empty arrays) ---
    @cached_property
    def g_alpha(self) -> wp.array:
        """Linesearch residual g for different alphas."""
        if self.has_linesearch:
            return self.res_alpha[:, : self.dims.dyn_dim]

    @cached_property
    def g_alpha_v(self) -> wp.array:
        """Linesearch residual g as spatial vectors."""
        if self.has_linesearch:
            return wp.array(
                self.g_alpha, shape=(self.N_alpha, self.dims.N_b), dtype=wp.spatial_vector
            )

    @cached_property
    def h_alpha(self) -> wp.array:
        """Linesearch residual h for different alphas."""
        if self.has_linesearch:
            return self.res_alpha[:, self.dims.dyn_dim :]

    @cached_property
    def h_alpha_j(self) -> Optional[wp.array]:
        """Linesearch joint constraint residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def h_alpha_n(self) -> Optional[wp.array]:
        """Linesearch normal contact residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def h_alpha_f(self) -> Optional[wp.array]:
        """Linesearch friction residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.friction_slice] if self.dims.N_c > 0 else None

    @property
    def has_linesearch(self) -> bool:
        """Returns True if linesearch arrays are allocated (N_alpha > 0)."""
        return self.N_alpha > 0

    def clear_working_buffers(self):
        """Clears non-persistent, working arrays."""
        self.res.zero_()
        self.J_values.zero_()
        self.C_values.zero_()
        self.JT_delta_lambda.zero_()
        self.b.zero_()
        self.delta_body_qd.zero_()
        self.delta_lambda.zero_()

        if self.has_linesearch:
            self.res_alpha.zero_()
            self.res_alpha_norm_sq.zero_()
            self.best_alpha_idx.zero_()


def create_engine_arrays(
    dims: EngineDimensions,
    device: wpc.Device,
    alphas: Tuple[float] = None,
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
        return wp.zeros(shape, dtype=dtype, device=device, ndim=ndim)

    def _empty(shape, dtype, device=device):
        return wp.empty(shape, dtype=dtype, device=device)

    # Allocate core arrays
    res = _zeros(dims.res_dim)
    J_values = _zeros((dims.con_dim, 2), wp.spatial_vector)
    C_values = _zeros(dims.con_dim)
    lambda_vec = _zeros(dims.con_dim)
    lambda_prev = _zeros(dims.con_dim)
    constraint_body_idx = _zeros((dims.con_dim, 2), wp.int32)

    JT_delta_lambda = _zeros(dims.N_b, wp.spatial_vector)
    delta_body_qd = _zeros(dims.dyn_dim)
    delta_lambda = _zeros(dims.con_dim)
    b = _zeros(dims.con_dim)

    gen_mass = _empty(dims.N_b, GeneralizedMass)
    gen_inv_mass = _empty(dims.N_b, GeneralizedMass)
    joint_interaction = _empty(dims.N_j, JointInteraction)
    contact_interaction = _empty(dims.N_c, ContactInteraction)

    # Allocate linesearch arrays (None if N_alpha=0)
    N_alpha = len(alphas)

    alphas_array = None
    best_alpha_idx = None

    res_alpha = None
    res_alpha_norm_sq = None
    if N_alpha > 0:
        res_alpha = _zeros((N_alpha, dims.res_dim))
        res_alpha_norm_sq = _zeros(N_alpha)
        best_alpha_idx = _zeros(1, wp.uint32)

        # Create default alpha values if not provide
        alphas_array = wp.array(alphas, dtype=wp.float32, device=device)

    return EngineArrays(
        dims=dims,
        device=device,
        res=res,
        J_values=J_values,
        C_values=C_values,
        lambda_vec=lambda_vec,
        lambda_prev=lambda_prev,
        constraint_body_idx=constraint_body_idx,
        JT_delta_lambda=JT_delta_lambda,
        delta_body_qd=delta_body_qd,
        delta_lambda=delta_lambda,
        b=b,
        gen_mass=gen_mass,
        gen_inv_mass=gen_inv_mass,
        joint_interaction=joint_interaction,
        contact_interaction=contact_interaction,
        N_alpha=N_alpha,
        res_alpha=res_alpha,
        res_alpha_norm_sq=res_alpha_norm_sq,
        best_alpha_idx=best_alpha_idx,
        alphas=alphas_array,
    )
