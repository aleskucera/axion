from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import warp as wp
import warp.context as wpc
from axion.constraints import update_constraint_body_idx_kernel
from axion.types import assemble_spatial_inertia_kernel
from axion.types import contact_interaction_kernel
from axion.types import ContactInteraction
from axion.types import joint_interaction_kernel
from axion.types import JointInteraction
from axion.types import SpatialInertia
from warp.sim import Model
from warp.sim import State

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

    body_f: wp.array  # External forces
    body_q: wp.array  # Positions
    body_q_prev: wp.array  # Positions at previous time step
    body_qd: wp.array  # Velocities
    body_qd_prev: wp.array  # Velocities at previous time step

    _lambda: wp.array  # Constraint impulses
    lambda_prev: wp.array  # Constraint impulses at previous newton step

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
    alpha: wp.array  # Step size
    alphas: wp.array = None  # Alpha values for linesearch
    res_alpha: wp.array = None  # Residuals for different alphas
    res_alpha_norm_sq: wp.array = None  # Squared norms of linesearch residuals

    Hinv_dense: wp.array = None
    J_dense: wp.array = None
    C_dense: wp.array = None

    g_accel: wp.types.vector = None

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
        return self._lambda[self.dims.joint_slice] if self.dims.N_j > 0 else None

    @cached_property
    def lambda_n(self) -> Optional[wp.array]:
        """Normal contact impulses."""
        return self._lambda[self.dims.normal_slice] if self.dims.N_c > 0 else None

    @cached_property
    def lambda_f(self) -> Optional[wp.array]:
        """Friction impulses."""
        return self._lambda[self.dims.friction_slice] if self.dims.N_c > 0 else None

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
                self.g_alpha, shape=(self.dims.N_alpha, self.dims.N_b), dtype=wp.spatial_vector
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
        return self.dims.N_alpha > 0

    def set_generalized_mass(self, model: Model):
        wp.launch(
            kernel=assemble_spatial_inertia_kernel,
            dim=self.dims.N_b,
            inputs=[
                model.body_mass,
                model.body_inertia,
            ],
            outputs=[self.gen_mass],
        )

        wp.launch(
            kernel=assemble_spatial_inertia_kernel,
            dim=self.dims.N_b,
            inputs=[
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[self.gen_inv_mass],
        )

    def set_gravitational_acceleration(self, model: Model):
        assert (
            self.g_accel is None
        ), "Setting gravitational acceleration more than once is forbidden"
        object.__setattr__(self, "g_accel", model.gravity)

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

    def update_state_data(self, model: Model, state_in: State, state_out: State):
        wp.copy(dest=self.body_f, src=state_in.body_f)
        wp.copy(dest=self.body_q, src=state_out.body_q)
        wp.copy(dest=self.body_q_prev, src=state_in.body_q)
        wp.copy(dest=self.body_qd, src=state_out.body_qd)
        wp.copy(dest=self.body_qd_prev, src=state_in.body_qd)

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=self.dims.N_c,
            inputs=[
                self.body_q,
                model.body_com,
                model.shape_body,
                model.shape_geo,
                model.shape_materials,
                model.rigid_contact_count,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
            ],
            outputs=[
                self.contact_interaction,
            ],
            device=self.device,
        )

        # wp.launch(
        #     kernel=joint_interaction_kernel,
        #     dim=self.dims.N_j,
        #     inputs=[
        #         self.body_q,
        #         model.body_com,
        #         model.joint_type,
        #         model.joint_enabled,
        #         model.joint_parent,
        #         model.joint_child,
        #         model.joint_X_p,
        #         model.joint_X_c,
        #         model.joint_axis_start,
        #         model.joint_axis,
        #         model.joint_linear_compliance,
        #         model.joint_angular_compliance,
        #     ],
        #     outputs=[
        #         self.joint_interaction,
        #     ],
        #     device=self.device,
        # )

        wp.launch(
            kernel=update_constraint_body_idx_kernel,
            dim=self.dims.con_dim,
            inputs=[
                model.shape_body,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.joint_parent,
                model.joint_child,
                self.dims.N_j,
                self.dims.N_c,
            ],
            outputs=[
                self.constraint_body_idx,
            ],
            device=self.device,
        )


def create_engine_arrays(
    dims: EngineDimensions,
    device: wpc.Device,
    allocate_dense: bool = False,
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

    def _ones(shape, dtype=wp.float32, ndim=None):
        return wp.ones(shape, dtype=dtype, device=device, ndim=ndim)

    def _empty(shape, dtype, device=device):
        return wp.empty(shape, dtype=dtype, device=device)

    # Allocate core arrays
    res = _zeros(dims.res_dim)
    J_values = _zeros((dims.con_dim, 2), wp.spatial_vector)
    C_values = _zeros(dims.con_dim)

    body_f = _zeros(dims.N_b, wp.spatial_vector)
    body_q = _zeros(dims.N_b, wp.transform)
    body_q_prev = _zeros(dims.N_b, wp.transform)
    body_qd = _zeros(dims.N_b, wp.spatial_vector)
    body_qd_prev = _zeros(dims.N_b, wp.spatial_vector)

    _lambda = _zeros(dims.con_dim)
    lambda_prev = _zeros(dims.con_dim)

    constraint_body_idx = _zeros((dims.con_dim, 2), wp.int32)

    JT_delta_lambda = _zeros(dims.N_b, wp.spatial_vector)
    delta_body_qd = _zeros(dims.dyn_dim)
    delta_lambda = _zeros(dims.con_dim)
    b = _zeros(dims.con_dim)

    gen_mass = _empty(dims.N_b, SpatialInertia)
    gen_inv_mass = _empty(dims.N_b, SpatialInertia)
    joint_interaction = _empty(dims.N_j, JointInteraction)
    contact_interaction = _empty(dims.N_c, ContactInteraction)

    # --- Arrays for linesearch ---
    alpha = _ones(1)  # Always set defaultly to one
    alphas_array = None

    res_alpha = None
    res_alpha_norm_sq = None
    if dims.N_alpha > 0:
        res_alpha = _zeros((dims.N_alpha, dims.res_dim))
        res_alpha_norm_sq = _zeros(dims.N_alpha)

        # Create default alpha values if not provide
        alphas = [1.0 / (2**i) for i in range(dims.N_alpha)]
        alphas_array = wp.array(alphas, dtype=wp.float32, device=device)

    # --- Dense representation of the arrays---
    Hinv_dense = None
    J_dense = None
    C_dense = None
    if allocate_dense:
        Hinv_dense = _zeros((dims.dyn_dim, dims.dyn_dim))
        J_dense = _zeros((dims.con_dim, dims.dyn_dim))
        C_dense = _zeros((dims.con_dim, dims.con_dim))

    return EngineArrays(
        dims=dims,
        device=device,
        res=res,
        J_values=J_values,
        C_values=C_values,
        body_f=body_f,
        body_q=body_q,
        body_q_prev=body_q_prev,
        body_qd=body_qd,
        body_qd_prev=body_qd_prev,
        _lambda=_lambda,
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
        alpha=alpha,
        alphas=alphas_array,
        res_alpha=res_alpha,
        res_alpha_norm_sq=res_alpha_norm_sq,
        Hinv_dense=Hinv_dense,
        J_dense=J_dense,
        C_dense=C_dense,
    )
