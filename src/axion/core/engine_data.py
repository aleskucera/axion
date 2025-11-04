from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import warp as wp
import warp.context as wpc
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.types import contact_interaction_kernel
from axion.types import ContactInteraction
from axion.types import joint_constraint_data_kernel
from axion.types import JointConstraintData
from axion.types import spatial_inertia_kernel
from axion.types import SpatialInertia
from newton import Contacts
from newton import Model
from newton import State

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
    h: wp.array  # Residual
    J_values: wp.array  # Jacobian values for sparse construction
    C_values: wp.array  # Compliance values

    body_f: wp.array  # External forces
    body_q: wp.array  # Positions
    body_q_prev: wp.array  # Positions at previous time step
    body_u: wp.array  # Velocities
    body_u_prev: wp.array  # Velocities at previous time step

    body_lambda: wp.array  # Constraint impulses
    body_lambda_prev: wp.array  # Constraint impulses at previous newton step

    s_n: wp.array  # Scale for normal impulse
    s_n_prev: wp.array  # Scale for normal impulse at previous newton step

    constraint_body_idx: wp.array  # Indices of bodies involved in constraints

    JT_delta_lambda: wp.array  # J^T * delta_body_lambda
    dbody_u: wp.array  # Change in body velocities
    dbody_lambda: wp.array  # Change in constraint impulses
    b: wp.array  # RHS vector for linear system

    body_M: wp.array
    body_M_inv: wp.array
    joint_constraint_data: wp.array
    joint_constraint_offsets: wp.array
    contact_interaction: wp.array

    # --- Linesearch specific arrays (always allocated, may be empty) ---
    alpha: wp.array  # Step size
    alphas: wp.array = None  # Alpha values for linesearch
    h_alpha: wp.array = None  # Residuals for different alphas
    h_alpha_norm_sq: wp.array = None  # Squared norms of linesearch residuals

    M_inv_dense: wp.array = None
    J_dense: wp.array = None
    C_dense: wp.array = None

    g_accel: wp.array = None

    @cached_property
    def h_d(self) -> wp.array:
        return self.h[: self.dims.N_u]

    @cached_property
    def h_d_v(self) -> wp.array:
        return wp.array(self.h_d, shape=self.dims.N_b, dtype=wp.spatial_vector)

    @cached_property
    def h_c(self) -> wp.array:
        return self.h[self.dims.N_u :]

    @cached_property
    def h_j(self) -> Optional[wp.array]:
        """Joint constraint residuals."""
        return self.h_c[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def h_n(self) -> Optional[wp.array]:
        """Normal contact constraint residuals."""
        return self.h_c[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def h_f(self) -> Optional[wp.array]:
        """Friction constraint residuals."""
        return self.h_c[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def J_j_values(self) -> Optional[wp.array]:
        """Joint Jacobian values."""
        return self.J_values[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def J_n_values(self) -> Optional[wp.array]:
        """Normal contact Jacobian values."""
        return self.J_values[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def J_f_values(self) -> Optional[wp.array]:
        """Friction Jacobian values."""
        return self.J_values[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def C_j_values(self) -> Optional[wp.array]:
        """Joint compliance values."""
        return self.C_values[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def C_n_values(self) -> Optional[wp.array]:
        """Normal contact compliance values."""
        return self.C_values[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def C_f_values(self) -> Optional[wp.array]:
        """Friction compliance values."""
        return self.C_values[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def body_lambda_j(self) -> Optional[wp.array]:
        """Joint constraint impulses."""
        return self.body_lambda[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def body_lambda_n(self) -> Optional[wp.array]:
        """Normal contact impulses."""
        return self.body_lambda[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def body_lambda_f(self) -> Optional[wp.array]:
        """Friction impulses."""
        return self.body_lambda[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def body_lambda_j_prev(self) -> Optional[wp.array]:
        """Previous joint constraint impulses."""
        return self.body_lambda_prev[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def body_lambda_n_prev(self) -> Optional[wp.array]:
        """Previous normal contact impulses."""
        return self.body_lambda_prev[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def body_lambda_f_prev(self) -> Optional[wp.array]:
        """Previous friction impulses."""
        return self.body_lambda_prev[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def dbody_u_v(self) -> wp.array:
        """Change in body velocities as spatial vectors."""
        return wp.array(self.dbody_u, shape=self.dims.N_b, dtype=wp.spatial_vector)

    @cached_property
    def dbody_lambda_j(self) -> Optional[wp.array]:
        """Change in joint impulses."""
        return self.dbody_lambda[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def dbody_lambda_n(self) -> Optional[wp.array]:
        """Change in normal contact impulses."""
        return self.dbody_lambda[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def dbody_lambda_f(self) -> Optional[wp.array]:
        """Change in friction impulses."""
        return self.dbody_lambda[self.dims.slice_f] if self.dims.N_f > 0 else None

    @cached_property
    def joint_constraint_body_idx(self) -> Optional[wp.array]:
        """Previous friction impulses."""
        return self.constraint_body_idx[self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def contact_constraint_body_idx(self) -> Optional[wp.array]:
        """Previous friction impulses."""
        return self.constraint_body_idx[self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def friction_constraint_body_idx(self) -> Optional[wp.array]:
        """Previous friction impulses."""
        return self.constraint_body_idx[self.dims.slice_f] if self.dims.N_f > 0 else None

    # --- Linesearch views (always available, may be empty arrays) ---
    @cached_property
    def h_d_alpha(self) -> wp.array:
        """Linesearch residual g for different alphas."""
        if self.has_linesearch:
            return self.h_alpha[:, : self.dims.N_u]

    @cached_property
    def h_d_alpha_v(self) -> wp.array:
        """Linesearch residual g as spatial vectors."""
        if self.has_linesearch:
            return wp.array(
                self.h_d_alpha, shape=(self.dims.N_alpha, self.dims.N_b), dtype=wp.spatial_vector
            )

    @cached_property
    def h_c_alpha(self) -> wp.array:
        """Linesearch residual h for different alphas."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.N_u :]

    @cached_property
    def h_j_alpha(self) -> Optional[wp.array]:
        """Linesearch joint constraint residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.slice_j] if self.dims.N_j > 0 else None

    @cached_property
    def h_n_alpha(self) -> Optional[wp.array]:
        """Linesearch normal contact residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.slice_n] if self.dims.N_n > 0 else None

    @cached_property
    def h_f_alpha(self) -> Optional[wp.array]:
        """Linesearch friction residuals."""
        if self.has_linesearch:
            return self.h_alpha[:, self.dims.slice_f] if self.dims.N_f > 0 else None

    @property
    def has_linesearch(self) -> bool:
        """Returns True if linesearch arrays are allocated (N_alpha > 0)."""
        return self.dims.N_alpha > 0

    def set_body_M(self, model: Model):
        wp.launch(
            kernel=spatial_inertia_kernel,
            dim=self.dims.N_b,
            inputs=[
                model.body_mass,
                model.body_inertia,
            ],
            outputs=[self.body_M],
        )

        wp.launch(
            kernel=spatial_inertia_kernel,
            dim=self.dims.N_b,
            inputs=[
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[self.body_M_inv],
        )

    def set_g_accel(self, model: Model):
        assert (
            self.g_accel is None
        ), "Setting gravitational acceleration more than once is forbidden"
        object.__setattr__(self, "g_accel", model.gravity)

    def update_state_data(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        contacts: Contacts,
    ):
        wp.copy(dest=self.body_f, src=state_in.body_f)
        wp.copy(dest=self.body_q, src=state_out.body_q)
        wp.copy(dest=self.body_q_prev, src=state_in.body_q)
        wp.copy(dest=self.body_u, src=state_out.body_qd)
        wp.copy(dest=self.body_u_prev, src=state_in.body_qd)

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=model.rigid_contact_max,
            inputs=[
                self.body_q,
                model.body_com,
                model.shape_body,
                model.shape_thickness,
                model.shape_material_mu,
                model.shape_material_restitution,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
            ],
            outputs=[
                self.contact_interaction,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=joint_constraint_data_kernel,
            dim=model.joint_count,
            inputs=[
                self.body_q,
                model.body_com,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_axis,
                model.joint_limit_lower,
                model.joint_limit_upper,
                self.joint_constraint_offsets,
            ],
            outputs=[
                self.joint_constraint_data,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_joint_constraint_body_idx_kernel,
            dim=self.dims.N_j,
            inputs=[
                self.joint_constraint_data,
            ],
            outputs=[
                self.joint_constraint_body_idx,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_contact_constraint_body_idx_kernel,
            dim=self.dims.N_n,
            inputs=[
                self.contact_interaction,
            ],
            outputs=[
                self.contact_constraint_body_idx,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_friction_constraint_body_idx_kernel,
            dim=self.dims.N_f,
            inputs=[
                self.contact_interaction,
            ],
            outputs=[
                self.friction_constraint_body_idx,
            ],
            device=self.device,
        )

    def clear_working_buffers(self):
        """Clears non-persistent, working arrays."""
        self.h.zero_()
        self.J_values.zero_()
        self.C_values.zero_()
        self.JT_delta_lambda.zero_()
        self.b.zero_()
        self.dbody_u.zero_()
        self.dbody_lambda.zero_()

        if self.has_linesearch:
            self.h_alpha.zero_()
            self.h_alpha_norm_sq.zero_()
            self.best_alpha_idx.zero_()


def create_engine_arrays(
    dims: EngineDimensions,
    joint_constraint_offsets: wp.array,
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
    h = _zeros(dims.N_u + dims.N_c)
    J_values = _zeros((dims.N_c, 2), wp.spatial_vector)
    C_values = _zeros(dims.N_c)

    body_f = _zeros(dims.N_b, wp.spatial_vector)
    body_q = _zeros(dims.N_b, wp.transform)
    body_q_prev = _zeros(dims.N_b, wp.transform)
    body_u = _zeros(dims.N_b, wp.spatial_vector)
    body_u_prev = _zeros(dims.N_b, wp.spatial_vector)

    body_lambda = _zeros(dims.N_c)
    body_lambda_prev = _zeros(dims.N_c)

    s_n = _zeros(dims.N_n)
    s_n_prev = _zeros(dims.N_n)

    constraint_body_idx = _zeros((dims.N_c, 2), wp.int32)

    JT_delta_lambda = _zeros(dims.N_b, wp.spatial_vector)
    dbody_u = _zeros(dims.N_u)
    dbody_lambda = _zeros(dims.N_c)

    b = _zeros(dims.N_c)

    body_M = _empty(dims.N_b, SpatialInertia)
    body_M_inv = _empty(dims.N_b, SpatialInertia)

    joint_constraint_data = _empty(dims.N_j, JointConstraintData)
    contact_interaction = _empty(dims.N_n, ContactInteraction)

    # --- Arrays for linesearch ---
    alpha = _ones(1)  # Always set defaultly to one
    alphas_array = None

    res_alpha = None
    res_alpha_norm_sq = None
    if dims.N_alpha > 0:
        res_alpha = _zeros((dims.N_alpha, dims.N_u + dims.N_c))
        res_alpha_norm_sq = _zeros(dims.N_alpha)

        # Create default alpha values if not provide
        alphas = [1.0 / (2**i) for i in range(dims.N_alpha)]
        alphas_array = wp.array(alphas, dtype=wp.float32, device=device)

    # --- Dense representation of the arrays---
    M_inv_dense = None
    J_dense = None
    C_dense = None
    if allocate_dense:
        M_inv_dense = _zeros((dims.N_u, dims.N_u))
        J_dense = _zeros((dims.N_c, dims.N_u))
        C_dense = _zeros((dims.N_c, dims.N_c))

    return EngineArrays(
        dims=dims,
        device=device,
        h=h,
        J_values=J_values,
        C_values=C_values,
        body_f=body_f,
        body_q=body_q,
        body_q_prev=body_q_prev,
        body_u=body_u,
        body_u_prev=body_u_prev,
        body_lambda=body_lambda,
        body_lambda_prev=body_lambda_prev,
        s_n=s_n,
        s_n_prev=s_n_prev,
        constraint_body_idx=constraint_body_idx,
        JT_delta_lambda=JT_delta_lambda,
        dbody_u=dbody_u,
        dbody_lambda=dbody_lambda,
        b=b,
        body_M=body_M,
        body_M_inv=body_M_inv,
        joint_constraint_data=joint_constraint_data,
        joint_constraint_offsets=joint_constraint_offsets,
        contact_interaction=contact_interaction,
        alpha=alpha,
        alphas=alphas_array,
        h_alpha=res_alpha,
        h_alpha_norm_sq=res_alpha_norm_sq,
        M_inv_dense=M_inv_dense,
        J_dense=J_dense,
        C_dense=C_dense,
    )
