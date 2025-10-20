from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class EngineDimensions:
    """
    Calculates and stores the dimensions and offsets for the physics simulation.
    This provides a single source of truth for the size and layout of all system
    vectors and matrices, with helper properties for easy slicing.
    """

    # --- Primary Inputs ---
    body_count: int
    contact_count: int
    joint_count: int
    linesearch_steps: int
    joint_constraint_count: int

    @cached_property
    def N_b(self) -> int:
        return self.body_count

    @cached_property
    def N_u(self) -> int:
        return 6 * self.body_count

    @cached_property
    def N_q(self) -> int:
        return 7 * self.body_count

    @cached_property
    def N_j(self) -> int:
        return self.joint_constraint_count

    @cached_property
    def N_n(self) -> int:
        return self.contact_count

    @cached_property
    def N_f(self) -> int:
        return 2 * self.contact_count

    @cached_property
    def N_c(self) -> int:
        return self.N_j + self.N_n + self.N_f

    @cached_property
    def N_alpha(self) -> int:
        return self.linesearch_steps

    # --- Derived Total Dimensions ---
    # @cached_property
    # def dyn_dim(self) -> int:
    #     """Dimension of the dynamics part (e.g., body velocities)."""
    #     return 6 * self.N_b

    # @cached_property
    # def con_dim(self) -> int:
    #     """Total dimension of the constraint part."""
    #     return self.joint_dim + self.normal_dim + self.friction_dim

    # @cached_property
    # def res_dim(self) -> int:
    #     """Total dimension of the residual (dyn_dim + con_dim)."""
    #     return self.dyn_dim + self.con_dim

    # --- Per-Constraint-Type Dimensions ---
    # @cached_property
    # def joint_dim(self) -> int:
    #     """Dimension of joint constraints."""
    #     return self.N_j
    #
    # @cached_property
    # def normal_dim(self) -> int:
    #     """Dimension of normal contact constraints."""
    #     return self.N_n
    #
    # @cached_property
    # def friction_dim(self) -> int:
    #     """Dimension of friction constraints."""
    #     return self.N_f

    # --- Per-Constraint-Type Offsets ---
    @cached_property
    def offset_j(self) -> int:
        """Start offset of joint constraints block."""
        return 0

    @cached_property
    def offset_n(self) -> int:
        """Start offset of normal contact constraints block."""
        return self.N_j

    @cached_property
    def offset_f(self) -> int:
        """Start offset of friction constraints block."""
        return self.N_j + self.N_n

    # --- Slicing Helper Properties ---
    @cached_property
    def slice_j(self) -> slice:
        """Returns a slice object for the joint-constraint block."""
        return slice(self.offset_j, self.offset_n)

    @cached_property
    def slice_n(self) -> slice:
        """Returns a slice object for the normal-constraint block."""
        return slice(self.offset_n, self.offset_f)

    @cached_property
    def slice_f(self) -> slice:
        """Returns a slice object for the friction-constraint block."""
        return slice(self.offset_f, self.N_c)
