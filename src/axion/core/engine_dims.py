from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar


@dataclass(frozen=True)
class EngineDimensions:
    """
    Calculates and stores the dimensions and offsets for the physics simulation.
    This provides a single source of truth for the size and layout of all system
    vectors and matrices, with helper properties for easy slicing.
    """

    # --- Primary Inputs ---
    N_b: int  # Number of bodies
    N_c: int  # Number of potential contacts
    N_rj: int  # Number of revolute joints
    N_sj: int   # Number of spherical joints
    N_j: int  # Total number of joints
    N_alpha: int  # Number of linesearch steps

    # --- Constants ---
    DOF_PER_BODY: ClassVar[int] = 6
    CON_PER_REV_JOINT: ClassVar[int] = 5
    CON_PER_SPH_JOINT: ClassVar[int] = 3
    CON_PER_FRICTION: ClassVar[int] = 2

    # --- Derived Total Dimensions ---
    @cached_property
    def dyn_dim(self) -> int:
        """Dimension of the dynamics part (e.g., body velocities)."""
        return self.DOF_PER_BODY * self.N_b

    @cached_property
    def con_dim(self) -> int:
        """Total dimension of the constraint part."""
        return self.joint_dim + self.normal_dim + self.friction_dim

    @cached_property
    def res_dim(self) -> int:
        """Total dimension of the residual (dyn_dim + con_dim)."""
        return self.dyn_dim + self.con_dim

    # --- Per-Constraint-Type Dimensions ---
    @cached_property
    def joint_dim(self) -> int:
        """Dimension of joint constraints."""
        return self.CON_PER_REV_JOINT * self.N_rj + self.CON_PER_SPH_JOINT * self.N_sj

    @cached_property
    def normal_dim(self) -> int:
        """Dimension of normal contact constraints."""
        return self.N_c

    @cached_property
    def friction_dim(self) -> int:
        """Dimension of friction constraints."""
        return self.CON_PER_FRICTION * self.N_c

    # --- Per-Constraint-Type Offsets ---
    @cached_property
    def revolute_joint_offset(self) -> int:
        """Start offset of revolute joint constraints block."""
        return 0

    @cached_property
    def spherical_joint_offset(self) -> int:
        """Start offset of revolute joint constraints block."""
        return self.CON_PER_REV_JOINT * self.N_rj

    @cached_property
    def normal_offset(self) -> int:
        """Start offset of normal contact constraints block."""
        return self.joint_dim

    @cached_property
    def friction_offset(self) -> int:
        """Start offset of friction constraints block."""
        return self.joint_dim + self.normal_dim

    # --- Slicing Helper Properties ---
    @cached_property
    def revolute_joint_slice(self) -> slice:
        """Returns a slice object for the joint-constraint block."""
        return slice(self.revolute_joint_offset, self.spherical_joint_offset)
    
    @cached_property
    def spherical_joint_slice(self) -> slice:
        """Returns a slice object for the joint-constraint block."""
        return slice(self.spherical_joint_offset, self.normal_offset)

    @cached_property
    def normal_slice(self) -> slice:
        """Returns a slice object for the normal-constraint block."""
        return slice(self.normal_offset, self.friction_offset)

    @cached_property
    def friction_slice(self) -> slice:
        """Returns a slice object for the friction-constraint block."""
        return slice(self.friction_offset, self.con_dim)
