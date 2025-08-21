from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar


@dataclass(frozen=True)
class ConstraintDimensions:
    """
    Calculates and stores the dimensions and offsets for the physics simulation.
    This provides a single source of truth for the size and layout of all system
    vectors and matrices, with helper properties for easy slicing.
    """

    # --- Primary Inputs ---
    N_b: int  # Number of bodies
    N_c: int  # Number of potential contacts
    N_j: int  # Number of joints

    # --- Derived Total Dimensions ---
    # Dimension of the dynamics part (e.g., body velocities)
    dyn_dim: int = field(init=False)
    # Total dimension of the constraint part
    con_dim: int = field(init=False)
    # Total dimension of the residual (dyn_dim + con_dim)
    res_dim: int = field(init=False)

    # --- Per-Constraint-Type Dimensions ---
    joint_dim: int = field(init=False)
    normal_dim: int = field(init=False)
    friction_dim: int = field(init=False)

    # --- Per-Constraint-Type Offsets ---
    # These define the start of each block within the constraint vector.
    joint_offset: int = field(init=False)
    normal_offset: int = field(init=False)
    friction_offset: int = field(init=False)

    # --- Constants ---
    # Use a ClassVar for constants that don't change per instance.
    DOF_PER_BODY: ClassVar[int] = 6
    CON_PER_JOINT: ClassVar[int] = 5
    CON_PER_FRICTION: ClassVar[int] = 2

    def __post_init__(self):
        """
        Automatically computes the derived dimensions and offsets.
        Must use object.__setattr__ because the instance is frozen.
        """
        # Calculate per-constraint dimensions first
        joint_d = self.CON_PER_JOINT * self.N_j
        normal_d = self.N_c
        friction_d = self.CON_PER_FRICTION * self.N_c

        # Calculate total dimensions
        dyn_d = self.DOF_PER_BODY * self.N_b
        con_d = joint_d + normal_d + friction_d
        res_d = dyn_d + con_d

        # Calculate the single, authoritative set of offsets
        joint_off = 0
        normal_off = joint_d
        friction_off = joint_d + normal_d

        # Assign all values using object.__setattr__
        object.__setattr__(self, "joint_dim", joint_d)
        object.__setattr__(self, "normal_dim", normal_d)
        object.__setattr__(self, "friction_dim", friction_d)
        object.__setattr__(self, "dyn_dim", dyn_d)
        object.__setattr__(self, "con_dim", con_d)
        object.__setattr__(self, "res_dim", res_d)
        object.__setattr__(self, "joint_offset", joint_off)
        object.__setattr__(self, "normal_offset", normal_off)
        object.__setattr__(self, "friction_offset", friction_off)

    # --- Slicing Helper Properties ---
    @property
    def joint_slice(self) -> slice:
        """Returns a slice object for the joint-constraint block."""
        return slice(self.joint_offset, self.normal_offset)

    @property
    def normal_slice(self) -> slice:
        """Returns a slice object for the normal-constraint block."""
        return slice(self.normal_offset, self.friction_offset)

    @property
    def friction_slice(self) -> slice:
        """Returns a slice object for the friction-constraint block."""
        return slice(self.friction_offset, self.con_dim)
