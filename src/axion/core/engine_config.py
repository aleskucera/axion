from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class EngineConfig:
    """
    Configuration parameters for the AxionEngine solver.

    This object centralizes all tunable parameters for the physics simulation,
    including solver iterations, stabilization factors, and compliance values.
    Making it a frozen dataclass ensures that configuration is immutable
    during a simulation run.
    """

    # Solver iteration counts
    newton_iters: int = 8
    linear_iters: int = 4

    # Stabilization and compliance
    joint_stabilization_factor: float = 0.01
    contact_stabilization_factor: float = 0.1
    contact_compliance: float = 1e-5
    friction_compliance: float = 1e-5

    # Feedback (Baumgarte) terms
    contact_fb_alpha: float = 0.25
    contact_fb_beta: float = 0.25
    friction_fb_alpha: float = 0.25
    friction_fb_beta: float = 0.25

    # Solver representation
    matrixfree_representation: bool = True

    # Linesearch parameters
    linesearch_alphas: tuple[float, ...] = field(
        default_factory=lambda: (4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.05)
    )
