from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    pass


@dataclass(frozen=True)
class AxionEngineConfig(EngineConfig):
    """
    Configuration parameters for the AxionEngine solver.

    This object centralizes all tunable parameters for the physics simulation,
    including solver iterations, stabilization factors, and compliance values.
    Making it a frozen dataclass ensures that configuration is immutable
    during a simulation run.
    """

    newton_iters: int = 8
    linear_iters: int = 4

    joint_stabilization_factor: float = 0.01
    contact_stabilization_factor: float = 0.1
    contact_compliance: float = 1e-4
    friction_compliance: float = 1e-6

    contact_fb_alpha: float = 0.25
    contact_fb_beta: float = 0.25
    friction_fb_alpha: float = 0.25
    friction_fb_beta: float = 0.25

    linesearch_steps: int = 0

    matrixfree_representation: bool = True

    def __post_init__(self):
        """Validate all configuration parameters."""

        def _validate_positive_int(value: int, name: str, min_value: int = 1) -> None:
            """Validate that a value is a positive integer >= min_value."""
            if value < min_value:
                raise ValueError(f"{name} must be >= {min_value}, got {value}")

        def _validate_non_negative_float(value: float, name: str) -> None:
            """Validate that a value is a non-negative float."""
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        def _validate_unit_interval(value: float, name: str) -> None:
            """Validate that a value is in the unit interval [0, 1]."""
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        def _validate_non_negative_int(value: int, name: str) -> None:
            """Validate that a value is a non-negative integer."""
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        # Validate iteration counts
        _validate_positive_int(self.newton_iters, "newton_iters")
        _validate_positive_int(self.linear_iters, "linear_iters")

        # Validate non-negative values
        _validate_non_negative_float(self.joint_stabilization_factor, "joint_stabilization_factor")
        _validate_non_negative_float(
            self.contact_stabilization_factor, "contact_stabilization_factor"
        )
        _validate_non_negative_float(self.contact_compliance, "contact_compliance")
        _validate_non_negative_float(self.friction_compliance, "friction_compliance")

        # Validate feedback parameters (should be in [0, 1])
        _validate_unit_interval(self.contact_fb_alpha, "contact_fb_alpha")
        _validate_unit_interval(self.contact_fb_beta, "contact_fb_beta")
        _validate_unit_interval(self.friction_fb_alpha, "friction_fb_alpha")
        _validate_unit_interval(self.friction_fb_beta, "friction_fb_beta")

        # Validate linesearch steps
        _validate_non_negative_int(self.linesearch_steps, "linesearch_steps")


@dataclass(frozen=True)
class FeatherstoneEngineConfig(EngineConfig):
    angular_damping: float = 0.05
    update_mass_matrix_every: int = 1
    friction_smoothing: float = 1.0
    use_tile_gemm: bool = False
    fuse_cholesky: bool = True


@dataclass(frozen=True)
class SemiImplicitEngineConfig(EngineConfig):
    angular_damping: float = 0.05
    friction_smoothing: float = 1.0


@dataclass(frozen=True)
class XPBDEngineConfig(EngineConfig):
    iterations: int = 2
    soft_body_relaxation: float = 0.9
    soft_contact_relaxation: float = 0.9
    joint_linear_relaxation: float = 0.7
    joint_angular_relaxation: float = 0.4
    rigid_contact_relaxation: float = 0.8
    rigid_contact_con_weighting: bool = True
    angular_damping: float = 0.0
    enable_restitution: bool = False
