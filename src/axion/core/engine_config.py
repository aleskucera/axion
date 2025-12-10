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
    linear_iters: int = 8

    joint_stabilization_factor: float = 0.01
    contact_stabilization_factor: float = 0.02

    joint_compliance: float = 1e-5
    contact_compliance: float = 1e-5
    friction_compliance: float = 1e-6

    regularization: float = 1e-6

    contact_fb_alpha: float = 0.25
    contact_fb_beta: float = 0.25
    friction_fb_alpha: float = 1.0
    friction_fb_beta: float = 1.0

    enable_linesearch: bool = False
    linesearch_step_count: int = 200
    linesearch_step_min: float = 1e-6
    linesearch_step_max: float = 10.0

    max_contacts_per_world: int = 20

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
        _validate_non_negative_float(self.joint_compliance, "joint_compliance")
        _validate_non_negative_float(self.contact_compliance, "contact_compliance")
        _validate_non_negative_float(self.friction_compliance, "friction_compliance")

        _validate_non_negative_float(self.regularization, "regularization")

        # Validate Fisher-Burmeister parameters (should be in [0, 1])
        _validate_unit_interval(self.contact_fb_alpha, "contact_fb_alpha")
        _validate_unit_interval(self.contact_fb_beta, "contact_fb_beta")
        _validate_unit_interval(self.friction_fb_alpha, "friction_fb_alpha")
        _validate_unit_interval(self.friction_fb_beta, "friction_fb_beta")

        # Validate linesearch steps
        _validate_positive_int(self.linesearch_step_count, "linesearch_step_count")
        _validate_non_negative_float(self.linesearch_step_min, "linesearch_step_min")
        _validate_non_negative_float(self.linesearch_step_max, "linesearch_step_max")

        _validate_positive_int(self.max_contacts_per_world, "max_contacts_per_world")

        if self.linesearch_step_min >= self.linesearch_step_max:
            raise ValueError("linesearch_step_min must be < linesearch_step_max")


@dataclass(frozen=True)
class FeatherstoneEngineConfig(EngineConfig):
    angular_damping: float = 0.05
    update_mass_matrix_interval: int = 1
    friction_smoothing: float = 1.0
    use_tile_gemm: bool = False
    fuse_cholesky: bool = True


@dataclass(frozen=True)
class MuJoCoEngineConfig(EngineConfig):
    separate_worlds: bool | None = None
    njmax: int | None = None
    nconmax: int | None = None
    iterations: int = 20
    ls_iterations: int = 10
    solver: int | str = "cg"
    integrator: int | str = "euler"
    cone: int | str = "pyramidal"
    impratio: float = 1.0
    use_mujoco_cpu: bool = False
    disable_contacts: bool = False
    default_actuator_gear: float | None = None
    actuator_gears: dict[str, float] | None = None
    update_data_interval: int = 1
    save_to_mjcf: str | None = None
    ls_parallel: bool = False
    use_mujoco_contacts: bool = True
    joint_solimp_limit: tuple[float, float, float, float, float] | None = None


@dataclass(frozen=True)
class XPBDEngineConfig(EngineConfig):
    iterations: int = 2
    soft_body_relaxation: float = 0.9
    soft_contact_relaxation: float = 0.9
    joint_linear_relaxation: float = 0.7
    joint_angular_relaxation: float = 0.4
    rigid_contact_relaxation: float = 0.8
    joint_linear_compliance: float = 0.01
    joint_angular_compliance: float = 0.01
    rigid_contact_con_weighting: bool = True
    angular_damping: float = 0.0
    enable_restitution: bool = False
