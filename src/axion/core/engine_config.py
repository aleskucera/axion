from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Optional


@dataclass(frozen=True)
class EngineConfig:
    """
    Base configuration class.
    Defines the factory interface for creating physics engines.
    """

    def create_engine(
        self,
        model: Any,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
    ) -> Any:
        """
        Factory method to create the appropriate solver instance.

        For standard Newton solvers (Featherstone, MuJoCo, etc.), this
        automatically passes all configuration fields as kwargs.

        Note: We intentionally DO NOT pass 'logging_config' to generic solvers
        because they do not support this custom Axion logging system.
        """
        solver_cls = self._get_solver_class()
        return solver_cls(model, **vars(self))

    def _get_solver_class(self):
        """Subclasses must implement this to return their solver class."""
        raise NotImplementedError(
            f"Solver class resolution not implemented for {type(self).__name__}"
        )


@dataclass(frozen=True)
class AxionEngineConfig(EngineConfig):
    """
    Configuration parameters for the AxionEngine solver.

    This object centralizes all tunable parameters for the physics simulation,
    including solver iterations, stabilization factors, and compliance values.
    Making it a frozen dataclass ensures that configuration is immutable
    during a simulation run.
    """

    # --- Physics Parameters ---
    max_newton_iters: int = 8
    max_linear_iters: int = 16

    newton_mode: str = "convergence"  # "convergence" / "fixed"
    linear_mode: str = "convergence"  # "convergence" / "fixed"

    newton_tol: float = 1e-2
    newton_atol: float = 5e-2

    linear_tol: float = 1e-5
    linear_atol: float = 1e-5

    joint_stabilization_factor: float = 0.01
    contact_stabilization_factor: float = 0.02

    joint_compliance: float = 1e-5
    equality_compliance: float = 1e-8
    contact_compliance: float = 1e-6
    friction_compliance: float = 1e-6

    regularization: float = 1e-6

    contact_fb_alpha: float = 1.0
    contact_fb_beta: float = 1.0
    friction_fb_alpha: float = 1.0
    friction_fb_beta: float = 1.0

    enable_linesearch: bool = False

    # --- 1. Conservative Cluster (Safety first) ---
    linesearch_conservative_step_count: int = 32
    linesearch_conservative_upper_bound: float = 0.05
    linesearch_min_step: float = 1e-6

    # --- 2. Optimistic Cluster (The "Attitude") ---
    linesearch_optimistic_step_count: int = 32
    linesearch_optimistic_window: float = 0.2

    max_contacts_per_world: int = 128

    joint_constraint_level: str = "pos"  # pos / vel
    contact_constraint_level: str = "pos"  # pos / vel

    # --- Differentiable Simulation ---
    differentiable_simulation: bool = False
    max_trajectory_steps: int = 0

    # --- Logging & Profiling (MOVED HERE from Base Class) ---
    enable_timing: bool = False
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"

    log_dynamics_state: bool = True
    log_linear_system_data: bool = True
    log_constraint_data: bool = True

    def create_engine(
        self,
        model: Any,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
    ):
        # Import internally to avoid circular imports
        from axion.core.engine import AxionEngine

        if init_state_fn is None:
            raise ValueError("AxionEngine requires an init_state_fn.")

        # Pass the separate config objects to the constructor
        return AxionEngine(
            model=model,
            init_state_fn=init_state_fn,
            config=self,
            logging_config=logging_config,
        )

    def __post_init__(self):
        """Validate all configuration parameters."""

        def _validate_positive_int(value: int, name: str, min_value: int = 1) -> None:
            if value < min_value:
                raise ValueError(f"{name} must be >= {min_value}, got {value}")

        def _validate_non_negative_float(value: float, name: str) -> None:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        def _validate_unit_interval(value: float, name: str) -> None:
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        # Validate iteration counts
        _validate_positive_int(self.max_newton_iters, "max_newton_iters")
        _validate_positive_int(self.max_linear_iters, "max_linear_iters")

        # Validate modes
        if self.newton_mode not in ("convergence", "fixed"):
            raise ValueError(
                f"newton_mode must be 'convergence' or 'fixed', got {self.newton_mode}"
            )
        if self.linear_mode not in ("convergence", "fixed"):
            raise ValueError(
                f"linear_mode must be 'convergence' or 'fixed', got {self.linear_mode}"
            )

        # Validate tolerances
        _validate_non_negative_float(self.newton_tol, "newton_tol")
        _validate_non_negative_float(self.newton_atol, "newton_atol")
        _validate_non_negative_float(self.linear_tol, "linear_tol")
        _validate_non_negative_float(self.linear_atol, "linear_atol")

        # Validate physics params
        _validate_non_negative_float(self.joint_stabilization_factor, "joint_stabilization_factor")
        _validate_non_negative_float(
            self.contact_stabilization_factor, "contact_stabilization_factor"
        )
        _validate_non_negative_float(self.joint_compliance, "joint_compliance")
        _validate_non_negative_float(self.equality_compliance, "equality_compliance")
        _validate_non_negative_float(self.contact_compliance, "contact_compliance")
        _validate_non_negative_float(self.friction_compliance, "friction_compliance")
        _validate_non_negative_float(self.regularization, "regularization")

        # Validate Fisher-Burmeister parameters
        _validate_unit_interval(self.contact_fb_alpha, "contact_fb_alpha")
        _validate_unit_interval(self.contact_fb_beta, "contact_fb_beta")
        _validate_unit_interval(self.friction_fb_alpha, "friction_fb_alpha")
        _validate_unit_interval(self.friction_fb_beta, "friction_fb_beta")

        if self.enable_linesearch:
            _validate_positive_int(
                self.linesearch_conservative_step_count, "linesearch_conservative_step_count"
            )
            _validate_positive_int(
                self.linesearch_optimistic_step_count, "linesearch_optimistic_step_count"
            )
            _validate_non_negative_float(
                self.linesearch_conservative_upper_bound, "linesearch_conservative_upper_bound"
            )
            _validate_non_negative_float(
                self.linesearch_optimistic_window, "linesearch_optimistic_window"
            )
            _validate_non_negative_float(self.linesearch_min_step, "linesearch_min_step")

            if self.linesearch_conservative_upper_bound <= self.linesearch_min_step:
                raise ValueError(
                    "linesearch_conservative_upper_bound must be > linesearch_min_step"
                )

        _validate_positive_int(self.max_contacts_per_world, "max_contacts_per_world")

        if self.differentiable_simulation:
            _validate_positive_int(self.max_trajectory_steps, "max_trajectory_steps", min_value=1)


@dataclass(frozen=True)
class FeatherstoneEngineConfig(EngineConfig):
    angular_damping: float = 0.05
    update_mass_matrix_interval: int = 1
    friction_smoothing: float = 1.0
    use_tile_gemm: bool = False
    fuse_cholesky: bool = True

    def _get_solver_class(self):
        from newton.solvers import SolverFeatherstone

        return SolverFeatherstone


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

    def _get_solver_class(self):
        from newton.solvers import SolverMuJoCo

        return SolverMuJoCo


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

    def _get_solver_class(self):
        from newton.solvers import SolverXPBD

        return SolverXPBD


@dataclass(frozen=True)
class SemiImplicitEngineConfig(EngineConfig):
    """Configuration for Newton's Semi-Implicit Euler solver."""

    angular_damping: float = 0.05
    friction_smoothing: float = 1.0
    joint_attach_ke: float = 1.0e4
    joint_attach_kd: float = 1.0e2
    enable_tri_contact: bool = True

    def _get_solver_class(self):
        from newton.solvers import SolverSemiImplicit

        return SolverSemiImplicit
