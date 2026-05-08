from dataclasses import dataclass, field
from typing import Any
from typing import Optional

from axion.collision import ContactReductionConfig

from .engine_profiler import VALID_MODES as _VALID_PROFILING_MODES


@dataclass(frozen=True)
class EngineConfig:
    """
    Base configuration class.
    Defines the factory interface for creating physics engines.
    """

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
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
    max_newton_iters: int = 16
    max_linear_iters: int = 16

    backtrack_min_iter: int = 2

    # Tightened from 5e-2 so Newton runs to actual convergence with the
    # dt-adaptive contact_compliance regularization. With looser atol Newton
    # can exit early at a state still distorted by the regularization.
    newton_atol: float = 1e-3

    linear_tol: float = 1e-5
    linear_atol: float = 1e-5

    # PCR preconditioner choice. "jacobi" is the original (1/diag(A))
    # preconditioner; "per_body_pair" is the per-body-pair block-Jacobi
    # preconditioner that captures inter-contact same-body coupling
    # (see docs/pcr_warm_start_options.md and
    # src/axion/optim/per_body_pair_preconditioner.py).
    # Default "jacobi" preserves existing behavior; flip per-config
    # to A/B test before changing the default.
    preconditioner_type: str = "jacobi"

    joint_compliance: float = 1e-5
    # contact_compliance is scaled by 1/dt^2 inside the contact constraint
    # (see contact_constraint.py). The effective regularization is therefore
    # contact_compliance / dt^2, which is ~negligible at large dt and ~100
    # at dt=1ms with the default below. See docs/dt_dependence_problem.md.
    contact_compliance: float = 1e-4
    friction_compliance: float = 1e-6

    regularization: float = 1e-6

    enable_linesearch: bool = False

    # --- 1. Conservative Cluster (Safety first) ---
    linesearch_conservative_step_count: int = 32
    linesearch_conservative_upper_bound: float = 0.05
    linesearch_min_step: float = 1e-6

    # --- 2. Optimistic Cluster (The "Attitude") ---
    linesearch_optimistic_step_count: int = 32
    linesearch_optimistic_window: float = 0.2

    max_contacts_per_world: int = 128

    contact_reduction: ContactReductionConfig = field(
        default_factory=ContactReductionConfig
    )

    # Cross-step warm-start of contact normal/friction forces.
    # When True, the post-backtrack converged forces from step N are
    # carried into _constr_force_prev_iter at step N+1, so the friction
    # kernel sees non-zero f_n_prev at NR iter 0 instead of starting
    # from a degenerate (no-friction) residual. See
    # axion/collision/warm_start.py for the predicted-position matching
    # algorithm. Default False until the matching kernel ships
    # (currently scaffolding only).
    enable_contact_warm_start: bool = True

    # Cold-start heuristics for unmatched contacts (phase 2.5 of warm
    # start). Each term in `_cold_start_kernel` is independently
    # gated so we can ablate (A: all off, B: gravity, C: gravity+impact,
    # D: all three) by flipping flags from the YAML.
    #   gravity:  per-body residual-gravity split, α
    #   impact:   m_eff·(-v_n)/dt for approaching contacts, β
    #   friction: μ·λ_n_cold·(-v̂_t) projected to (t1,t2), γ
    # The friction term auto-disables itself when v_threshold ≤ 0.
    warm_start_cold_gravity: bool = True
    warm_start_cold_impact: bool = True
    warm_start_cold_friction_v_threshold: float = 0.1

    # --- Adjoint Backward Pass Interventions ---
    # NOTE: differentiable simulation is enabled via the
    # `differentiable_simulation` kwarg on `create_engine()` /
    # `AxionEngine.__init__`, NOT via a config field. Logging-related
    # fields (enable_timing, enable_hdf5_logging, hdf5_log_file,
    # log_dynamics_state, log_linear_system_data, log_constraint_data)
    # live on `LoggingConfig` in `axion/core/logging_config.py`.
    adjoint_soft_blending: bool = True
    adjoint_soft_blending_temperature: float = 0.05
    adjoint_regularization: float = 0.0
    adjoint_gradient_normalization: bool = False

    # --- Profiling ---
    # "off" | "end_to_end" | "per_component". See engine_profiler.py for the
    # phase breakdown each mode emits. ``per_component`` replaces the
    # capture_while NR loop with a fixed unroll of length max_newton_iters,
    # so it forces the worst-case iteration cost every step.
    profiling_mode: str = "off"

    @property
    def num_linesearch_steps(self):
        num_steps = self.linesearch_conservative_step_count
        num_steps += self.linesearch_optimistic_step_count
        return num_steps

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
    ):
        from axion.core.engine import AxionEngine

        return AxionEngine(
            model=model,
            sim_steps=sim_steps,
            config=self,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )

    def __post_init__(self):
        """Validate all configuration parameters."""

        # Hydra passes nested overrides as a DictConfig, not as a real
        # ContactReductionConfig instance. Coerce so downstream code can
        # always assume a frozen dataclass.
        coerced = ContactReductionConfig.coerce(self.contact_reduction)
        if coerced is not self.contact_reduction:
            object.__setattr__(self, "contact_reduction", coerced)

        def _validate_positive_int(value: int, name: str, min_value: int = 1) -> None:
            if value < min_value:
                raise ValueError(f"{name} must be >= {min_value}, got {value}")

        def _validate_non_negative_float(value: float, name: str) -> None:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        # Validate iteration counts
        _validate_positive_int(self.max_newton_iters, "max_newton_iters")
        _validate_positive_int(self.max_linear_iters, "max_linear_iters")

        if self.backtrack_min_iter >= self.max_newton_iters:
            raise ValueError(
                f"backtrack_min_iter mush be smaller than max_newton_iters, got {self.backtrack_min_iter} and {self.max_newton_iters}"
            )

        # Validate tolerances
        _validate_non_negative_float(self.newton_atol, "newton_atol")
        _validate_non_negative_float(self.linear_tol, "linear_tol")
        _validate_non_negative_float(self.linear_atol, "linear_atol")

        if self.preconditioner_type not in ("jacobi", "per_body_pair"):
            raise ValueError(
                "preconditioner_type must be 'jacobi' | 'per_body_pair', "
                f"got {self.preconditioner_type!r}"
            )

        # Validate physics params
        _validate_non_negative_float(self.joint_compliance, "joint_compliance")
        _validate_non_negative_float(self.contact_compliance, "contact_compliance")
        _validate_non_negative_float(self.friction_compliance, "friction_compliance")
        _validate_non_negative_float(self.regularization, "regularization")

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

        if self.profiling_mode not in _VALID_PROFILING_MODES:
            raise ValueError(
                f"profiling_mode must be one of {_VALID_PROFILING_MODES}, "
                f"got {self.profiling_mode!r}"
            )


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
