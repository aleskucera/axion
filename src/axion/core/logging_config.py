from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Persistent-state logging configuration.

    Three independent subsystems plus generic flags. The host-side
    "segment timing" coarse timer used to live here as ``enable_timing``;
    it has been moved to ``engine.profiling.segment_timing`` since
    profiling and persistent logging are different concerns.
    """

    # HDF5 simulation log
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"
    max_simulation_steps: int = 300
    log_dynamics_state: bool = True
    log_linear_system_data: bool = True
    log_constraint_data: bool = True

    # Dataset log (state-only, for ML)
    enable_dataset_logging: bool = False
    dataset_log_file: str = "dataset.h5"
    dataset_simulation_steps: int = 300

    # Adjoint log (gradient-mode)
    enable_adjoint_logging: bool = False
    adjoint_log_file: str = "adjoint.h5"
