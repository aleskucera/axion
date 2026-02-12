from dataclasses import dataclass


@dataclass
class LoggingConfig:
    enable_timing: bool = False
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"
    max_simulation_steps: int = 300
    log_dynamics_state: bool = True
    log_linear_system_data: bool = True
    log_constraint_data: bool = True
