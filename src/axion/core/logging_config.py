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
    enable_dataset_logging: bool = False
    dataset_log_file: str = "dataset.h5"
    dataset_simulation_steps: int = 300
    enable_neural_lambdas_logging: bool = False             # One-off logger for AxionEngineWithNeuralLambdas.
    neural_lambdas_log_file: str = "neural_lambdas.h5"      # One-off logger for AxionEngineWithNeuralLambdas.
    neural_lambdas_simulation_steps: int = 300              # One-off logger for AxionEngineWithNeuralLambdas.
    enable_adjoint_logging: bool = False
    adjoint_log_file: str = "adjoint.h5"
