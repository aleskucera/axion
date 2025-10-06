# Configurable Parameters
The physics engine exposes a variety of parameters that allow you to configure the simulation, solver, rendering and logging. How to work with these parameters was described in the [Configuration System](../getting-started/configuration.md) guide. Here, we provide a detailed explanation of the parameters. 

## `EngineConfig`: Tuning the Solver

The `EngineConfig` dataclass centralizes all parameters that control the solver's behavior. Below is a breakdown of these parameters, grouped by their function.

```python
from axion import EngineConfig

@dataclass(frozen=True)
class EngineConfig:
    # Solver iterations
    newton_iters: int = 8
    linear_iters: int = 4
    linesearch_steps: int = 2
    
    # Baumgarte stabilization
    joint_stabilization_factor: float = 0.01
    contact_stabilization_factor: float = 0.1
    
    # Fischer-Burmeister scaling
    contact_fb_alpha: float = 0.25
    contact_fb_beta: float = 0.25
    friction_fb_alpha: float = 0.25
    friction_fb_beta: float = 0.25

    # Constraint compliance (softness)
    contact_compliance: float = 1e-4
    friction_compliance: float = 1e-6
    
    # Performance
    matrixfree_representation: bool = True
```

!!! success "Built-in Validation"
    `EngineConfig` includes a `__post_init__` method that validates your settings. If you provide an invalid value (e.g., a negative number of iterations), it will immediately raise a `ValueError`, preventing hard-to-debug issues later.

### Group 1: Solver Iterations

These parameters control the computational "effort" the solver expends at each time step.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `newton_iters` | 8 | **Newton Iterations.** The number of "outer loop" iterations for the nonlinear solver. More iterations lead to better constraint satisfaction (less penetration, stiffer joints). |
| `linear_iters` | 4 | **Linear Solver Iterations.** The number of "inner loop" iterations for the Conjugate Residual solver, which solves the linearized system at each Newton step. |
| `linesearch_steps` | 2 | Number of steps in the linesearch to find an optimal step size for each Newton update. Set to `0` to disable and take the full step. |

### Group 2: Baumgarte Stabilization

These parameters help the solver correct for positional drift from constraints over time.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `joint_stabilization_factor` | 0.01 | **Joint Drift Correction.** Controls how aggressively the solver corrects positional errors in joints. |
| `contact_stabilization_factor` | 0.1 | **Contact Penetration Correction.** Controls how aggressively the solver pushes penetrating objects apart. |

### Group 3: Fischer-Burmeister (FB) Scaling

These parameters are scaling factors for the Fischer-Burmeister function, which transforms a complementarity problem (like contact) into a root-finding problem that the Newton solver can handle.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `contact_fb_alpha` | 0.25 | Scales the *primal* variable (e.g., gap distance) of the contact complementarity problem. |
| `contact_fb_beta` | 0.25 | Scales the *dual* variable (e.g., contact force λ) of the contact complementarity problem. |
| `friction_fb_alpha` | 0.25 | Scales the *primal* variable (e.g., relative velocity) of the friction complementarity problem. |
| `friction_fb_beta` | 0.25 | Scales the *dual* variable (e.g., friction force λ) of the friction complementarity problem. |

!!! info "What is Fischer-Burmeister Scaling?"
    A contact constraint follows the rule `0 ≤ distance ⊥ force ≥ 0`. The FB function turns this into an equation `phi(distance, force) = 0`.

    However, `distance` (in meters) and `force` (in Newtons) can have vastly different numerical magnitudes. This imbalance can make the problem numerically difficult for the solver. The scaling factors `alpha` and `beta` are used to precondition the problem by solving a scaled version: `phi(alpha * distance, beta * force) = 0`.
    
    This brings the two arguments into a similar numerical range, improving the solver's stability and convergence speed.

### Group 4: Constraint Compliance (Softness)

Compliance is the inverse of stiffness. These parameters introduce a controlled amount of "softness," which can improve stability and simulate non-rigid behaviors.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `contact_compliance` | 1e-4 | Adds softness to contact constraints. `0.0` represents a perfectly hard contact. Larger values (e.g., `1e-2`) simulate softer materials. |
| `friction_compliance`| 1e-6 | Adds softness to the friction model. This is typically kept very low to simulate rigid friction. |

### Group 5: Performance

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `matrixfree_representation` | `True` | If `True`, the solver uses matrix-free linear operators (memory-efficient). If `False`, it builds an explicit system matrix (can be faster for small systems). |


## `SimulationConfig`: General Simulation Parameters

The `SimulationConfig` dataclass holds parameters for controlling the overall simulation behavior.

```python
from axion import SimulationConfig

@dataclass
class SimulationConfig:
    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `duration_seconds` | 3.0 | **Simulation Duration.** Controls how long the simulation runs in seconds. |
| `target_timestep_seconds` | 0.001 | **Target Timestep.** Controls the desired simulation timestep in seconds. If rendering is enabled, this parameter is recalculated to fit integer number of timesteps in 1/FPS seconds. |

## `RenderingConfig`: Visualization Settings
The `RenderingConfig` dataclass holds parameters for controlling the visual output of the simulation. The rendering is done using the [USD format](https://openusd.org/release/index.html), which can be viewed in tools like [Blender](https://www.blender.org/) or [Pixar's USDView](https://www.pixar.com/openusd).

```python
from axion import RenderingConfig

@dataclass
class RenderingConfig:
    enable: bool = True
    fps: int = 30
    scaling: float = 100.0
    usd_file: str = "sim.usd"
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enable` | `True` | If `True`, enables export of the simulation to a USD file. |
| `fps` | 30 | Target frames per second for rendering. The simulation timestep is adjusted to fit an integer number of steps per frame. |
| `scaling` | 100.0 | Scaling factor for the object meshes to convert from simulation units to meters. |
| `usd_file` | "sim.usd" | The filename where the USD scene is saved, relative to the working directory. |

## `ExecutionConfig`: Execution Settings
The `ExecutionConfig` dataclass holds parameters for controlling the performance and execution strategy of the simulation.

```python
from axion import ExecutionConfig

@dataclass
class ExecutionConfig:
    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `use_cuda_graph` | `True` | If `True`, enables CUDA Graphs to minimize kernel launch overhead and improve performance. |
| `headless_steps_per_segment` | 10 | Number of simulation steps to run in each headless segment when rendering is disabled. |

## `ProfilingConfig`: Profiling and Logging Settings
The `ProfilingConfig` dataclass holds parameters for controlling the profiling, timing and logging of the simulation. The logging is done using an [HDF5](https://www.hdfgroup.org/solutions/hdf5/)-based logger, which records detailed simulation data for later analysis, for example using [myHDF5](https://myhdf5.hdfgroup.org/).

```python
from axion import ProfilingConfig

@dataclass
class ProfilingConfig:
    enable_timing: bool = False

    # HDF5 Logging
    enable_hdf5_logging: bool = False
    hdf5_log_file: str = "simulation.h5"
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enable_timing` | `False` | If `True`, enables detailed printing of timing of simulation stages. |
| `enable_hdf5_logging` | `False` | If `True`, enables logging of detailed simulation data to an HDF5 file for later analysis. |
| `hdf5_log_file` | "simulation.h5" | The filename where the HDF5 log is saved, relative to the working directory