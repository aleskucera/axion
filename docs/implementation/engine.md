# Engine API

The `AxionEngine` is the low-level physics solver at the heart of the simulation. Most users will interact with it indirectly through the `AbstractSimulator` class, but understanding its API and configuration is key to tuning performance and achieving specific physical behaviors.

The engine implements a **Non-Smooth Newton Method** to solve the entire physics state—including dynamics, contacts, and joints—as a single, unified problem at each time step. This monolithic approach provides exceptional stability, especially for complex, highly-constrained systems like articulated robots.

---

## The `AxionEngine` Class

This is the main class that takes a physics `Model` and advances it through time.

```python
from axion.core import AxionEngine

class AxionEngine(Integrator):
    def __init__(
        self,
        model: Model,
        config: Optional[EngineConfig],
        logger: Optional[HDF5Logger | NullLogger],
    )
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model` | `warp.sim.Model` | The physics model created by a `ModelBuilder`, containing all bodies, shapes, and joints. |
| `config` | `EngineConfig` | A configuration dataclass that holds all tunable solver parameters. |
| `logger` | `HDF5Logger` or `NullLogger` | An optional logger for recording detailed simulation data to a file. |

### Key Methods

#### `simulate()`

The primary method for running the physics simulation for a single time step.

```python
def simulate(
    self,
    model: Model,
    state_in: State,
    state_out: State, 
    dt: float,
    control: Optional[Control] = None,
) -> List[Dict[str, Event]]
```

This method executes the core solver loop on the GPU, applying control inputs and calculating the resulting state after `dt` seconds.

#### `simulate_scipy()`

An alternative solver implementation that uses SciPy's numerical root-finding algorithms.

```python
def simulate_scipy(
    self, 
    model: Model, 
    # ...
    method: str = "hybr",
)
```

!!! warning "For Debugging and Validation Only"
    The `simulate_scipy` method runs on the CPU and is **significantly slower** than the native GPU-based `simulate` method. It is provided as a tool for validating the physics results against a well-established numerical library, not for performance-critical applications.

---

# The Solving Process

The `AxionEngine.simulate` method orchestrates a multi-stage process for each time step, executed entirely on the GPU.

## 1. Apply Controls & Integrate
External forces and torques from the `AbstractSimulator.control` object are applied, and an initial "guess" for the next state's velocity is calculated.

## 2. The Non-Smooth Newton Loop
The engine then enters the main Newton iteration loop for `EngineConfig.newton_iters` iterations. Each iteration aims to refine the solution.

### a) Linearize (`compute_linear_system`)
The engine evaluates all constraints and linearizes them, forming a large linear system of equations as described in the [theory section](../theory/linear-system.md). Since we know the structure of the system, we can construct the system efficiently without explicitly forming large matrices, which consist of many zero elements. The matrices can be simplified as follows:

- **Dynamic Matrix (H)** is a block diagonal matrix. It can be represented via one float for mass and 3x3 matrix for inertia per body.
- **Compliance (C)** is a diagonal matrix, represented as a vector of its diagonal elements.
- **Jacobian (J)** is a matrix representing constraint between two bodies. Each constraint can be represented as two integer indices of the two bodies and two Nx6 matrices, where 6 is DoF of a spatial body and N is the number of constraint equations. Rotational joint constraints are represented by 5 equations, contact constraint by 1 equation, and friction by 2 equations.

### b) Solve (`cr_solver`) and Compute Velocities
The core of the iteration. It solves the linear system for `Δλ` (the change in constraint forces) using a **Conjugate Residual (CR)** iterative solver. This step runs for `linear_iters`. Since the system is represented in a matrix-free form, the solver uses matrix-free operator to compute required quantities efficiently. 

Given the change in constraint forces `Δλ`, the corresponding change in body velocities `Δu` is computed.

### d) Update (`update_variables`)
The body velocities (`body_qd`) and constraint forces (`_lambda`) are updated.

## 3. Finalize State
After the Newton loop completes, the final velocities and integrated positions (`body_q`) are copied to the `state_out` object.
