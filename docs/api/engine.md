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

## `EngineConfig`: Tuning the Solver

The `EngineConfig` dataclass centralizes all parameters that control the solver's behavior. Below is a breakdown of these parameters, grouped by their function.

```python
from axion.core import EngineConfig

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

---

## The Solving Process

The `simulate()` method orchestrates a multi-stage process for each time step, executed entirely on the GPU.

1. **Apply Controls & Integrate:** External forces and torques from the `control` object are applied, and an initial "guess" for the next state's velocity is calculated.

2. **The Non-Smooth Newton Loop:** The engine then enters the main iterative loop (`newton_iters`). Each iteration aims to refine the solution.

    a. **Linearize (`compute_linear_system`)**: The engine evaluates all constraints and linearizes them, forming a large linear system of equations `Ax = b`.

    b. **Solve (`cr_solver`)**: The core of the iteration. It solves the linear system for `Δλ` (the change in constraint forces) using a **Conjugate Residual (CR)** iterative solver. This step runs for `linear_iters`.

    c. **Linesearch (`perform_linesearch`)**: The solver may optionally perform a linesearch to find an optimal fraction of the calculated update to apply.

    d. **Update (`update_variables`)**: The body velocities (`body_qd`) and constraint forces (`_lambda`) are updated.

3. **Finalize State**: After the Newton loop completes, the final velocities and integrated positions (`body_q`) are copied to the `state_out` object.

