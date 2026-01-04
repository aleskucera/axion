# Axion Testing Plan

This document outlines the testing strategy for the Axion physics engine. The goal is to ensure correctness of the batched simulation model, constraint stability, and generation tools.

## 1. Core Data Structures (`src/axion/core/`)

### `axion_model.py`
This is critical as it slices the monolithic `newton.Model` into per-world views.

*   **`test_batched_model_slicing`**:
    *   **Goal**: Verify that data is correctly distributed across worlds.
    *   **Method**: Create a `newton.Model` with $W=2$ worlds. World 0 has a Box (mass=1), World 1 has a Sphere (mass=2).
    *   **Assertion**: Check `AxionModel.body_mass[0]` vs `AxionModel.body_mass[1]`. Verify `joint_count`, `shape_count` per world.

*   **`test_batched_contacts_collection`**:
    *   **Goal**: Ensure `AxionContacts` correctly bins contacts into worlds.
    *   **Method**: Manually inject `newton.Contacts` data where contact 0 is in World 0 and contact 1 is in World 1.
    *   **Assertion**: `AxionContacts.contact_count[0] == 1` and `AxionContacts.contact_count[1] == 1`.

### `engine_data.py`
*   **`test_engine_arrays_allocation`**:
    *   **Goal**: Verify memory allocation sizes.
    *   **Method**: Initialize `EngineArrays` with specific `EngineDimensions`.
    *   **Assertion**: Check `.shape` properties of `body_q`, `body_lambda`, `_J_values`.

## 2. Constraints (`src/axion/constraints/`)

These tests should verify the mathematical correctness of constraint kernels.

### `positional_joint_constraint.py`
*   **`test_joint_kinematics`**:
    *   **Goal**: Verify `compute_joint_kinematics` returns correct world-space anchors.
    *   **Method**: Setup two bodies at known positions/rotations.
    *   **Assertion**: Calculated `pos_p_world` and `pos_c_world` match manual calculation.

*   **`test_revolute_joint_locking`**:
    *   **Goal**: Ensure revolute joint prevents translation but allows rotation along axis.
    *   **Method**: Single step simulation of two bodies connected by revolute joint. Apply force perpendicular to axis.
    *   **Assertion**: Relative translation should be near zero. Relative rotation along axis should be non-zero.

### `positional_contact_constraint.py`
*   **`test_signed_distance`**:
    *   **Goal**: Verify collision detection math.
    *   **Method**: Place two spheres with radius $R$ at distance $D$.
    *   **Assertion**: `signed_distance` should be $D - 2R$.

*   **`test_fisher_burmeister_complementarity`**:
    *   **Goal**: Verify the complementarity function.
    *   **Method**: Unit test `scaled_fisher_burmeister` with various inputs (penetrating, separating).
    *   **Assertion**: Should behave like a smooth `max(0, x)`.

## 3. Generation (`src/axion/generation/`)

### `scene_generator.py`
*   **`test_chain_generation`**:
    *   **Goal**: Verify topology of generated chains.
    *   **Method**: Call `generate_chain(length=5)`.
    *   **Assertion**: Result should have 5 bodies and (typically) 5 joints. All bodies should be connected (graph traversal check).

*   **`test_collision_free_placement`**:
    *   **Goal**: Ensure generator checks for overlaps.
    *   **Method**: Try to generate a box exactly where another box exists.
    *   **Assertion**: Generator should return `None` or find a new position, not overlap.

## 4. Integration & Simulation (`src/axion/core/engine.py`)

### Determinism & Independence
*   **`test_simulation_determinism`**:
    *   **Goal**: Same seed = Same result.
    *   **Method**: Run simulation twice with same seed.
    *   **Assertion**: `norm(state1 - state2) == 0`.

*   **`test_world_independence`**:
    *   **Goal**: Activity in World A should not affect World B.
    *   **Method**: Run $W=2$. World 0 is empty/static. World 1 has a chaotic pile of boxes.
    *   **Assertion**: World 0 bodies remain at rest (velocity = 0).

### Physics Quality
*   **`test_energy_conservation`**:
    *   **Goal**: Basic sanity check for integrators.
    *   **Method**: Simple pendulum, no damping, no friction.
    *   **Assertion**: Total Energy (KE + PE) should be roughly constant over short time.

*   **`test_stack_stability`**:
    *   **Goal**: Verify solver convergence.
    *   **Method**: Stack of 5 boxes. Simulate 100 steps.
    *   **Assertion**: Stack should not collapse or jitter excessively (max velocity threshold).

## Running Tests
Use `pytest` to run these tests.
```bash
pytest tests/
```
