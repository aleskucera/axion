# Your First Simulation

This tutorial will guide you through creating a custom physics simulation with Axion.

## Basic Structure

Every Axion simulation follows the same pattern:

1. **Inherit from `AbstractSimulator`** - This base class handles the simulation loop, USD export, and configuration
2. **Override `build_model()`** - Define your physics scene (bodies, joints, constraints)
3. **Configure and run** - Set physics parameters and execute the simulation

```python
from axion import AbstractSimulator
import warp as wp

class MySimulator(AbstractSimulator):
    def build_model(self) -> wp.sim.Model:
        # Build your physics model here
        pass
```

## Step 1: Create a Falling Rod

Let's create a simple simulation of a rod falling under gravity.

Create a file called `first_simulation.py`:

```python
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> wp.sim.Model:
        # Create model builder with Z-axis up (gravity points down in -Z)
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Define initial rotation: 15 degrees tilt around X-axis
        # This makes the rod fall instead of standing perfectly upright
        angle = 0.2618  # radians (approx. 15 degrees)
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

        # Create a rigid body "rod A" at position (0, 0, 3) meters
        # The rod starts 3 meters above ground with a slight tilt
        rod_a = builder.add_body(
            origin=wp.transform((0, 0, 3), rot_a_quat), 
            name="rod A"
        )
        
        # Attach a box shape to the body (this defines its collision geometry)
        builder.add_shape_box(
            body=rod_a,
            hx=0.2,  # half-width: 0.4m total width
            hy=0.2,  # half-depth: 0.4m total depth  
            hz=1.0,  # half-height: 2.0m total height (tall rod)
            density=1000.0,      # kg/m³ (like water)
            mu=0.8,              # friction coefficient (fairly grippy)
            restitution=0.3,     # bounce factor (0=no bounce, 1=perfect bounce)
        )
        
        # This creates an infinite horizontal surface at Z=0
        builder.set_ground_plane(mu=0.8, restitution=0.3)

        # Finalize the model (required to prepare for simulation)
        return builder.finalize()

if __name__ == "__main__":
    # Create simulator instance with configuration
    sim = Simulator(
        sim_config=SimulationConfig(duration_seconds=5.0),  # Run for 5 seconds
        render_config=RenderingConfig(),                    # Default rendering settings
        exec_config=ExecutionConfig(),                      # Default execution (GPU if available)
        profile_config=ProfilingConfig(),                   # No profiling by default
        engine_config=EngineConfig(),                       # Default physics settings
    )
    sim.run()  # Execute the simulation and generate USD file
```

### What happens when you run it?

1. **Initialization**: The rod spawns 3 meters above ground with a 15° tilt
2. **Gravity**: The rod falls under gravity (9.81 m/s² downward)
3. **Impact**: When it hits the ground (implied by the model), contact forces are computed
4. **Friction & Bounce**: The rod may slide (friction=0.8) and slightly bounce (restitution=0.3)
5. **Output**: A USD file is saved containing the entire motion sequence

Run it:

```bash
uv run python first_simulation.py
```

## Step 2: Add Multiple Bodies

Now let's create two rods that fall and collide. This demonstrates body-body interactions:

```python
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> wp.sim.Model:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Create two rods with opposite tilts - they'll fall toward each other
        angle = 0.2618  # 15 degrees in radians
        
        # Rod A: tilted forward (+15°)
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
        rod_a = builder.add_body(
            origin=wp.transform((0, 0, 3), rot_a_quat), 
            name="rod A"
        )
        builder.add_shape_box(
            body=rod_a,
            hx=0.2, hy=0.2, hz=1.0,  # Same dimensions as before
            density=1000.0,
            mu=0.8, restitution=0.3,
        )

        # Rod B: tilted backward (-15°) and offset 1m in Y direction
        rot_b_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -angle)
        rod_b = builder.add_body(
            origin=wp.transform((0, 1, 3), rot_b_quat),  # 1m apart in Y
            name="rod B"
        )
        builder.add_shape_box(
            body=rod_b,
            hx=0.2, hy=0.2, hz=1.0,
            density=1000.0,  # Same material properties
            mu=0.8, restitution=0.3,
        )

        # Add ground plane for both rods to land on
        builder.set_ground_plane(mu=0.8, restitution=0.3)

        return builder.finalize()


if __name__ == "__main__":
    sim = Simulator(
        sim_config=SimulationConfig(duration_seconds=5.0),
        render_config=RenderingConfig(),
        exec_config=ExecutionConfig(),
        profile_config=ProfilingConfig(),
        engine_config=EngineConfig(),
    )
    sim.run()
```

### What's new in Step 2?

- **Multiple bodies**: Two rods instead of one
- **Body-body collision**: The rods can collide with each other, not just the ground
- **Different initial conditions**: Opposite tilts create interesting dynamics
- **Ground plane**: An explicit infinite ground surface for both objects

When you run this, you'll see both rods fall under gravity, potentially colliding mid-air or on the ground depending on their trajectories.

## Step 3: Add Joints

Now let's connect the two rods with a joint to create an articulated system:

```python
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> wp.sim.Model:
        # Create model builder with Z-axis up
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Define a small rotation (15 degrees around x-axis)
        angle = 0.2618  # radians (approx. 15 degrees)
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

        # Add rod A with initial rotation
        rod_a = builder.add_body(origin=wp.transform((0, 0, 3), rot_a_quat), name="rod A")
        builder.add_shape_box(
            body=rod_a,
            hx=0.2,  # half extents in x
            hy=0.2,  # half extents in y
            hz=1.0,  # half extents in z
            density=1000.0,
            mu=0.8,  # friction
            restitution=0.3,
        )

        rot_b_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -angle)

        # Add rod B with initial rotation
        rod_b = builder.add_body(origin=wp.transform((0, 1, 3), rot_b_quat), name="rod B")
        builder.add_shape_box(
            body=rod_b,
            hx=0.2,  # half extents in x
            hy=0.2,  # half extents in y
            hz=1.0,  # half extents in z
            density=1000.0,
            mu=0.8,  # friction
            restitution=0.3,
        )

        # Add revolute joint connecting the two rods
        # This creates a hinge that allows rotation around the X-axis
        builder.add_joint_revolute(
            parent=rod_a,                                    # First rod acts as parent
            child=rod_b,                                     # Second rod is the child
            axis=wp.vec3(1.0, 0.0, 0.0),                    # Rotation axis (X-axis)
            # Connection points are computed to be at the bottom corners of each rod
            # These coordinates account for the initial rotations and positions
            parent_xform=wp.transform(
                wp.vec3(0.0, 0.2329623, -1.06242221),      # Bottom corner of rod A
                wp.quat_identity()
            ),
            child_xform=wp.transform(
                wp.vec3(0.0, -0.2329623, -1.06242221),     # Bottom corner of rod B
                wp.quat_identity()
            ),
        )

        # Add ground plane
        builder.set_ground_plane(mu=0.8, restitution=0.3)

        return builder.finalize()


if __name__ == "__main__":
    sim = Simulator(
        sim_config=SimulationConfig(duration_seconds=5.0),
        render_config=RenderingConfig(),
        exec_config=ExecutionConfig(),
        profile_config=ProfilingConfig(),
        engine_config=EngineConfig(),
    )
    sim.run()
```

### Understanding Joints

The revolute joint creates a mechanical connection between the rods:

- **Parent-Child relationship**: Rod A is the parent, Rod B is the child
- **Constraint**: The connection points must remain coincident (touching)
- **Freedom**: Rod B can rotate relative to Rod A around the specified axis
- **Physics**: Joint forces automatically maintain the constraint

When you run this simulation, the connected rods will behave like a hinged structure falling under gravity.

## Understanding the Physics Parameters

Let's break down the key parameters you've been using:

### Material Properties

```python
density=1000.0       # Mass per unit volume (kg/m³)
                     # 1000 = water density, affects inertia

mu=0.8               # Coulomb friction coefficient
                     # 0 = frictionless, 1 = very grippy

restitution=0.3      # Coefficient of restitution
                     # 0 = no bounce, 1 = perfect elastic bounce
```

### Transform and Rotation

```python
# Position and orientation combined
wp.transform((x, y, z), quaternion)

# Create rotation quaternion from axis and angle
wp.quat_from_axis_angle(axis_vector, angle_radians)

# Common rotations:
# X-axis: wp.vec3(1, 0, 0)  # pitch (nose up/down)
# Y-axis: wp.vec3(0, 1, 0)  # yaw (turn left/right)  
# Z-axis: wp.vec3(0, 0, 1)  # roll (lean left/right)
```

## Configuration Options

### Simulation Parameters

```python
sim_config = SimulationConfig(
    duration_seconds=3.0,      # Total simulation time
    dt=5e-3,          # Time step
)
```

### Engine Settings

```python
engine_config = EngineConfig(
    newton_iters=8,           # Newton solver iterations
    matrixfree_representation=True
)
```

### Rendering Options

```python
render_config = RenderingConfig(
    fps=30,            # Frames per second for USD output
    headless=False     # True for no visualization
)
```

## Next Steps

- Explore the [User Guide](../user-guide/concepts.md) to understand physics concepts
- Learn about [constraints](../user-guide/constraints.md) for complex interactions
- See [configuration options](../user-guide/configuration.md) for fine-tuning
