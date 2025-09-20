# Core Concepts

This guide explains the fundamental physics and computational models that drive Axion. Understanding these concepts will help you build more complex, stable, and efficient simulations.

The simulation framework is heavily inspired by the "Non-Smooth Newton" methods used in advanced robotics, where all physical laws (motion, contact, friction, joints) are solved simultaneously in a unified, implicit system.

---

## 1. The Governing Equations: A Unified Approach

At its heart, Axion's physics engine represents the entire state of the world as a large system of equations and inequalities that must be solved at every time step. This is known as a **monolithic** or **all-at-once** approach.

Instead of handling collisions, then joint forces, then dynamics in separate stages, Axion formulates everything as a single **Nonlinear Complementarity Problem (NCP)**. The core task at each discrete time step is to find a state that satisfies all of the following simultaneously:

* **Equations of Motion**: How bodies accelerate under forces.
* **Joint Constraints**: How bodies are connected (e.g., a hinge must stay on its axle).
* **Contact Constraints**: How bodies avoid interpenetration and respond to collision.

This unified system is then solved using a **Non-Smooth Newton Method**.

!!! success "Why This Matters"
    This approach allows for robust and stable simulation of complex, highly-coupled systems like articulated robots making contact with deformable objects—scenarios where simpler, staged methods often fail or become unstable.

---

## 2. Rigid Bodies and Shapes

In Axion, it's crucial to distinguish between a "body" and a "shape."

* A **Body** (`builder.add_body()`) is a abstract entity that holds the *dynamic state*: its position, orientation, velocity, and mass properties. It is essentially a coordinate frame that moves according to physics.
* A **Shape** (`builder.add_shape_*()`) defines the *geometry and material properties* of an object. It is attached to a body and is used for collision detection and calculating mass and inertia.

A single body can have multiple shapes to represent a complex object, or it can have no shape at all (useful for creating abstract attachment points for joints).

```python
# Create a body—the dynamic object with a position and orientation.
my_body = builder.add_body(origin=wp.transform(position, rotation))

# Give it a collision shape, which also defines its material properties.
builder.add_shape_box(
    body=my_body,         # Attach the shape to the body
    hx=0.5, hy=0.5, hz=0.5, # Defines a 1x1x1 meter cube
    density=1000.0,       # Used to compute mass and inertia
    mu=0.8                # Friction coefficient for contacts
)
```

The `density` of a shape, combined with its volume, is used to automatically compute the mass and rotational inertia of the parent body.

---

## 3. Contact as a Complementarity Problem

Axion models contact not with classic springs, but with hard, non-penetration constraints. This is mathematically formulated as a **complementarity condition**.

Think of the two fundamental rules of contact for any two objects:

1. Their separation distance must be non-negative (`distance ≥ 0`).
2. The contact force pushing them apart must be non-negative (force is repulsive, not attractive: `force ≥ 0`).

Furthermore, these two conditions are complementary:

* If there is a gap (`distance > 0`), the contact force must be zero (`force = 0`).
* If there is a contact force (`force > 0`), there must be no gap (`distance = 0`).

This relationship is concisely written as `0 ≤ distance ⊥ force ≥ 0`.

Axion converts this complementarity condition into a set of non-smooth equations, which are then fed into the main solver. The solver's job is to find a set of forces that satisfy this condition for all potential contact points in the scene.

#### Contact Parameters

The key physical parameters you control for contact are:

| Parameter | Description |
| :--- | :--- |
| `mu` | **Coefficient of Friction**. A value of 0.0 is frictionless, while 1.0 or higher represents a very high-friction surface. This is used to solve a similar complementarity problem for tangential (friction) forces. |
| `restitution` | **Coefficient of Restitution**. Controls the "bounciness" of a collision. A value of 0.0 means the bodies will not bounce at all, while 1.0 represents a perfectly elastic collision with no energy loss. |

---

## 4. The Solver: A Non-Smooth Newton Method

The collection of all dynamics equations, joint constraints, and contact complementarity conditions forms one large, non-smooth system of equations. To solve this system at each time step, Axion uses an iterative **Non-Smooth Newton Method**.

This method is powerful because it can handle the instantaneous, non-smooth events inherent in physics simulation, such as a body making initial contact or transitioning from static to kinetic friction.

!!! info "Solver Iterations (`newton_iters`)"
    The `newton_iters` parameter in the `EngineConfig` controls how many iterations this solver performs per time step.
    **More iterations** lead to a more accurate solution of the constraints, resulting in less drift, less penetration, and more rigid-feeling joints.
    **Fewer iterations** are faster but may introduce visible errors, especially in complex scenes.

    *After a certain point (e.g., 8-20 iterations), the visible improvement is minimal.*

