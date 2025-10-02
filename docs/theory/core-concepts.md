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

## 2. The Solver: A Non-Smooth Newton Method

The collection of all dynamics equations, joint constraints, and contact complementarity conditions forms one large, non-smooth system of equations. To solve this system at each time step, Axion uses an iterative **Non-Smooth Newton Method**.

This method is powerful because it can handle the instantaneous, non-smooth events inherent in physics simulation, such as a body making initial contact or transitioning from static to kinetic friction.

!!! info "Solver Iterations (`newton_iters`)"
    The `newton_iters` parameter in the `EngineConfig` controls how many iterations this solver performs per time step.
    **More iterations** lead to a more accurate solution of the constraints, resulting in less drift, less penetration, and more rigid-feeling joints.
    **Fewer iterations** are faster but may introduce visible errors, especially in complex scenes.

    *After a certain point (e.g., 8-20 iterations), the visible improvement is minimal.*
