# Constraints Formulation

This section establishes the mathematical foundation for representing articulated bodies and their interactions as constraint equations. Understanding these constraint formulations is essential before exploring how they are enforced through optimization principles and numerical methods.

In Axion, all physical interactions—joints, contacts, and friction—are unified under a single mathematical framework of constraint equations. This creates a large-scale system that captures the essential physics while enabling robust numerical solution.

---

## 1. Constraint Formulations: Position and Velocity

In computational mechanics, constraint equations can be enforced at the level of position or velocity. The choice of which level to use has profound implications for the simulator's stability and accuracy.

### Position-Level Formulation (Integral Form)

This formulation defines constraints using the geometric configuration of the bodies, \(q\). It is the most direct and robust method as it targets the "ground truth" of the physical system.

* **Unilateral (Inequality) Constraints:** For non-penetration, the gap distance function **c**(**q**) must be non-negative:

\[
\mathbf{c}(\mathbf{q}) \geq \mathbf{0}
\]

* **Bilateral (Equality) Constraints:** For a joint, a constraint function **c**(**q**) must be exactly zero:

\[
\mathbf{c}(\mathbf{q}) = \mathbf{0}
\]

Solving constraints at this level guarantees zero positional error, but it requires more advanced non-linear solvers.

### Velocity-Level Formulation (Differential Form)

This common alternative is derived by taking the time derivative of the position-level constraints. Using the chain rule, we get

\[
\dot{\mathbf{c}} = \frac{\partial \mathbf{c}}{\partial q} \frac{dq}{dt} = \mathbf{J}\mathbf{u},
\]

where **J** is the constraint Jacobian and **u** is the generalized velocity vector.

* **Bilateral Constraint:** The velocity constraint becomes

\[
\mathbf{J}\mathbf{u} = \mathbf{0},
\]

enforcing that the relative velocity in the constrained directions is zero.

* **Unilateral Constraint:** The non-penetration condition becomes a complementarity problem on velocities: when a contact is active (\(c(q) = 0\)), the relative normal velocity must be non-negative

\[
\dot{c} = J \mathbf{u} \ge 0.
\]

This forms the basis of many **Linear Complementarity Problem (LCP)** solvers.

### The Problem of "Drift" and Stabilization

While computationally simpler, velocity-level solvers suffer from a critical flaw: **numerical drift**. Enforcing \(\mathbf{J}\mathbf{u} = \mathbf{0}\) only ensures the velocity is correct at a given instant. Due to numerical integration errors accumulating over many time steps, the underlying position constraint \(\mathbf{c}(q)\) will inevitably "drift" away from zero. This manifests as joints slowly pulling apart or objects gradually sinking into one another.

To combat this, velocity-based solvers must add a **feedback rule** to push the system back toward a valid state. The most common method is **Baumgarte Stabilization**, which re-frames the constraint force as a physical spring-damper.

Conceptually, the constraint force **λ** is modeled as an implicit Hookean spring that acts to close any existing positional error **c**(**q**):

\[
\boldsymbol{\lambda} = -k \mathbf{c}(\mathbf{q}) - b \mathbf{v}_{rel}
\]

Here, \(\mathbf{c}(\mathbf{q})\) is the position error (e.g., penetration depth), \(\mathbf{v}_{\text{rel}}\) is the relative velocity, \(k\) is a spring stiffness, and \(b\) is a damping coefficient. This force pulls the bodies back into alignment.

To be used in a velocity-level solver, this is rearranged into a modified velocity constraint. The result is a velocity goal that not only enforces the constraint but also tries to correct a fraction of the position error over the next time step \(h\). This introduces two user-facing parameters:

1. **Error Reduction Parameter (ERP):** A factor, typically between 0 and 1, that specifies what fraction of the positional error to correct in the next time step. It is often calculated as \(\text{ERP} = \frac{h k}{h k + b}\). An ERP of 0.2 means the system will attempt to resolve 20% of the penetration depth in the current step.

2. **Constraint Force Mixing (CFM):** A small, soft parameter, proportional to \(\frac{1}{h k + b}\), that is added to the diagonal of the constraint matrix. It makes the constraint "softer," allowing for a small amount of violation in exchange for a much more stable and well-conditioned numerical system. This is especially useful when dealing with redundant contacts.

While this spring-based stabilization is more physically intuitive, tuning ERP and CFM values is notoriously difficult. Poor tuning can lead to spongy, oscillating joints or bouncy contacts.

Axion's direct, position-level DVI formulation elegantly sidesteps this entire problem. By solving for the position constraints directly, it eliminates drift by design, removing the need for fragile stabilization hacks and ensuring superior long-term stability.

---

## 2. Contact Constraints

Contact constraints prevent bodies from interpenetrating, representing one of the most challenging aspects of physics simulation due to their unilateral (inequality) nature.

### Position-Level Formulation

For each potential contact point, we define a **gap function** \(\mathbf{c}_{\text{contact}}(\mathbf{q})\) that measures the signed distance between bodies:

\[
c_{\text{contact}}(\mathbf{q}) \geq 0
\]

When \(c_{\text{contact}} > 0\), bodies are separated; when \(c_{\text{contact}} = 0\), bodies are just touching.

The contact constraint is formulated as a **Nonlinear Complementarity Problem (NCP)**:

\[
0 \leq \lambda_n \perp \mathbf{c}_{\text{contact}}(\mathbf{q}) \geq 0
\]

This mathematical relationship captures the essential physics:

* **Non-negativity**: Contact impulses are repulsive (\(\lambda_n \geq 0\)) and no penetration occurs (\(\mathbf{c}(\mathbf{q}) \geq 0\))  
* **Complementarity**: Either bodies are separated (\(\mathbf{c}(\mathbf{q}) > 0\), \(\lambda_n = 0\)) or in contact (\(\mathbf{c}(\mathbf{q}) = 0\), \(\lambda_n \geq 0\))

By solving this position-level problem directly, Axion finds the exact impulses required to satisfy the non-penetration constraint, resulting in stable simulation without drift.

### Velocity-Level Formulation (Alternative)

For contrast, contacts can also be formulated at the velocity level. Taking the time derivative of the position constraint, we get a condition on the relative normal velocity \(\mathbf{v}_n = \mathbf{J}_n\mathbf{u}\):

\[
0 \leq \lambda_n \perp \mathbf{v}_n \geq 0
\]

This forms a **Linear Complementarity Problem (LCP)** that ensures bodies don't move further into each other when in contact. However, this approach requires additional stabilization mechanisms to prevent drift, as discussed in Section 1.

---

## 3. Friction Constraints

Friction constraints apply tangential impulses that resist sliding motion between contacting bodies. Axion uses the smooth, isotropic Coulomb friction model derived from the **principle of maximal dissipation**.

### Mathematical Formulation

The principle of maximal dissipation states that the friction impulse **λ**<sub>t</sub> will remove the maximum amount of kinetic energy from the system, subject to the Coulomb constraint that its magnitude is limited by the normal impulse **λ**<sub>n</sub> and the coefficient of friction **μ**:

\[
\|\boldsymbol{\lambda}_t\| \leq \mu \cdot \lambda_n
\]

The Karush-Kuhn-Tucker (KKT) conditions for this model precisely describe the friction behavior:

1. **Sliding**: If there is relative tangential velocity (\(\mathbf{v}_t \neq \mathbf{0}\)), the friction impulse opposes it at maximum magnitude: \(\boldsymbol{\lambda}_t = -\mu\lambda_n \mathbf{v}_t/\|\mathbf{v}_t\|\)
2. **Sticking**: If there is no relative tangential velocity (\(\mathbf{v}_t = \mathbf{0}\)), the friction impulse is whatever is necessary to prevent motion, up to the maximum limit: \(\|\boldsymbol{\lambda}_t\| \leq \mu\lambda_n\)

This principle-based formulation is implemented using an NCP-function and a fixed-point iteration, which recasts the friction model into a symmetric system that fits seamlessly into the non-smooth Newton solver covered in [Nonlinear System](./non-linear-system.md).

---

## 4. Joint Constraints

Joints connect bodies and restrict their relative motion. In Axion, joints are implemented using the same unified constraint framework as contacts and friction. The engine employs a **constraints-based** (or **full-coordinate**) approach, where each rigid body retains its full 6 degrees of freedom (DOFs), and the solver computes the exact joint impulses required to enforce the desired motion restriction.

Currently, the primary joint type implemented in Axion is the **Revolute Joint**.

#### Mathematical Model

Like contacts, joint constraints can be defined at either the position or velocity level. A revolute joint (or hinge) is modeled as a set of five simultaneous **bilateral (equality) constraints**. Axion solves these directly at the position level to ensure maximum stability.

The constraint functions \(\mathbf{c}(q)\) are defined to be zero when the joint is perfectly aligned:

1. **Anchor Points Coincide (3 DOFs):** The world-space positions of the anchor points on each body must be equal.

\[
\mathbf{c_\text{trans}}(q) = \mathbf{p_\text{child}} - \mathbf{p_\text{parent}} = \mathbf{0}
\]

2. **Hinge Axes Collinear (2 DOFs):** The designated axes on each body must remain aligned.

The solver finds the joint impulses \(\boldsymbol{\lambda}_j\) required to enforce \(\mathbf{c}(q) = \mathbf{0}\). Because Axion solves the position-level equation directly via its non-smooth Newton method, any numerical error that would cause the joint to separate is corrected automatically.

**Alternative: Velocity-Level Formulation**

In contrast, many simulators formulate joints at the velocity level by taking the time derivative of the position constraint:

\[
\dot{\mathbf{c}}(q) = \mathbf{J}\mathbf{u} = \mathbf{0}
\]

Here, the solver finds impulses that force the relative velocity **u** along the constrained degrees of freedom to be zero. As discussed in Section 1, this approach suffers from numerical drift, where integration errors cause the positional error **c**(**q**) to grow over time, making the joint appear to pull apart. This necessitates corrective measures like **Baumgarte stabilization**, which can be difficult to tune.

---

## Conclusion: From Constraints to Optimization

The constraint formulations presented in this section—position/velocity-level approaches, contact complementarity, friction stick-slip behavior, and joint restrictions—create a complex system mixing equalities and inequalities that must be satisfied simultaneously.

The key question becomes: **How do we determine the constraint impulses λ that enforce all these constraints while respecting the system dynamics?**

This challenge is addressed through **Gauss's Principle of Least Constraint**, which provides a principled optimization framework for determining these impulses. The principle transforms the constraint enforcement problem into an optimization problem that can be solved numerically.

→ **Next**: [Gauss's Principle of Least Constraint](./gauss-least-constraint.md)
