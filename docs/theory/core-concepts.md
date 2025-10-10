# Core Concepts

This section provides a high-level mathematical overview of how Axion formulates and solves the physics simulation problem. Understanding these concepts is essential for grasping the theoretical foundation underlying the simulator's robust and unified approach.

---

## Mathematical Foundation

Axion's physics engine is built on a unified mathematical framework that treats all physical phenomena—articulated body dynamics, contact interactions, and joint constraints—as a single, coupled system of equations. This approach provides superior stability and accuracy compared to traditional methods that handle these phenomena separately.

### Articulated Bodies

Mathematically, an articulated body system consists of:

* **Configuration Space**: The system's state is described by generalized coordinates \(\mathbf{q} \in \mathbb{R}^n\), where each rigid body has position and orientation parameters
* **Velocity Space**: Generalized velocities \(\mathbf{u} \in \mathbb{R}^m\) represent the system's motion, typically with 6 degrees of freedom per rigid body (3 translational, 3 rotational)
* **Dynamics**: The equations of motion are governed by:

\[
\mathbf{M}(\mathbf{q}) \Delta\mathbf{u} = \mathbf{f}_{\text{ext}} h + \mathbf{J}^T(\mathbf{q}) \boldsymbol{\lambda}
\]

where \(\mathbf{M}\) is the mass matrix, \(\Delta\mathbf{u} = \mathbf{u}^+ - \mathbf{u}^-\) is the velocity change over timestep \(h\), \(\mathbf{f}_{\text{ext}} h\) represents external impulses, \(\mathbf{J}\) is the constraint Jacobian, and \(\boldsymbol{\lambda}\) are constraint impulses. The meaning and derivation of these constraint impulses will be explained in [Gauss's Principle of Least Constraint](./gauss-least-constraint.md).

### Contact and Constraint Formulation

Physical interactions are mathematically encoded as constraints:

* **Joint Constraints** (bilateral): \(\mathbf{c}_{\text{joint}}(\mathbf{q}) = \mathbf{0}\) — joints enforce exact geometric relationships between bodies
* **Contact Constraints** (unilateral): \(\mathbf{c}_{\text{contact}}(\mathbf{q}) \geq 0\) — bodies cannot interpenetrate
* **Friction Constraints**: Complex complementarity conditions that model stick-slip behavior

These constraints create a system mixing equalities and inequalities, requiring specialized mathematical treatment.

---

## Solution Approach

Axion's approach follows a four-step mathematical progression:

### 1. Constraint Formulation

First, we mathematically formulate how articulated bodies and their interactions are represented as constraint equations. This establishes the mathematical foundation for describing joints, contacts, and friction.

→ **Next**: [Constraints Formulation](./constraints.md)

### 2. Optimization Principle  

We apply **Gauss's Principle of Least Constraint**, which provides a principled way to determine how the system should evolve when subject to constraints. This principle frames constraint enforcement as an optimization problem.

→ **Next**: [Gauss's Principle of Least Constraint](./gauss-least-constraint.md)

### 3. Nonlinear System

The optimization principle, combined with time discretization, leads to a large nonlinear system of equations that must be solved at each time step. This system encodes all physical laws simultaneously.

→ **Next**: [Nonlinear System](./non-linear-system.md)

### 4. Numerical Solution

Finally, we numerically solve this nonlinear system using a specialized Newton-type method designed to handle the non-smooth nature of contact and friction.

→ **Next**: [Numerical Solution](./linear-system.md)

---

## Why This Unified Approach?

Traditional physics engines handle dynamics, contacts, and joints in separate phases, leading to:

* **Instability** in tightly coupled systems
* **Drift** and constraint violation accumulation  
* **Artificial softness** in joints and contacts

Axion's unified mathematical formulation addresses these issues by:

* **Solving everything simultaneously** — no artificial sequencing
* **Position-level constraint enforcement** — eliminates drift by design
* **Principled optimization framework** — mathematically grounded decisions

This mathematical rigor enables stable simulation of complex scenarios like articulated robots making contact with the environment, which often challenge traditional approaches.
