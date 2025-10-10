# Nonlinear System

This section describes how the optimization problem from [Gauss's Principle of Least Constraint](./gauss-least-constraint.md) is transformed into a discretized nonlinear system of equations that can be solved numerically. This transformation is necessary to handle the complex mix of bilateral and unilateral constraints in a unified framework.

## From Constraints to Nonlinear Equations

The optimization problem from Gauss's principle must be combined with the specific constraint formulations from [Constraints Formulation](./constraints.md). Each constraint type requires transformation into a form suitable for numerical solution. This section shows how each constraint becomes part of the final nonlinear system.

We examine each constraint type and show how it contributes to the nonlinear system, considering both position-level and velocity-level formulations where applicable.

## Bilateral Constraints (Joints)

Bilateral constraints enforce exact geometric relationships between bodies, such as revolute joints that connect two bodies at a specific point and orientation. Axion can formulate these constraints at either the position or velocity level.

### Position-Level Formulation

From [Constraints Formulation](./constraints.md), bilateral constraints are defined as:

\[
\mathbf{c}_b(\mathbf{q}) = \mathbf{0}
\]

In the nonlinear system, this becomes:

\[
\mathbf{c}_b(\mathbf{q}^+) + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+ = \mathbf{0}
\]

where:

- \(\mathbf{q}^+\) is the configuration at the end of the time step
- \(\boldsymbol{\lambda}_b^+\) are the bilateral constraint impulses
- \(\boldsymbol{\Sigma}\) is the compliance matrix (typically zero for rigid constraints)

The compliance matrix is introduced for numerical stability and can model slight flexibility in otherwise rigid constraints.

### Velocity-Level Formulation (Alternative)

Alternatively, bilateral constraints can be enforced at the velocity level. The basic velocity constraint is:

\[
\mathbf{J}_b(\mathbf{q}) \cdot \mathbf{u} = \mathbf{0}
\]

However, this suffers from drift. To address this, Axion can use **Baumgarte stabilization**, which modifies the constraint to:

\[
\mathbf{J}_b \cdot \mathbf{u}^+ + \boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_b(\mathbf{q}^-)}{h} + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+ = \mathbf{0}
\]

where:

- \(\mathbf{J}_b(\mathbf{q}) = \frac{\partial \mathbf{c}_b}{\partial \mathbf{q}}\) is the constraint Jacobian
- \(\boldsymbol{\Upsilon}\) is the matrix of error correction coefficients (values between 0 and 1)
- \(h\) is the time step
- \(\mathbf{c}_b(\mathbf{q}^-)\) is the current constraint violation
- \(\boldsymbol{\Sigma}\) is the compliance matrix

The correction term \(\boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_b(\mathbf{q}^-)}{h}\) attempts to eliminate a fraction of the existing position error over the current time step.

### Contribution to Nonlinear System

The bilateral constraints contribute to the final nonlinear system as \(\mathbf{R}_b^{(\text{pos/vel})}(\mathbf{q}^+, \mathbf{u}^+, \boldsymbol{\lambda}_b^+) = \mathbf{0}\), where the root-finding function \(\mathbf{R}_b\) can be formulated as:

- **Position-level**: \(\mathbf{R}_b^{(\text{pos})} = \mathbf{c}_b(\mathbf{q}^+) + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+\)
- **Velocity-level**: \(\mathbf{R}_b^{(\text{vel})} = \mathbf{J}_b \cdot \mathbf{u}^+ + \boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_b(\mathbf{q}^-)}{h} + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+\)

Axion defaults to the position-level formulation to eliminate drift.

## Unilateral Constraints (Contacts)

Contact constraints prevent bodies from interpenetrating and are fundamentally different from bilateral constraints due to their unilateral (inequality) nature. These constraints involve complementarity conditions that require special treatment.

### Position-Level Formulation (Axion's Default)

From [Constraints Formulation](./constraints.md), contact constraints are defined as complementarity conditions:

\[
0 \leq \lambda_n \perp \mathbf{c}_{\text{contact}}(\mathbf{q}) \geq 0
\]

where \(\lambda_n\) is the normal contact impulse and \(\mathbf{c}_{\text{contact}}(\mathbf{q})\) is the gap function.

However, complementarity conditions cannot be directly solved by standard numerical methods. To convert this into a root-finding problem suitable for Newton's method, we use the **Fischer-Burmeister NCP-function**:

\[
\boldsymbol{\phi}_n(\mathbf{q}^+, \boldsymbol{\lambda}_n^+) = \boldsymbol{\lambda}_n^+ + \mathbf{c}_{\text{contact}}(\mathbf{q}^+) - \sqrt{(\boldsymbol{\lambda}_n^+)^2 + (\mathbf{c}_{\text{contact}}(\mathbf{q}^+))^2} = \mathbf{0}
\]

This reformulation turns the complementarity condition into a nonlinear equation that can be solved by Newton's method. The contact constraint equation in the nonlinear system becomes:

\[
\boldsymbol{\phi}_n(\mathbf{q}^+, \boldsymbol{\lambda}_n^+) = \mathbf{0}
\]

### Velocity-Level Formulation (Alternative)

Alternatively, contacts can be formulated at the velocity level. The basic velocity complementarity condition is:

\[
0 \leq \lambda_n \perp \mathbf{v}_n = \mathbf{J}_n \cdot \mathbf{u} \geq 0
\]

where \(\mathbf{v}_n\) is the relative normal velocity and \(\mathbf{J}_n = \frac{\partial \mathbf{c}_{\text{contact}}}{\partial \mathbf{q}}\) is the contact Jacobian.

To address drift, Axion can use **Baumgarte stabilization**. Additionally, **restitution** can be incorporated to model bouncing behavior:

\[
\mathbf{v}_n^+ = \mathbf{J}_n \cdot (\mathbf{u}^+ + e \cdot \mathbf{u}^- ) + \boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_{\text{contact}}(\mathbf{q}^-)}{h}
\]

where:

- \(e\) is the coefficient of restitution (0 ≤ e ≤ 1)

The restitution term ensures that separating contacts have the appropriate outgoing velocity based on the coefficient of restitution. When \(e = 0\) (perfectly inelastic), contacts stick; when \(e = 1\) (perfectly elastic), contacts bounce with no energy loss.

This modified velocity incorporates both position error correction and restitution effects. The complementarity condition becomes:

\[
0 \leq \lambda_n \perp \mathbf{v}_n^+ \geq 0
\]

Using the Fischer-Burmeister function:

\[
\boldsymbol{\phi}_n(\mathbf{u}^+, \boldsymbol{\lambda}_n^+) = \boldsymbol{\lambda}_n^+ + \mathbf{v}_n^+ - \sqrt{(\boldsymbol{\lambda}_n^+)^2 + (\mathbf{v}_n^+)^2} = \mathbf{0}
\]

### Contribution to Nonlinear System

The contact constraints contribute to the final nonlinear system as \(\mathbf{R}_n^{(\text{pos/vel})}(\mathbf{q}^+, \mathbf{u}^+, \boldsymbol{\lambda}_n^+) = \mathbf{0}\), where the root-finding function \(\mathbf{R}_n\) can be formulated as:

- **Position-level**: \(\mathbf{R}_n^{(\text{pos})} = \boldsymbol{\phi}_n(\mathbf{q}^+, \boldsymbol{\lambda}_n^+)\)
- **Velocity-level**: \(\mathbf{R}_n^{(\text{vel})} = \boldsymbol{\phi}_n(\mathbf{u}^+, \boldsymbol{\lambda}_n^+)\)

where \(\boldsymbol{\phi}_n\) is the Fischer-Burmeister transformation of the complementarity conditions, ensuring they are satisfied while making the system amenable to Newton-type solution methods.

## Friction Constraints

Friction constraints apply tangential forces that resist sliding motion between contacting bodies. Unlike bilateral and contact constraints, friction is formulated only at the velocity level using the **principle of maximal dissipation**.

### Velocity-Level Formulation (Only)

From [Constraints Formulation](./constraints.md), friction constraints are derived from the principle of maximal dissipation. The friction impulse \(\boldsymbol{\lambda}_t\) is constrained by the Coulomb friction law:

\[
\|\boldsymbol{\lambda}_t\| \leq \mu \cdot \lambda_n
\]

where \(\mu\) is the coefficient of friction and \(\lambda_n\) is the normal contact impulse.

The principle of maximal dissipation determines the friction impulse by solving an optimization problem that removes the maximum amount of kinetic energy from the system, subject to the Coulomb constraint. This leads to complementarity conditions that govern stick-slip behavior:

### Maximal Dissipation and KKT Conditions

The optimization problem results in Karush-Kuhn-Tucker (KKT) conditions that precisely define stick-slip behavior:

1. **Sliding**: If there is tangential velocity (\(|\mathbf{J}_f^T \cdot \mathbf{u}| > 0\)), the friction impulse opposes it at maximum magnitude (\(\|\boldsymbol{\lambda}_f\| = \mu \cdot \lambda_n\))
2. **Sticking**: If there is no tangential velocity (\(|\mathbf{J}_f^T \cdot \mathbf{u}| = 0\)), the friction impulse is whatever is needed to prevent motion, up to its maximum magnitude (\(\|\boldsymbol{\lambda}_f\| \leq \mu \cdot \lambda_n\))

The complete KKT conditions consist of both an equation and a complementarity condition:

**KKT Equation:**

\[
\mathbf{J}_f^T \cdot \mathbf{u} + \frac{|\mathbf{J}_f^T \cdot \mathbf{u}|}{|\boldsymbol{\lambda}_f|} \cdot \boldsymbol{\lambda}_f = \mathbf{0}
\]

**KKT Complementarity:**

\[
0 \leq |\mathbf{J}_f^T \cdot \mathbf{u}| \perp \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\| \geq 0
\]

However, this system cannot be directly inserted into a standard root-finding solver, as it involves inequalities and non-smooth functions.

### Transformation to Solvable System

Axion achieves this transformation through a two-step process:

**Step 1: Convert Complementarity to Root-Finding**  
The complementarity condition is converted into a nonlinear equation using an **NCP-function** \(\psi_f\) (Fischer-Burmeister):

\[
\psi_f(|\mathbf{J}_f^T \cdot \mathbf{u}|, \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\|) = 0
\]

This equation is mathematically equivalent to the original complementarity problem.

**Step 2: Create Symmetric System via Fixed-Point Iteration**  
A direct linearization would result in an asymmetric system matrix, which is inefficient to solve. To avoid this, a **fixed-point iteration** introduces a scalar "friction compliance" term \(W\) that is updated at each Newton iteration:

\[
W = \frac{|\mathbf{J}_f^T \cdot \mathbf{u}| - \psi_f(|\mathbf{J}_f^T \cdot \mathbf{u}|, \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\|)}{\|\boldsymbol{\lambda}_f\| + \psi_f(|\mathbf{J}_f^T \cdot \mathbf{u}|, \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\|)}
\]

This term is constructed so that when the Newton method converges, the original complementarity conditions are satisfied. The friction constraint for each contact simplifies to:

\[
\boldsymbol{\phi}_f(\mathbf{u}^+, \boldsymbol{\lambda}_f^+) = \mathbf{J}_f^T \cdot \mathbf{u}^+ + \mathbf{W} \cdot \boldsymbol{\lambda}_f^+ = \mathbf{0}
\]

where \(\mathbf{W}\) is a diagonal matrix containing the scalar \(W\) values for each contact, treated as constant during linearization within a single Newton step.

### Benefits of This Formulation

This elegant approach provides two major advantages:

- **Accurate modeling**: Correctly represents the smooth, isotropic Coulomb friction cone without approximation
- **Computational efficiency**: Results in a **symmetric system of equations**, enabling the use of highly efficient iterative solvers like Preconditioned Conjugate Residual on the Schur complement

### Contribution to Nonlinear System

The friction constraints contribute to the final nonlinear system as \(\mathbf{R}_f^{(\text{vel})}(\mathbf{u}^+, \boldsymbol{\lambda}_f^+) = \mathbf{0}\), where the root-finding function is:

\[
\mathbf{R}_f^{(\text{vel})} = \boldsymbol{\phi}_f(\mathbf{u}^+, \boldsymbol{\lambda}_f^+) = \mathbf{J}_f^T \cdot \mathbf{u}^+ + \mathbf{W} \cdot \boldsymbol{\lambda}_f^+
\]

Here \(\boldsymbol{\phi}_f\) incorporates the Fischer-Burmeister function and fixed-point iteration to ensure the friction complementarity conditions are satisfied while maintaining the solvable form for Newton-type methods.

## System Assembly Components

Now that we have established how each constraint type contributes to the nonlinear system, we need two additional components to complete the formulation: kinematic mapping for quaternion-based rotations and time discretization for numerical integration.

### Kinematic Mapping

To support quaternion-based rotation representation, a kinematic mapping \(\mathbf{G}\) transforms generalized velocities \(\mathbf{u}\) into configuration derivatives:

\[
\mathbf{\dot{q}} = \mathbf{G}(\mathbf{q}) \cdot \mathbf{u}
\]

For rigid bodies with position \(\mathbf{x}\) and quaternion orientation \(\boldsymbol{\theta}\), this maps 6-DOF velocities (3 translational, 3 angular) to 7-element configuration derivatives.

### Time Discretization

The kinematic mapping enables implicit time integration:

\[
\mathbf{q}^+ = \mathbf{q}^- + h \cdot \mathbf{G} \cdot \mathbf{u}^+
\]

where \(h\) is the time step. The kinematic equation becomes another constraint in our system:

\[
\mathbf{q}^+ - \mathbf{q}^- - h \cdot \mathbf{G} \cdot \mathbf{u}^+ = \mathbf{0}
\]

## Complete Nonlinear System

Combining all constraint types and system components, we obtain the complete discrete-time nonlinear system:

\[
\begin{align}
\text{Dynamics:} \quad & \mathbf{\tilde{M}} \cdot \frac{\mathbf{u}^+ - \mathbf{\tilde{u}}}{h} - \mathbf{J}_b^T \cdot \boldsymbol{\lambda}_b^+ - \mathbf{J}_n^T \cdot \boldsymbol{\lambda}_n^+ - \mathbf{J}_f^T \cdot \boldsymbol{\lambda}_f^+ = \mathbf{0} \\
\text{Bilateral:} \quad & \mathbf{R}_b^{(\text{pos/vel})}(\mathbf{q}^+, \mathbf{u}^+, \boldsymbol{\lambda}_b^+) = \mathbf{0} \\
\text{Contacts:} \quad & \mathbf{R}_n^{(\text{pos/vel})}(\mathbf{q}^+, \mathbf{u}^+, \boldsymbol{\lambda}_n^+) = \mathbf{0} \\
\text{Friction:} \quad & \mathbf{R}_f^{(\text{vel})}(\mathbf{u}^+, \boldsymbol{\lambda}_f^+) = \mathbf{0} \\
\text{Kinematics:} \quad & \mathbf{q}^+ - \mathbf{q}^- - h \cdot \mathbf{G} \cdot \mathbf{u}^+ = \mathbf{0}
\end{align}
\]

### System Components

- **Unknowns**: Positions \(\mathbf{q}^+\), velocities \(\mathbf{u}^+\) and constraint impulses \(\boldsymbol{\lambda}^+ = [\boldsymbol{\lambda}_b^+, \boldsymbol{\lambda}_n^+, \boldsymbol{\lambda}_f^+]\)
- **Scaled quantities**:

\[
\mathbf{\tilde{M}} = \mathbf{G}^T \cdot \mathbf{M} \cdot \mathbf{G},
\]

\[
\mathbf{\tilde{u}} = \mathbf{u}^- + h \cdot \mathbf{G}^T \mathbf{f}(\mathbf{q}^-, \mathbf{u}^-)
\]

- **Root-finding functions**: \(\mathbf{R}_b\), \(\mathbf{R}_n\), \(\mathbf{R}_f\) represent the constraint enforcement equations, formulated at position level (pos) or velocity level (vel) as described in previous sections
- **Matrix evaluation**: All system matrices (\(\mathbf{J}_b\), \(\mathbf{J}_n\), \(\mathbf{J}_f\), \(\boldsymbol{\Sigma}\), etc.) are evaluated using the configuration \(\mathbf{q}^-\) from the previous time step

This system represents all physical laws simultaneously:

- **Gauss's principle** (dynamics equation)
- **Constraint enforcement** (bilateral, contact, friction)  
- **Implicit time integration** (kinematics)

The unified formulation ensures that all constraints are satisfied exactly while maintaining numerical stability and eliminating drift.

## Numerical Solution

This nonlinear system cannot be solved analytically and requires iterative numerical methods. Axion employs a specialized Newton-type approach designed to handle the non-smooth nature of contact and friction constraints.

→ **Next**: [Numerical Solution](./linear-system.md)
