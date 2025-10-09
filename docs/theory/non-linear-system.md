# Nonlinear system
This section describes how the continuous system of nonlinear equations with bilateral and unilateral constraints is transformed into a discretized system, which can be fed into the numerical solver.

## Need for reformulation
The combination of differential equation governing the [dynamics](./constraints.md#the-unified-dynamics-equation) together with equality and complementarity conditions arising from the [constraints](./constraints.md#1-the-unified-constraint-formulation) creates a problem that is in general:
- non-convex,
- in a form not suitable for a numerical solver,
- not discretized.

The following steps address these issues.

## Nonlinear complementarity
The complementarity conditions in the form \( 0 \le a \perp b \ge 0\) can be reformulated using a NCP-function \(\phi(a,b)\) whose roots satisfy the original complementarity conditions, i.e.:

\[
    \phi(a,b) \iff 0 \le a \perp b \ge 0 \;.
\]

This reformulation turns the original problem with inequality-type constraints into a root-finding one. 

Axion uses the **Fischer-Burmeister function** NCP-function:

\[
\phi_{FB}(a,b) = a + b - \sqrt{a^2 + b^2} = 0 \;,
\]

implemented in [`scaled_fisher_burmeister_new`](https://github.com/aleskucera/axion/blob/main/src/axion/constraints/utils.py#L27-L45).

Specifically, the constraints that need reformulation via the \( \phi_{FB} \) function are:

- [contact constraints](./constraints.md#2-contact-constraints),
- [frictional constraints](./constraints.md#3-friction-constraints).

The reformulation is described in depth in [Macklin et al. 2019](https://arxiv.org/abs/1907.04587v1).

## Kinematic mapping
In order to support quaternion-based description of rotation (common for rigid body calculations), a kinematic mapping \( \mathbf{G} \) has to be defined. This mapping transforms the generalized velocity \(\mathbf{u}\) into \(\mathbf{\dot{q}}\) via:

\[
\mathbf{\dot{q}} = \mathbf{G(q)u} \;.
\]

If \(\mathbf{q} = [\; \mathbf{x} \; \mathbf{\theta} \; ]^T \) is a 7-element vector where \( \mathbf{\theta} \) is the 4-tuple of quaternions, and \(\mathbf{u} = [\; \mathbf{\dot{x} \; \mathbf{\omega} \;}]^T \) is the 6-element vector of generalized velocity, then \(\mathbf{G} \) has to be a 7-by-6 matrix. 

## Time discretization
Taking into account the kinematic mapping, the simulation update for the generalized coordinates in terms of generalized velocities is:

\[
    \mathbf{q}^+ = \mathbf{q}^- + h \mathbf{G(q^+)}\mathbf{u^+}\;,
\]

where \(h\) is the time step length and the superscript "-" notes the values in previous step and "+" notes the values in the next step. This makes the simulation **implicit**.

By applying the previous reformulations one gets non-smooth the discrete-time nonlinear system with equality constraints only:

\[
\begin{align}
    \nonumber
    {\mathbf{\tilde{M}}\left( \frac{\mathbf{u}^+-\mathbf{\tilde{u}}}{h}  \right) - \mathbf{J^T_b(q^+) - J^T_n(q^+)\lambda^+_n - J^T_f(q^+)\lambda^+_f = 0} } \\ 
    \nonumber
    \mathbf{c_b(q^+) + E(q^+)\lambda^+_b = 0} \\
    \nonumber
    \mathbf{\phi_n(q^+, \lambda^+) = 0}\\
    \nonumber
    \mathbf{\phi_f(u^+, \lambda^+) = 0}\\
    \nonumber
    \mathbf{q^+ - q^-}-h\mathbf{Gu^+ = 0}
\end{align}
\]

Here, \( \mathbf{u^+, \lambda^+} \) are the unknown velocities and multipliers at the end of the time step. Description of the symbols:

- normal  and friction NCP-functions for all contacts are grouped into the 2 vectors, i.e. \( \mathbf{\phi_n = [ \phi_{n,1}, ... , \phi_{n,nc}} ]^T \) 
- mass matrix and Jacobians are scaled with respect to the kinematic mapping, \( \mathbf{\tilde{M} = G^T MG},\;\mathbf{J_b = \nabla c_b G} \), ... 
- constant \( \mathbf{\tilde{u} = u^-} + h\mathbf{G^T f(q^-,\dot{q}^-)}  \) is the unconstrained velocity.

Each symbol is discussed more in depth in [Macklin et al. 2019](https://arxiv.org/abs/1907.04587v1). This discretized system is handed to the non-smooth Newton solver, which linearizes it in each of its iteration, read [Linear system](./linear-system.md). 


## Further reading
- Continue by reading about the underlying linear system assembled during the step of the Newton's method in [Linear system](./linear-system.md).
- More about specific constraints and their realization can be found in [Constraints Formulation](./constraints.md).