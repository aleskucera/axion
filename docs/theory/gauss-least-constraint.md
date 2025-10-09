# Gauss's Principle of Least Constraint

## Overview
The principle of least constraint describes how the acceleration of a physical system bounded by constraints relates to an unconstrained system. It frames the constraint enforcement as an **optimization problem** - it states that at any given time, the true accelerations \( \mathbf{a} \) of a constrained system are those that minimize the "constraint effort," measured as the mass-weighted squared deviation from the accelerations \( \mathbf{a}_{\text{unconstrained}} \) the system would experience if no constraints were present:

\[
    Z_{\text{accel}}(\mathbf{a}) = \frac{1}{2} \| \mathbf{a} - \mathbf{a}_{\text{unconstrained}} \|^2_{\mathbf{\tilde{M}}},\quad (1)
\]

where $\|\mathbf{v}\|^2_{\mathbf{\tilde{M}}} = \mathbf{v}^\top \mathbf{\tilde{M}} \mathbf{v}$ denotes the mass-weighted squared norm. The full optimization problem then becomes:

\[
\begin{equation}
\begin{aligned}
\min_{\mathbf{a}} \quad & Z_{\text{accel}}(\mathbf{a}) \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad(2) \\
\textrm{s.t.} \quad & \text{Acceleration-level constraints are satisfied} \;.
\end{aligned}
\end{equation}
\]

## Reformulation into a simulator-friendly form
The formulation of Gauss's principle in (2) is elegant but so far not suitable to use in a numerical setting. To adhere to the principle, the following changes need to be made.

- We need to discretize the optimization cost (1), so that it fits within simulator's time-stepping scheme.
- The optimization problem (2) is acceleration based. As Axion operates predominantly on the velocity level, the problem needs to be reformulated via generalized velocities \(\mathbf{u}\).

The acceleration over a time step \( h \) is approximated using a first-order difference involving velocities at the start (-) and at the end (+) of the step:

\[
    \mathbf{a}^+ \approx \frac{\mathbf{u^+ - u^-}}{h} \;.   \quad (3)
\]

The unconstrained acceleration is approximated using external forces \(\mathbf{f}^-\) at the beginning of the time step:

\[
\mathbf{a}_{\text{unconstrained}}^+ \approx \mathbf{\tilde{M}}^{-1} \mathbf{f}^-.   \quad(4)
\]

Substituting these approximations into the continuous objective function in (1) and pre-multiplying it by \(h^2\) (to arrive at an impulse formulation) yields

\[
    Z_{\text{vel}}(\mathbf{u}^+) = \frac{1}{2} \left\| \mathbf{u}^{+} - (\mathbf{u}^- + h \mathbf{\tilde{M}}^{-1} \mathbf{f}^-) \right\|^2_{\mathbf{\tilde{M}}}.    \quad (5)
\]

We can define \( \mathbf{u}_{\text{free}}^{+} = \mathbf{u}^- + h \mathbf{\tilde{M}}^{-1} \mathbf{f}^- \) as the predicted velocity at the end of the time interval that would result from applying only the external forces \( \mathbf{f}^- \) over the time step \( h \). 

Note that from the constrained velocities \(\mathbf{u}^+\), the actual positions are updated using an integration rule,

\[
    \mathbf{q^+ \approx q^-} + h\mathbf{G(q^-)u}^+ \quad (6)
\]

where \(\mathbf{G(q^-)}\) is the [kinematic mapping](./non-linear-system.md#kinematic-mapping).

## Solution of the optimization problem
Utilizing the steps in the previous section, the full optimization problem (2) is given at the velocity level as

\[
    \begin{align}
    \min_{\mathbf{u}^{+}} \quad & Z_{\text{vel}}(\mathbf{u}^{+}) = \frac{1}{2} \left\| \mathbf{u}^{+} - (\mathbf{u}^- + h\mathbf{\tilde{M}}^{-1}\mathbf{f}^-) \right\|^2_{\mathbf{\tilde{M}}} \quad\quad (7)\\
    \textrm{s.t.} \quad & \mathbf{J}_b^- \mathbf{u}^{+} = \mathbf{0} \\
    & \mathbf{J}_u^- \mathbf{u}^{+} \geq \mathbf{0} \;.
    \end{align}
\]

Here, \( \mathbf{J}_b^- \mathbf{u}^{+} = \mathbf{0} \) and \( \mathbf{J}_u^- \mathbf{u}^{+} \geq \mathbf{0} \) are the [bilateral and unilateral constraints](./constraints.md#unilateral-and-bilateral-constraints) respectively. Interestingly, if we focus on the simpler case involving only bilateral type constraints, it becomes a quadratic programme with equality constraints. Such a problem can be solved efficiently using **the method of Lagrange multipliers**. The method yields

\[
    \begin{align}
    \mathbf{\tilde{M}} \mathbf{u}^{+} + (\mathbf{J}^-)^\top \boldsymbol{\lambda}^{+} &= \mathbf{\tilde{M}} \mathbf{u}^- + h \mathbf{f}^- \quad (8)\\
    \mathbf{J}^- \mathbf{u}^{+} &= \mathbf{0} \quad\quad\quad\quad\quad\;\;\; (9)
    \end{align}
\]

where \( \lambda^+ \) are the Lagrange multipliers. Equations (8) and (9) together create a coupled system of linear equations for the unknowns
\( \mathbf{u}^+ \) and \( \lambda^+ \). Equations (8) and (9) form a baseline of the complete nonlinear system of equations described in [Nonlinear system](./non-linear-system.md).

## Further Reading
- Continue by reading about specific constraints in [Constraints Formulation](./constraints.md).
- Read about how Axion further treats the underlying nonlinear system in [Nonlinear system](./non-linear-system.md).