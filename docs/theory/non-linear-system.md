# TBD

- From constraints, how do we actually create the non-linear system of equations (in paper from 3.5 Governing Equations to equations 46.-50.)

# Non-linear system
This section describes how the continuous system of nonlinear equations with bilateral and unilateral constraints is transformed into a discretized system, which can be fed into the numerical solver.

## Need for reformulation
The combination of differential equation governing the [dynamics](./constraints.md#the-unified-dynamics-equation) together with equality and complementarity conditions arising from the [constraints](./constraints.md#1-the-unified-constraint-formulation) creates a problem that is in general:
- non-convex,
- in a form not suitable for a numerical solver,
- not discretized.

The following steps address these issues.

## Nonlinear complementarity
The complementarity conditions in the form $ 0 \le a \perp b \ge 0$ can be reformulated using a NCP-function $\phi(a,b)$ whose roots satisfy the original complementarity conditions, i.e.:
$$
    \phi(a,b) \iff 0 \le a \perp b \ge 0 \;.
$$
This reformulation turns the original problem with inequality-type constraints into a root-finding one. 

Axion uses the **Fischer-Burmeister function** NCP-function:
$$
\phi_{FB}(a,b) = a + b - \sqrt{a^2 + b^2} = 0 \;,
$$
implemented in [utils.py](/src/axion/constraints/utils.py).

Specifically, the constraints that need reformulation via the $\phi_{FB}$ function are:
- [contact constraints](./constraints.md#2-contact-constraints),
- [frictional constraints](./constraints.md#3-friction-constraints).

The reformulation is described in depth in [Macklin et al. 2019](https://arxiv.org/abs/1907.04587v1).

## Kinematic mapping
In order to support quaternion-based description of rotation (common for rigid body calculations), a kinematic mapping $\bold{G}$ has to be defined. This mapping transforms the generalized velocity $\bold{u}$ into $\bold{\dot{q}}$ via:
$$
\bold{\dot{q}} = \bold{G(q)u} \;.
$$.
If $\bold{q} = [\; \bold{x} \; \bold{\theta} \; ]^T $ is a 7-element vector where $\bold{\theta}$ is the 4-tuple of quaternions, and $\bold{u} = [\; \bold{\dot{x} \; \bold{\omega} \;}]^T$ is the 6-element vector of generalized velocity, then $\bold{G}$ has to be a 7-by-6 matrix. 

## Time discretization
Taking into account the kinematic mapping, the simulation update for the generalized coordinates in terms of generalized velocities is:
$$
    \bold{q}^+ = \bold{q}^- + h \bold{G(q^+)}\bold{u^+}\;,
$$
where $h$ is the time step length.

.... final non-linear system here..
