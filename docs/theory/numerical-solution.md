# Numerical Solution

This section describes how Axion numerically solves the nonlinear system of equations derived in [Nonlinear System](./non-linear-system.md). The approach uses a specialized Newton-type method designed to handle the non-smooth nature of contact and friction constraints.

## Newton's Method for Non-Smooth Systems

The nonlinear system from the previous section must be solved iteratively. Axion employs a **Non-Smooth Newton Method** that can handle the discontinuities arising from contact and friction.

The typical Newton iteration with index \(n\) can be written as:

\[
    \mathbf{x}^{n+1} = \mathbf{x}^n - \mathbf{A}^{-1}(\mathbf{x}^n)\mathbf{F}(\mathbf{x}^n) \;,\quad(1)
\]

where \(\mathbf{F}(\mathbf{x}^n) = 0\) represents the nonlinear system of equations and \(\mathbf{A}(\mathbf{x}^n)\) represents the generalized Jacobian matrix that linearizes the system at the current state.

## Generalized System Jacobian

For the linearized system, Axion computes the generalized Jacobian: \(\mathbf{A}(\mathbf{x}^n) = \mathbf{J_F}(\mathbf{x}^n) \equiv \frac{\partial \mathbf{F}}{\partial \mathbf{x}}(\mathbf{x}^n)\). This matrix contains partial derivatives evaluated at the current system state.

!!! info "Generalized Jacobian for Non-Smooth Functions"
    Since contact and friction constraints create non-smooth functions, the Jacobian is a *generalized Jacobian*. At non-smooth points, this represents the convex hull of all directional derivatives, allowing Newton's method to handle discontinuities in the constraint functions.

### Block Structure

The system Jacobian has a structured block form that reflects the different types of constraints:

\[  \mathbf{J_F} =
    \begin{bmatrix}
        \mathbf{\tilde{M}} & -\mathbf{J}_b^\top & -\mathbf{J}_n^\top & -\mathbf{J}_f^\top \\
        \mathbf{J}_b & \mathbf{E} & \mathbf{0} & \mathbf{0} \\
        \mathbf{J}_n & \mathbf{0} & \mathbf{S} & \mathbf{0} \\
        \mathbf{J}_f & \mathbf{0} & \mathbf{0} & \mathbf{W}
    \end{bmatrix}\;, \quad(2)
\]

where:

- \(\mathbf{E}\) is the compliance matrix for bilateral constraints
- \(\mathbf{S} = \frac{\partial \mathbf{\phi}_n}{\partial \mathbf{\lambda}_n}\) and \(\mathbf{W} = \frac{\partial \mathbf{\phi}_f}{\partial \mathbf{\lambda}_f}\) are partial

derivatives of the normal and friction [NCP-functions](./non-linear-system.md#nonlinear-complementarity)

### Linear System for Newton Step

The complete linear system for each Newton iteration is:

\[  
    \begin{bmatrix}
        \mathbf{\tilde{M}} & -\mathbf{J}_b^\top & -\mathbf{J}_n^\top & -\mathbf{J}_f^\top \\
        \mathbf{J}_b & \mathbf{E} & \mathbf{0} & \mathbf{0} \\
        \mathbf{J}_n & \mathbf{0} & \mathbf{S} & \mathbf{0} \\
        \mathbf{J}_f & \mathbf{0} & \mathbf{0} & \mathbf{W}
    \end{bmatrix}\;
    \begin{bmatrix}
        \Delta\mathbf{u} \\
        \Delta\mathbf{\lambda_b} \\
        \Delta\mathbf{\lambda_n} \\
        \Delta\mathbf{\lambda_f} \\
    \end{bmatrix} =
    -\begin{bmatrix}
        \mathbf{g} \\
        \mathbf{h}_b \\
        \mathbf{h}_n \\
        \mathbf{h}_f
    \end{bmatrix}   \quad(3)
\]

where:

- Left-hand side: Updates to velocities \(\Delta\mathbf{u}\) and constraint impulses \(\Delta\boldsymbol{\lambda}\)
- Right-hand side: Current residuals of the nonlinear system

The function [`compute_linear_system`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linear_utils.py#L82-L218){:target = "_blank"} efficiently assembles this Jacobian using GPU-accelerated Warp kernels.  

## Efficient Solution Strategy

The large, structured linear system requires specialized solution techniques. Rather than solving directly, Axion uses **Schur complement reduction** to exploit the block structure and create a smaller, better-conditioned system.

### Schur Complement Reduction

More specifically, one can rewrite the system (3) into a more compact form,

\[
    \begin{bmatrix}
        \mathbf{\tilde{M}} & -\mathbf{J}^\top \\
        \mathbf{J} & \mathbf{C}
    \end{bmatrix}
    \begin{bmatrix}
        \Delta\mathbf{u} \\
        \Delta\mathbf{\lambda} \\
    \end{bmatrix}
    =
    -\begin{bmatrix}
        \mathbf{g} \\
        \mathbf{h}
    \end{bmatrix} \;,  \quad(4)
\]

where:

- \(\mathbf{J} = \left[ \mathbf{J}_b \; \mathbf{J}_n \; \mathbf{J}_f \right]^\top\) (combined constraint Jacobian)
- \(\Delta\boldsymbol{\lambda} = \left[\Delta\boldsymbol{\lambda}_b \; \Delta\boldsymbol{\lambda}_n \; \Delta\boldsymbol{\lambda}_f \right]^\top\) (all constraint impulse updates)
- \(\mathbf{C} = \text{diag}( \mathbf{E}, \mathbf{S}, \mathbf{W})\) (compliance block)

The original system can be indefinite and ill-conditioned. To obtain a better-conditioned positive semi-definite system, we eliminate the velocity variables using the Schur complement with respect to \(\mathbf{\tilde{M}}\):

\[
    \left[\mathbf{J \tilde{M}^{-1} J^\top + C } \right] \Delta\lambda =
    \left( \mathbf{J \tilde{M}^{-1} g - h}  \right)\;. \quad (5)
\]

The [`JacobiPreconditioner`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/preconditioner.py){:target = "_blank"} class efficiently computes the Schur complement matrix \(\mathbf{J \tilde{M}^{-1} J^\top + C}\).

### Solving the Reduced System

The Schur complement system (5) is then solved using the Conjugate Residual (CR) algorithm with Jacobi preconditioning. This iterative approach is well-suited for the symmetric positive semi-definite structure of the reduced system.

### Linear system solving pipeline

The overall computation is implemented in the [`simulate`](https://github.com/aleskucera/axion/blob/main/src/axion/core/engine.py#L123-L176){:target = "_blank"} method of the class [`AxionEngine`](https://github.com/aleskucera/axion/blob/main/src/axion/core/engine.py){:target = "_blank"}. The computation is executed *roughly* as follows:

1. [`compute_linear_system`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linear_utils.py#L82-L218){:target = "_blank"} assembles the linear system effectively using Warp kernels.
2. [`self.preconditioner.update`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/preconditioner.py#L81-L97){:target = "_blank"} computes the Schur complement matrix.
3. The Schur complement system is solved using the Conjugate Residual algorithm with Jacobi preconditioning via [`cr_solver`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/cr.py#L62-L158){:target = "_blank"}.
4. Because the solution of equation (5) is only the vector of impulses \(\Delta\mathbf{\lambda}\), the generalized velocities are calculated as
\(
    \Delta\mathbf{u =  \tilde{M}^{-1} \left( J^\top \Delta\lambda - g \right)}\;.
\)
This is done in the [`compute_delta_body_qd_from_delta_lambda`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linear_utils.py#L189-L218){:target = "_blank"} function.
5. Optionally, the Line search algorithm (implemented via [`perform_linesearch`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linesearch_utils.py#L48-L151){:target = "_blank"} function) can be executed to find the best parameter \(t\),

\[
    \begin{align}
    \lambda^{n+1} &= \lambda^{n} + t\Delta\lambda \\
    \mathbf{u}^{n+1} &= \mathbf{u}^{n} + t\Delta\mathbf{u} \;.
    \end{align}
\]

The full description of the implementation of the [`simulate`](https://github.com/aleskucera/axion/blob/main/src/axion/core/engine.py#L123-L176){:target = "_blank"} method can be found in [Engine API](./../implementation/engine.md).
