# TBD

- essentially system assembly system 6.1


# Linear system assembly and computation

## Newton's method
The typical iteration of the Newton method with iteration index \(n\) can be written as

\[
    \mathbf{x}^{n+1} = \mathbf{x}^n - \mathbf{A}^{-1}(\mathbf{x}^n)\mathbf{F}(\mathbf{x}^n) \;,
\]

where \(\mathbf{F}(\mathbf{x}^n) = 0\) represents the nonlinear system of equations described in [Nonlinear system](./non-linear-system.md) and matrix \( \mathbf{A}(\mathbf{x}^n) \) represents the current linearization of the system.

## System Jacobian
Axion uses the Jacobian matrix for the linearized system, i.e. \( \mathbf{A}(\mathbf{x}^n) = \mathbf{J_F}(\mathbf{x}^n) \equiv \frac{\partial \mathbf{F}}{\partial \mathbf{x}}(\mathbf{x}^n)\). Meaning, \( \mathbf{A}(\mathbf{x}^n) \) is a matrix of partial derivatives evaluated at the current system state.

!!! info "Partial derivatives of a non-smooth function"
    In the non-smooth case, the Jacobian is the *generalized Jacobian*. Intuitively, it can be viewed as the convex hull of all directional derivatives at the non-smooth point.

The System Jacobian used by Axion has the following block structure:

\[  \mathbf{J_F} = 
    \begin{bmatrix}
        \mathbf{\tilde{M}} & -\mathbf{J}_b^\top & -\mathbf{J}_n^\top & -\mathbf{J}_f^\top \\
        \mathbf{J}_b & \mathbf{E} & \mathbf{0} & \mathbf{0} \\
        \mathbf{J}_n & \mathbf{0} & \mathbf{S} & \mathbf{0} \\
        \mathbf{J}_f & \mathbf{0} & \mathbf{0} & \mathbf{W}
    \end{bmatrix}\;,
\]

where \(\mathbf{E}\) is the compliance matrix and \( \mathbf{S} = \frac{\partial \mathbf{\phi}_n}{\partial \mathbf{\lambda}_n}, \mathbf{W} = \frac{\partial \mathbf{\phi}_f}{\partial \mathbf{\lambda}_f}\) are the partial derivatives of the normal and frictional [NCP-functions](./non-linear-system.md#nonlinear-complementarity). The complete set of linear equations of the Newton iterate is then given by the form 

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
    \end{bmatrix} = \mathbf{r}
\]

where the left hand side vector is the vector of unknowns and \(\mathbf{r}\) is the vector of residuals. The function [`compute_linear_system`](/src/axion/core/engine.py) implemented in `linear_utils.py` performs the Jacobian assembly.  

## Solving the Linear system
via Schur complement preconditioner

