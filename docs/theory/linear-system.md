# TBD

- essentially system assembly system 6.1


# Linear system assembly and computation

## Newton's method
The typical iteration of the Newton method with iteration index \(n\) can be written as

\[
    \mathbf{x}^{n+1} = \mathbf{x}^n - \mathbf{A}^{-1}(\mathbf{x}^n)\mathbf{F}(\mathbf{x}^n) \;,\quad(1)
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
    \end{bmatrix}\;, \quad(2)
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
    \end{bmatrix} = 
    -\begin{bmatrix}
        \mathbf{g} \\
        \mathbf{h}_b \\
        \mathbf{h}_n \\
        \mathbf{h}_f
    \end{bmatrix}   \quad(3)
\]

where the left hand side vector is the vector of unknowns (differeneces in generalized velocities and constraint impulses) and the right hand side is the vector of residuals. The function [`compute_linear_system`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linear_utils.py#L82-L218){:target = "_blank"} performs the Jacobian assembly via Warp kernels.  

## Solving of the Linear System
The linear system is not solved directly - the solving process is preceded by a **preconditioning** procedure. Axion implements a Jacobi preconditioner constructed via Schur complement. 

### Preconditioner
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

where \( \mathbf{J} = \left[ \mathbf{J}_b \; \mathbf{J}_n \; \mathbf{J}_f \right]^\top \), \( \Delta\mathbf{\lambda} = \left[\Delta\mathbf{\lambda_b} \; \Delta\mathbf{\lambda_n} \; \Delta\mathbf{\lambda_f} \right]^\top \), \( \mathbf{h} = \left[ \mathbf{h}_b \; \mathbf{h}_n \; \mathbf{h}_f \right]^\top \) and \(\mathbf{C} = \text{diag}( \mathbf{E}, \mathbf{S}, \mathbf{W}) \) is the compliance sub-block. The system (4) can be indefinite and possibly singular. To obtain a reduced positive semi-definite system, we take the Schur complement with respect to the matrix \( \mathbf{\tilde{M}} \) to obtain

\[
    \left[\mathbf{J \tilde{M}^{-1} J^\top + C } \right] \Delta\lambda = 
    \left( \mathbf{J \tilde{M}^{-1} g - h}  \right)\;. \quad (5)
\]

The class [`JacobiPreconditioner`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/preconditioner.py){:target = "_blank"} implements the preconditioning, i.e, the computation \( \mathbf{J \tilde{M}^{-1} J^\top + C }\).

### Linear system solving pipeline
The overall computation is implemented in the [`simulate`](https://github.com/aleskucera/axion/blob/main/src/axion/core/engine.py#L123-L176){:target = "_blank"} method of the class [`AxionEngine`](https://github.com/aleskucera/axion/blob/main/src/axion/core/engine.py){:target = "_blank"}. The computation is executed *roughly* as follows:

1. [`compute_linear_system`](https://github.com/aleskucera/axion/blob/main/src/axion/core/linear_utils.py#L82-L218){:target = "_blank"} assembles the linear system effectively using Warp kernels.
2. [`self.preconditioner.update`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/preconditioner.py#L81-L97){:target = "_blank"} is called in order to update the Jacobians or the compliance term.
3. The preconditioiner is applied and an approximate solution to a symmetric, positive-definite linear system using the Conjugate Residual algorithm is computed by calling the [`cr_solver`](https://github.com/aleskucera/axion/blob/main/src/axion/optim/cr.py#L62-L158){:target = "_blank"} function.
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
