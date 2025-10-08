# TBD

- essentially system assembly system 6.1


# Linear system assembly and computation
The typical iteration of the Newton method with iteration index \(n\) can be written as
\[
    \bold{x}^{n+1} = \bold{x}^n - \bold{A}^{-1}(\bold{x}^n)\bold{r}(\bold{x}^n) \;,
\]
where \(\bold{r(x)} = 0\) represents the nonlinear system of equations described in [Nonlinear system](./non-linear-system.md) and matrix \( \bold{A} \) represents the current linearization of the system - matrix of partial derivatives evaluated at the current system state.    
