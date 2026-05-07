# Eisenstat-Walker Forcing Term: Adaptive Linear Tolerance

## Motivation

PCR runs once per Newton-Raphson iteration. Profiling shows it's
~85% of the step's wall time, and per-iteration logs show many PCR
calls hit `max_linear_iters = 16` — meaning the linear-tolerance bar
was tighter than achievable in 16 PCR iters, so we paid all 16 every
time.

Most of that work is wasted in the early NR iters. Newton's
linearization is built around the *current iterate*; if that iterate
is far from the solution, the linearized system has its own bias —
and solving it to machine precision just locks in error that the
next NR step will undo. Tight linear solves only convert into outer
NR progress when Newton is already close to convergence.

The **Eisenstat-Walker (1996)** forcing term is the classical
quantitative version of that intuition: scale the linear tolerance
by the *outer NR residual ratio*, so early iters tolerate sloppy
linear solves and only the final iters demand precision. Cited 2k+
times. Used in PETSc, Trilinos, KINSOL, and most production
nonlinear solvers.

## The math

At NR iter k we want to solve `A_k Δλ_k = b_k` where `b_k = -r_NR(λ_k)`.
Standard PCR exits when

    ‖A_k Δλ_k - b_k‖ ≤ max(atol, η · ‖b_k‖)

where `η` is the *forcing term*. Today `η = linear_tol = 1e-5`,
constant.

Eisenstat-Walker's two choices:

**Choice 1** (more sophisticated):

    η_k = ‖r_NR_k - (r_NR_{k-1} - A_{k-1} Δλ_{k-1})‖ / ‖r_NR_{k-1}‖

Numerator: how much did the *actual* outer residual differ from what
the linear model predicted? When Newton is converging well (model
predicts well), η_k is tiny. When it's stagnating (model is wrong),
η_k is closer to 1 (give up on tight solves until things move).
Maintains superlinear NR convergence.

**Choice 2** (simpler, more common):

    η_k = γ · (‖r_NR_k‖ / ‖r_NR_{k-1}‖)^α

with γ ∈ (0, 1] and α ∈ (1, 2]. Just scales by the residual reduction
rate. Maintains Q-quadratic NR convergence with appropriate bounds.

Standard parameters: `γ = 0.9`, `α = 2.0`. Bound `η_k ∈ [η_min, η_max]`
to prevent both overshoot (η too large → no progress) and underflow
(η too small → asking PCR for impossible precision). Typical bounds:
`η_min = 1e-10`, `η_max = 0.99`.

Concretely, at iter k:

    ratio = ‖r_NR_k‖ / ‖r_NR_{k-1}‖
    η_k = clip(0.9 · ratio², 1e-10, 0.99)
    PCR.solve(tol = η_k, atol = linear_atol, ...)

Worked example with `α = 2`:
* outer residual halves: ratio=0.5, η=0.9·0.25=0.225 (loose)
* outer residual goes 1 → 1e-3 (great progress): η=0.9·1e-6=9e-7 (very tight)
* outer residual stagnates: ratio=1, η=0.9 (give up on precision)

There's also a "safeguard": if `η_k` is much smaller than the
previous one, *don't* let it shrink too fast. The standard rule
(used in PETSc):

    η_k = max(η_k, γ · η_{k-1}^α)   if γ · η_{k-1}^α > 0.1

Prevents the "honeymoon" effect where one good iter
over-tightens and subsequent iters can't keep up.

## Why this fits our solver

Three reasons specific to the Axion setup:

1. **PCR call already takes `tol` as a parameter** —
   `pcr_solver.py:177`. We just pass a different number per NR iter.
   No changes to the PCR kernel. The convergence check inside PCR
   (`_check_residuals_kernel` at `pcr_solver.py:85`) already uses
   `tol_sq = float(tol**2)` with proper relative-tolerance semantics
   (`max(atol_sq, b_sq * tol_sq)`). Drop-in.

2. **Outer residual is already computed each iter** —
   `data.res_norm_sq` is updated by
   `tiled_sq_norm.compute(res.full, res_norm_sq)` after every NR
   step. We have the full ratio for free.

3. **PCR consistently hits max_linear_iters in early NR iters** —
   visible in the per-component profile log. Those are the iters
   where the forcing-term technique has the most slack to harvest.

## Risks specific to FB-NR

The classical convergence proofs assume smooth Newton on a smooth
problem. FB has C¹ but non-C² behavior at the cone corner — the
same regime that broke the iterate-seeding warm-start (see
[`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)).
Three risks worth measuring:

1. **Loose linear solves near the corner could amplify
   FB-degeneracy.** When the FB Jacobian is nearly rank-deficient,
   sloppy linear solves might produce search directions that backtracking
   can't redeem. If iter-0 of an impact step gets `η ≈ 0.9`, the Δλ
   is approximate — and the impact regime is exactly where Newton
   needs accurate steps.

   *Mitigation*: lower-bound `η_max` more aggressively (e.g., 0.1
   instead of 0.99) for the first few NR iters. Hybrid scheme.

2. **Backtracking interaction.** Eisenstat-Walker assumes Newton
   converges monotonically. We don't — backtracking selects the
   best of all NR iters post-hoc. A run of "loose-η" iters could
   produce candidates that don't include the eventual optimum,
   degrading picked-max residual.

   *Mitigation*: start with a tighter `α` (e.g., 1.5 instead of 2)
   so loosening is more gradual; measure picked-max as primary
   acceptance criterion.

3. **Friction warmup window.** Loose linear solves at iter 0-1
   might not produce an outer-residual reduction big enough for
   `η` to tighten by iter 2-3. If `backtrack_min_iter = 2` and
   `η` is still loose at iter 2, the picked iter could be one with
   sloppy friction.

   *Mitigation*: don't apply the forcing term until iter
   `backtrack_min_iter` (i.e., keep `η = linear_tol` for the
   warmup window). Only adapt afterwards.

## Implementation plan

Roughly 50-80 LOC. Phases:

### Phase 1: scaffolding

* Add config knobs to `AxionEngineConfig`:
  ```
  eisenstat_walker_enabled: bool = False  # opt-in
  eisenstat_walker_gamma: float = 0.9
  eisenstat_walker_alpha: float = 1.5     # conservative start
  eisenstat_walker_eta_min: float = 1e-6
  eisenstat_walker_eta_max: float = 0.5   # capped tighter than usual
  eisenstat_walker_warmup_iters: int = 2  # use linear_tol for first N iters
  ```
* Add `EngineData` buffers:
  * `prev_res_norm_sq: wp.array(num_worlds,)` — residual at iter k-1
  * `eta_k: wp.array(num_worlds,)` — current forcing term per world
* `_compute_forcing_term_kernel`: per-world, reads `res_norm_sq` and
  `prev_res_norm_sq`, applies the formula above with bounds.

### Phase 2: per-iter integration

Inside `nr_loop_step`, after `tiled_sq_norm.compute(res, res_norm_sq)`:

```python
if config.eisenstat_walker_enabled and current_iter >= warmup_iters:
    wp.launch(_compute_forcing_term_kernel, ...)
else:
    self.data.eta_k.fill_(self.config.linear_tol)

wp.copy(self.data.prev_res_norm_sq, self.data.res_norm_sq)
```

Then in the PCR call:

```python
self.cr_solver.solve(
    A=self.A_op,
    b=self.data.rhs,
    x=self.data.dconstr_force.full,
    preconditioner=self.preconditioner,
    iters=self.config.max_linear_iters,
    tol=self.data.eta_k,         # ← was self.config.linear_tol scalar
    atol=self.config.linear_atol,
    ...
)
```

This requires PCR's `solve()` to accept a per-world `tol` array
instead of a scalar. The current implementation (`pcr_solver.py:177`)
takes a scalar; the inner kernel `_check_residuals_kernel` at
`pcr_solver.py:85` already takes `tol_sq: float` and applies it
per-row, so we'd need to change it to read per-world from an array.
Small kernel change (~5 LOC).

### Phase 3: evaluation

A/B sweeps on:

* **obstacle_benchmark** (impact-heavy) — primary risk scene. Look
  at picked-max residual; if it degrades vs forcing-term-off, the
  FB-NR risk is real and we'd need the safer α/η_max bounds.
* **surface_drive_benchmark** (rolling) — primary win scene. Look
  at total PCR iters across the run (will be in the per-component
  profile output).
* **helhest_flipup** (high tangential velocity) — friction stress
  test. Same as obstacle: watch picked-max.

Compare:
* PCR iters per NR iter (mean, p95, max)
* Total PCR iters across run
* NR iter count distribution
* Picked-max residual
* Wall-clock per step (profile_e2e mode)

Acceptance: PCR iter savings of at least 20% on rolling AND
picked-max residual within 2× of baseline on impact scenes.

### Phase 4: tuning

If phase 3 looks good, sweep:

* `α ∈ {1.5, 1.7, 2.0}`
* `γ ∈ {0.7, 0.9}`
* `eta_max ∈ {0.1, 0.3, 0.5, 0.99}`
* `warmup_iters ∈ {1, 2, 3}`

Pick the most aggressive setting that keeps picked-max
non-regressed on all three eval scenes. That's the new default.

## Expected outcome

Published speedups on FB-style complementarity problems
(SICONOS, IPOPT) range from 1.5× to 3× on the linear solver. We
*should* see something in that range — our PCR cost is the same
order of magnitude as those benchmarks.

If we don't see at least 20% PCR-iter reduction at safe parameters,
something is wrong (probably the FB corner ate it; safer α might
help) or the technique just doesn't fit our problem (try option 3
from `pcr_warm_start_options.md` instead — preconditioner reuse).

## See also

* `axion/optim/pcr_solver.py` — current PCR with fixed-tol behavior
* `axion/core/base_engine.py:nr_loop_step` — where PCR is called
* [`pcr_warm_start_options.md`](pcr_warm_start_options.md) — broader
  context, lists this approach as Option 5 of 5
* [`convergence_criterion_options.md`](convergence_criterion_options.md)
  — sister doc on the *outer* NR convergence check (different lever)
* Eisenstat & Walker, "Choosing the Forcing Terms in an Inexact
  Newton Method" (1996) — original paper, all formulas above
