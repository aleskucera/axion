# Can We Warm-Start PCR?

The Preconditioned Conjugate Residual solver in `axion/optim/pcr_solver.py`
runs once per Newton-Raphson iteration. NR currently calls it 5–16 times
per simulation step, and PCR inside each call runs up to
`max_linear_iters = 16`. That makes PCR the dominant cost (~85% of step
time per `engine_profiler` runs). Cutting its iters or reusing
information across calls is the most promising remaining lever.

This note enumerates what "warm-starting PCR" could actually mean,
which options are dead ends, and which are worth implementing. Order
roughly matches "obviousness on first read" — the dead ends are first.

## Option 1: Seed `Δλ_{k+1}` with `Δλ_k` (within an NR step)

The natural-sounding option. PCR solves `A_k · Δλ_k = b_k` at NR
iter k, then `A_{k+1} · Δλ_{k+1} = b_{k+1}` at iter k+1. If A and b
change slowly, why not start PCR's iterate at the previous solution?

**Implementation cost**: zero. Today
`base_engine.py:428` does `self.data._dconstr_force.zero_()` right
before each PCR call. Just removing that line is the change.

**Why it doesn't help**: as NR converges, `Δλ → 0` (each Newton step
gets smaller). At a late NR iter where the true `Δλ_{k+1}` is tiny,
seeding with the LARGER `Δλ_k` puts PCR's initial residual
`||b - A·Δλ_k||` *above* `||b||`, since `A·Δλ_k` is order
`||Δλ_k|| · ||A||` while `b = -r(x)` is order `||residual||` which is
small near convergence. PCR then has to spend iterations *undoing* the
warm-start before it can converge. Seed-from-zero is closer to the
truth than seed-from-last-Δλ once Newton is in the basin.

The one regime where this could help is iter 0 — but iter 0 doesn't
have a "previous Δλ". So zero benefit here.

**Verdict**: dead end. Worth a 5-minute test (delete the `.zero_()`,
measure on `obstacle_benchmark`) just to confirm the regression — but
do not invest beyond that.

## Option 2: Cross-step warm-start (seed first PCR call of step N+1 with last step's converged λ)

Same idea, scaled up: at the start of step N+1, NR iter 0 starts at
λ = 0; PCR will solve for Δλ such that `0 + Δλ ≈ λ_converged^{N+1}`.
If the next step's solution is close to the previous step's, why not
warm-start `Δλ` with `λ_converged^N`?

**Implementation cost**: a few lines — copy `_constr_force` into
`_dconstr_force` at the top of NR iter 0, then DON'T zero
`_constr_force` (which would then need to track the cumulative
update — also non-trivial).

**Why it doesn't help — and would actively hurt**: this is
mathematically equivalent to starting NR iter 0 at λ = λ_converged^N.
We tried this directly in the cross-step warm-start branch and
documented the failure in
[`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md):
the FB Jacobian is degenerate at the touching corner `(λ > 0, g = 0)`
where `∂φ_FB/∂λ = 1 - λ/√(λ² + g²) → 0`, and Newton diverges from
any non-zero starting iterate near that corner. Laundering the
warm-start through PCR's initial guess instead of through `_constr_force`
doesn't change the math — Newton still has to evaluate the FB
Jacobian at the warm-started state, and that Jacobian is still
degenerate.

**Verdict**: dead end, same root cause as the iterate-seeding failure.

## Option 3: Reuse the preconditioner across NR iterations

Today `preconditioner.update()` runs every NR iter. The Jacobi
preconditioner is `1/diag(A)`, which costs a full matvec-of-diagonal
plus an element-wise inverse. If A's diagonal barely changes between
NR iters (which it does once Newton is in the convergence basin),
recomputing it each iter is wasted GPU time.

**Implementation cost**: low. Two reasonable strategies:
  * Refresh every K iters (e.g., K = 4): simple, predictable, no
    measurement needed at runtime.
  * Refresh when residual stops dropping fast enough: needs a
    cheap residual-history check, but adapts to scenes.

**Why it might help**: the preconditioner update is a kernel launch
plus one diagonal extraction — small, but adds up over 5–16 NR iters
times 200 steps. Concrete impact needs measurement (the diagonal
extraction is probably memory-bandwidth-bound and may be cheap), but
it's the kind of change that's easy to A/B test cleanly.

**Risk**: a stale Jacobi preconditioner is just a worse preconditioner,
which means PCR iters might GO UP — net loss if those extra PCR iters
cost more than the saved updates. Need to measure on a representative
scene before committing.

**Verdict**: worth a 1–2 day prototype + benchmark. The mechanism is
cheap, the risk is bounded, and the result tells you something about
how much A's diagonal really moves.

## Option 4: Krylov subspace recycling

The textbook answer when the matrix changes slowly across solves.
After each PCR solve, save the basis vectors of the Krylov subspace
`{r₀, A·r₀, A²·r₀, …}` (or rather, an orthogonalized version of
them). Use approximate eigenvectors of A from those basis vectors to
"deflate" the next PCR solve — start the new Krylov iteration in a
subspace orthogonal to the small-eigenvalue directions that slow
ordinary CG/CR.

**Implementation cost**: high. Needs:
  * Modified Gram-Schmidt or reorthogonalization to keep the basis
    well-conditioned;
  * Eigenvector estimation (Rayleigh-Ritz on the saved basis);
  * Storage for ~10–20 vectors per system (small relative to the
    contact-constraint state, but adds GPU memory pressure);
  * Careful integration with CUDA graph capture.

References: Parks, de Sturler et al. ("Recycling Krylov Subspaces for
Sequences of Linear Systems") for the canonical algorithm.

**Why it might help (a lot)**: this is exactly the regime where
recycling pays off — PCR runs many times on slowly-varying
matrices. Published speedups range from 1.5× to 5× depending on how
much the matrix moves between solves. For our case (A changes
within an NR step but slowly; A changes more across steps but is
still structurally similar), the benefit is plausibly real.

**Risk**: implementation complexity. Probably 2–3 weeks of work plus
debugging cycles. Worth it only if Options 3 and 5 don't move the
needle.

**Verdict**: real technique, real wins, big project. Defer until
cheaper options are exhausted.

## Option 5: Adaptive `max_linear_iters` / Eisenstat-Walker forcing

PCR already has tolerance-based early exit
(`pcr_solver.py:_check_residuals_kernel` + the `wp.capture_while`
loop at line 295). At late NR iters where `||b||` is small, PCR
naturally exits in fewer iterations than at iter 0. This is the
existing inexpensive-Newton behavior.

The **forcing-term** technique (Eisenstat-Walker, 1996) takes this
further: tighten the linear tolerance only as the NR residual drops,
so early NR iters tolerate sloppy linear solves and only the final
NR iters demand tight ones. Today we use a fixed `linear_atol`; an
adaptive one keyed off the outer NR residual would let the early
PCR calls exit much sooner.

**Implementation cost**: low. One-line change in the PCR call inside
`nr_loop_step`: pass `tol = max(linear_atol_min, eta * outer_res_norm)`
where `eta` is a small constant (~0.01).

**Why it might help**: profiles show many PCR calls hitting
`max_linear_iters`. Many of those are wasted precision — at NR iter
0–2, the linear system is built from a far-from-converged outer
state, so solving it tightly just locks in noise. Loosening the
early calls saves PCR iters per NR iter.

**Risk**: too-loose early PCR can break NR's Q-quadratic convergence
near the solution. The classical Eisenstat-Walker analysis bounds
this (forcing term must shrink to maintain superlinear convergence),
so the standard schedule is well-tested.

**Verdict**: cheap to prototype, well-understood, likely real wins.
Probably the second-best place to look after Option 3.

## Recommendation

Order to chase if optimizing PCR:

1. **Option 5** (adaptive linear tol): cheapest, most well-trodden,
   likely visible win. Do this first.
2. **Option 3** (preconditioner reuse): cheap prototype, bounded
   risk, gives you data on how stable A's diagonal is.
3. **Option 4** (Krylov recycling): only if 5 and 3 didn't get you
   to your performance target.
4. **Options 1 and 2**: don't pursue. They run into either the
   "Δλ → 0 at convergence" wall (Option 1) or the FB-NR
   iterate-seeding wall (Option 2, see
   [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)).

## See also

* `axion/optim/pcr_solver.py` — current PCR implementation
* [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)
  — why warm-starting the NR iterate (which Option 2 reduces to)
  doesn't work with FB complementarity
* `engine_profiler` per-component output — shows PCR's share of step
  time (~85% in current configurations) so any of these options
  has a meaningful step-time ceiling
