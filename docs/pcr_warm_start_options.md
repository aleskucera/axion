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

## Postscript: options 3, 4, 5 measured and ruled out

After implementing options 1 and 2 (and reverting them — they're
dead-ends as documented above), we measured the upper bound for
options 3, 4, and 5 directly. None deliver enough to be worth
pursuing on this codebase at this scale.

### Option 3 (preconditioner reuse): 2% ceiling

Per-component profile of obstacle_benchmark (200 steps, 3200 NR
iters total):

```
phase                count     mean ms     share
linear_system        3200       0.049       9.5%
preconditioner       3200       0.010       2.0%   ← option 3 ceiling
cr_solve             3200       0.386      74.4%
step_or_linesearch   3200       0.047       9.1%
convergence_check    3200       0.026       5.0%
```

`preconditioner.update()` is **2% of NR-iter time**. Skipping every
non-first update saves at most ~1.5% of step time. Run-to-run
variance on wall-clock is 3–5%. The maximum possible win is below
noise, even before counting any PCR-iter increase from a stale
preconditioner.

The PCR-doc pitch claimed "5–15%" payoff; that was wrong. Always
profile before committing.

### Option 4 (Krylov recycling): A is too volatile to recycle

Recycling pays off when A's slow-eigenvector subspace persists
across solves. Measured how much A actually moves between PCR calls
in our setup, using two probes:

  1. `||Δdiag(A)|| / ||diag(A)||` — cheap, captures only diagonal.
  2. `||A·v − A_prev·v|| / ||A·v||` for fixed random `v` — captures
     full A change (diagonal + off-diagonal entries).

Within-step (NR iter k vs k-1, same simulation step):

```
  ||Δdiag(A)|| / ||diag(A)|| :  p50 = 7e-2,  p95 = 8e-1,  max = 8
  ||Δ(A·v)|| / ||A·v||       :  p50 = 0.85,  p95 = 4.5,   max = 3e4
```

A changes by **85% per NR iter on average within a single step**.
The diagonal changes "only" 7%, meaning off-diagonal entries shift
much more than diagonals — plausible because each NR iter's
`apply_newton_step` integrates body pose (a wheel at 1.5 m/s with
a Newton-step-driven velocity update can move several cm per iter),
which alters the geometric alignment between J rows of constraints
sharing a body. The diagonal `||J_i row||² / m_eff + C_ii` depends
on a single row's magnitude (less sensitive); the off-diagonal
`J_i · M⁻¹ · J_j^T` depends on inner products of geometrically
shifting rows (more sensitive).

For comparison: published Krylov-recycling work (Parks & de Sturler;
PETSc) targets relative A-change in the 10–20% range. **Our 85%
makes recycling within a step infeasible.**

Across-step (last iter of step n-1 vs first iter of step n):

```
  ||Δdiag|| / ||diag|| :  p50 = 3e-7  (essentially identical)
  ||Δ(A·v)|| / ||A·v|| :  p50 = 2.7e5  (broken — see caveat)
```

The `A·v` measurement here is contaminated by changing constraint
sets between steps: when contacts come/go, the same slot in the
residual vector now indexes a different physical constraint, so a
fixed probe `v` measures slot-relabeling rather than actual matrix
shift. The diagonal measurement (which masks inactive slots) shows
the *active* part of A is essentially unchanged across steps —
because step n's first iter starts with body_pose copied from
step n-1's picked iter (warm-start trajectory continuity).

To get a clean cross-step A-change number, the probe would need to
track active-constraint identity across steps. Doable but moot:
within-step recycling is by far the dominant regime (3200 calls vs
200 step boundaries), and within-step recycling is dead at 85%
relative change.

**Verdict on option 4: not viable for this solver.** Within-step A
changes too much for any subspace recycling scheme. Reproducer:
`test_scripts/measure_A_stability.py`.

### Option 5 (Eisenstat-Walker): tried and reverted

See [`eisenstat_walker.md`](eisenstat_walker.md) for the full
postscript. Summary: implemented, measured, reverted. The "smoothness
assumptions of E-W's classical proofs" don't hold for FB-NR; loose
early-iter linear solves amplify FB-corner degeneracy, breaking
convergence on impact scenes. Even with the tightest η_max bounds
that didn't break convergence, PCR-iter savings were negligible.

### What's left

Of the five options in this doc, four are now ruled out by
measurement (1 and 2 by analysis, 3 and 4 by direct profiling, 5
by implementation+measurement). The remaining linear-solver-side
ideas are at the *kernel* level rather than the *algorithm* level:

  * Replace CR with CG (one-line swap; CG is theoretically optimal
    for SPD A while CR is better for indefinite — A is mostly SPD
    here, so CG might give 10–20% iter reduction at zero risk).
  * Block-Jacobi preconditioner (smarter preconditioner; fewer PCR
    iters per call). 1–2 weeks; published wins for similar problems
    are 10–30%.
  * Sub-kernel optimization (kernel fusion in the CR inner loop;
    reduce kernel-launch overhead; lower precision in inner products).
    Tedious but tractable, 5–15% plausible.

Or, for a structural rethink that **bypasses the FB corner entirely**
(which has been the dominant source of dead-ends across this entire
investigation), see APGD discussion in solver-side notes — that's a
4–6 week reformulation, not a linear-solver swap.

## See also

* `axion/optim/pcr_solver.py` — current PCR implementation
* `test_scripts/measure_A_stability.py` — the A-volatility probe
  used for option 4's verdict
* [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)
  — why warm-starting the NR iterate (which Option 2 reduces to)
  doesn't work with FB complementarity
* [`eisenstat_walker.md`](eisenstat_walker.md) — option 5
  postscript with implementation data
* `engine_profiler` per-component output — shows PCR's share of step
  time (~74% in current configurations) so any of these options
  has a meaningful step-time ceiling
