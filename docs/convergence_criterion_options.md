# Newton-Raphson Convergence: What Are We Actually Checking?

## The problem

Newton-Raphson stops when `||r||² < newton_atol²`, with `newton_atol = 1e-3`
in `examples/conf/engine/axion.yaml`. There are two things wrong with that:

**1. The threshold has no physical meaning.** It was picked by trial
and error — tighten until convergence looks "good", loosen until
convergence is "fast enough". When asked "is 1e-3 the right number?"
the only honest answer is "we don't know, but it works." That's not
how a tolerance should be specified for a physics solver where users
will reason about the output ("how accurate is my contact force?",
"how much does my robot drift per second?").

**2. The residual vector has heterogeneous units.** `r` is the
concatenation of:

| block         | row content                                  | physical units            |
|---------------|----------------------------------------------|---------------------------|
| dynamics      | `M(v − v_old)/dt − f_ext − J^T λ`            | force (N)                 |
| contact-FB    | `φ_FB(λ_n, J_n v + b_n)` per contact         | mixed (force × velocity)  |
| friction-FB   | `φ_FB^{2D}(λ_t, J_t v, μ·f_n_prev)` per c.   | mixed                     |
| joint (pos)   | `g_joint(q)` per joint                       | meters or radians         |
| control       | `J_ctrl·v − v_target` per actuator           | m/s or rad/s              |

`||r||²` sums squares of entries with different units. The norm is
dominated by whichever block has the largest *natural* scale,
regardless of which one the engineer cares about. A residual of 1e-3
could mean "10 mN dynamic imbalance", "1 mm joint drift", "1e-3
unitless complementarity violation", or any combination — they're
all in the same scalar bucket.

This means tightening `newton_atol` doesn't reliably improve the
quantity you actually care about. It improves *something* — maybe
the dominant-scale block — but maybe not the one that was bothering
you.

## Why this matters

Two practical consequences:

* **You can't reason about the solver's output.** "1mm penetration
  tolerance" or "10 mN force balance" is the kind of statement an
  engineer wants to make. With a unit-mixed scalar threshold, you
  can't.
* **Tuning is fragile.** Change `dt` or `friction_compliance` or
  enable a new constraint, and the relative scales of the blocks
  shift. Yesterday's `1e-3` becomes today's "too tight" or "too
  loose", with no principled way to know which.

## Options

Five approaches, roughly ordered by effort.

### Option 1: Use `||Δλ|| < ε_step` as the convergence check

Stop NR when the Newton step itself is small. Since `Δλ` has units of
force (N), the threshold is engineering-meaningful: "I will accept the
solution when one more Newton iteration would change my contact forces
by less than 0.01 N."

Implementation: trivial. We already compute `Δλ` (`data._dconstr_force`)
each iter as the linear-solve output. One reduction kernel
(`||Δλ||_∞` or `||Δλ||_2`) and a comparison.

Trade-offs:
* **Pro:** physical units, single threshold, the threshold means
  exactly what it says.
* **Con:** can declare convergence too early near the FB cone corner
  where Newton stagnates (small `Δλ` but residual still large).
  Mitigated by keeping `||r||` as a secondary check.
* **Con:** doesn't catch the case where Newton is *not* converging at
  all — large `||Δλ||` indefinitely. Need a max-iter cap (which we
  have).

**Production solvers** (e.g., MuJoCo CG/Newton paths) typically use
`||Δλ||` (or `||Δq||`) as a *primary* check alongside a residual
check.

### Option 2: Per-block residual thresholds

Split `r` at the offsets we already track in `EngineDimensions`
(`offset_n`, `offset_f`, joint slots, control slots) and check each
block independently:

```
||r_dyn||_∞      < ε_force      e.g. 1e-2 N
||r_contact||_∞  < ε_compl      e.g. 1e-4 (FB units)
||r_friction||_∞ < ε_compl      e.g. 1e-4
||r_joint||_∞    < ε_pos        e.g. 1e-3 m
||r_ctrl||_∞     < ε_vel        e.g. 1e-3 m/s
```

NR exits only when *all* hold. Each threshold is in its own physical
units and can be set by the engineer based on what they care about
("I'm OK with 1 mm joint drift, but not with > 10 mN force imbalance").

Implementation: medium. Slice the residual vector at offsets we
already track, run several reduction kernels, AND the convergence
flags. Modest kernel changes to `_check_residuals_kernel` and
config additions for the per-block thresholds.

Trade-offs:
* **Pro:** every threshold is physically meaningful and orthogonal
  from the others.
* **Con:** more knobs to tune (4–5 instead of 1). Mostly mitigated
  by sensible defaults.
* **Con:** complementarity rows (`r_contact`, `r_friction`) still
  have mixed units inside themselves — FB combines λ (N) and gap
  velocity (m/s). Option 4 fixes this; option 2 alone leaves it.

### Option 3: Diagonal scaling of the residual norm

Compute `||r||² = Σ_i r_i² / diag(A)_ii` instead of `Σ_i r_i²`. This
is the diagonal approximation of the *energy norm* `r^T A^{-1} r`,
which has physical meaning (it equals `||Δx||²` where Δx is the
Newton step in the diagonal-A approximation). The scalar threshold
then has the same units as `||Δλ||²` from option 1.

Implementation: low. We already compute `diag(A)` for the Jacobi
preconditioner. One scaled-reduction kernel.

Trade-offs:
* **Pro:** keeps the single-threshold ergonomics, just makes the
  scalar mean something.
* **Pro:** automatically rebalances when you change `dt`, compliance,
  etc. — `diag(A)` shifts with them.
* **Con:** still fundamentally a single number — engineers can't
  state per-block tolerances.

### Option 4: Convert each block to an engineering metric, then mix

Translate each row of `r` to a homogeneous physical unit before
combining:
* `r_dyn` → divide by mass to get velocity error (m/s)
* `r_contact` → convert FB to penetration-velocity (m/s) or
  λ-violation (N/typical_normal_force)
* `r_friction` → convert to tangential-force violation (N)
* `r_joint` → already in meters (pos-level)
* `r_ctrl` → already in m/s

Then take `||r_normalized||_∞` with a single threshold in normalized
units.

Implementation: high. Per-block conversion functions, careful about
numerics near the FB corner (the "gradient" of FB w.r.t. its inputs
goes to zero, so the conversion is ill-conditioned exactly where
convergence is hardest).

Trade-offs:
* **Pro:** single threshold AND physical meaning AND
  unit-homogeneous.
* **Con:** non-trivial design decisions (what's the "right"
  conversion for FB?), and the conversions themselves can be
  numerically delicate.

### Option 5: KKT-residual decomposition (textbook approach)

Split into the three classical KKT components and threshold each:
* **Stationarity**: `M(v − v_old)/dt − f_ext − J^T λ` (force units)
* **Primal feasibility**: `min(0, g(q))` — penetration depth (m)
* **Complementarity**: `λ_n · g_n` — should go to 0

This is what Bullet, MuJoCo, ODE, and most rigid-body literature
compute. Conceptually cleaner than option 2 because each component
has a well-defined optimization-theory meaning.

Implementation: medium-to-high. Restructures how the residual is
assembled — stationarity is mostly what we already compute as
`r_dyn`, but primal feasibility (gap depth) isn't currently in `r`
at all (it's hidden inside the FB function). Adding it explicitly
might change the linearization structure.

Trade-offs:
* **Pro:** matches the formal optimization view, makes the solver
  state interpretable in standard terms.
* **Con:** invasive — touches both the residual-assembly code and
  the linearization. Not a "weekend project".

## Recommendation

The original recommendation was to start with option 1 (`||Δλ|| < ε`).
That was wrong. We tried it; it didn't deliver. See the postscript
below for what we measured. If you're still after a *speed* lever for
the solver, the right place to look is the *linear* solver, not the
NR convergence check — see
[`pcr_warm_start_options.md`](pcr_warm_start_options.md), which lays
out the Eisenstat-Walker forcing-term technique among other options.

**Don't tune `newton_atol` further as a speed lever.** It's a
unit-mixed scalar threshold; tweaking it is exactly the local-optimum
trap this doc was meant to call out. If you need different
quality/speed trade-offs, options 2-5 above are the principled paths,
all of them larger commitments than they're probably worth right now.

## Postscript: option 1 was tried and reverted

Implemented in `676cea4` (May 2026), reverted in `85cc8c3`. The data
that motivated the revert:

  obstacle_benchmark, 200 steps, by step_atol value:

  step_atol  sumI   hit16  picked_max  bad
  off        1962    56    7.8e-4       0
  1e-3       1977    56    1.5e-4       0   ← chosen "safe" default
  1e-2       1895    49    8.1e-2       5   ← danger zone
  1e-1       1853    46    1.0e-3       1

Two findings:

1. **The "safe" default (1e-3 N) never fires** on these scenes.
   Newton step magnitudes don't actually drop below 1e-3 N until the
   residual has already converged. Adding the check imposed a small
   overhead (one reduction kernel) for zero behavior change. Net:
   tiny *regression* at default settings.

2. **There's a narrow danger zone around 1e-2 N.** The step check
   fires *just early enough* on impact steps to short-circuit
   FB-friction's warmup window — the same warmup window
   `backtrack_min_iter` exists to protect. NR exits before friction
   stabilizes, picked-max residual jumps to 8e-2 (effectively
   broken). Looser still (1e-1) saves iters but eats one bad step.

The check is *not bad as insurance* — the underlying motivation
(residual oscillates near the FB cone corner with tiny Δλ) is real,
just rare enough on the test scenes that we can't measure it. If
that regime ever shows up empirically (e.g., a scene where NR
provably stagnates per the candidates_res log), the check is a
valid response. Without that evidence, it's overengineering — paying
for an unobserved failure mode.

The lesson, for future would-be improvers of this convergence check:

* Don't ship a check whose default doesn't fire. If 1e-3 is too tight
  to ever trigger, the code path is dead — and dead code never
  delivers the safety it's supposed to provide either, because it
  hasn't been validated against real failures.
* `backtrack_min_iter` is a *load-bearing wall* for friction lag.
  Anything that lets NR exit before iter `min_iter` is at risk of
  the same broken-friction regime. Touch with care.
* Speed wins for this solver come from the linear-solve side
  (~85% of step time), not the NR-check side. Look at
  `pcr_warm_start_options.md` for that lever.

## See also

* `axion/core/residual_utils.py` — current residual assembly
* `axion/core/base_engine.py:228` — the `_check_residuals_kernel`
  call where this decision lives today
* `engine_dims.offset_n`, `offset_f`, etc. — already-tracked block
  boundaries (option 2 keys off these)
* [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)
  — the FB cone corner is the regime where option 1 specifically
  helps (residual oscillates, Δλ doesn't)
* [`pcr_warm_start_options.md`](pcr_warm_start_options.md) — sister
  doc on the linear-solver side; same flavor of "what's the principled
  thing to do" question
