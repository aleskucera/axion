"""Per-body-pair block-Jacobi preconditioner for the Schur-complement
matrix A = J·M⁻¹·Jᵀ + C.

Design rationale: see `docs/pcr_warm_start_options.md` (section
"Five block schemes considered"). The 3×3 per-contact block-Jacobi
fails because J_n ⊥ {J_t1, J_t2} by construction makes the within-
contact 3×3 block of A approximately diagonal. The off-diagonal
coupling that diagonal Jacobi misses lives between *different
contacts that share a body* — for a wheel with 4 ground contacts,
M_wheel⁻¹ couples those 4 contacts.

Per-body-pair grouping handles this cleanly:
  * each constraint has exactly one pair (b₁, b₂) — no overlap, no
    ownership-rule arbitrariness;
  * the per-pair block A_pair captures both bodies' contributions to
    A_ij within the pair;
  * the only coupling missed is cross-pair through a shared body
    (e.g., (chassis, wheel-1) <-> (chassis, wheel-2) joint
    constraints share chassis), which we measured to be small for
    Helhest's heavy chassis.

If per-body-pair turns out marginal, the natural escalation is
Schur-on-constraint-types (invert the joint block A_jj exactly,
Jacobi on the contact-block Schur complement) — see PCR doc.

This file is the new home for everything specific to this
preconditioner so it can be reverted cleanly if it doesn't pan out.
The existing JacobiPreconditioner in `preconditioner.py` is
untouched.

PHASE 1 (this file): pair-assignment only. Computes which pair each
active constraint belongs to and builds (pair_id → constraint_list)
member lists. No matvec yet; that's phase 4.
"""
import warp as wp


# Maximum number of constraints we expect in any one pair group. With
# cluster contact reduction (max_per_pair=8) plus a few friction +
# joint slots, the largest realistic pair has ~24 constraints. Set
# the bound generously to absorb future growth; overflow is silently
# clipped in phase 1's atomic-insertion path and detected at phase 2.
MAX_MEMBERS_PER_PAIR = 64


@wp.kernel
def _compute_pair_ids_kernel(
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),     # (W, N_c, 2)
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),  # (W, N_c)
    n_bodies: int,
    # Output: per-constraint pair id, or -1 if inactive.
    constr_pair_id: wp.array(dtype=wp.int32, ndim=2),  # (W, N_c)
):
    """Compute a canonical pair id for each constraint.

    Encoding: pair_id = b_lo · (n_bodies + 1) + b_hi
    where (b_lo, b_hi) = sorted(b₁, b₂) with -1 mapped to n_bodies
    (the "ground sentinel"). Symmetric: pair_id is the same regardless
    of which body sits in slot 0 vs slot 1 of constr_body_idx.

    Inactive constraints get pair_id = -1.

    Range of pair_id for active constraints:
        0 ≤ pair_id < (n_bodies + 1)²
    For Helhest with n_bodies=16, this is 0..288.
    """
    w, c = wp.tid()
    if c >= constr_body_idx.shape[1]:
        return

    if constr_active_mask[w, c] == 0.0:
        constr_pair_id[w, c] = -1
        return

    b1 = constr_body_idx[w, c, 0]
    b2 = constr_body_idx[w, c, 1]

    # Map -1 (no body / static reference) to n_bodies (ground sentinel).
    e1 = b1
    if b1 < 0:
        e1 = n_bodies
    e2 = b2
    if b2 < 0:
        e2 = n_bodies

    # Canonicalize: sort so smaller body id is first.
    b_lo = e1
    b_hi = e2
    if e2 < e1:
        b_lo = e2
        b_hi = e1

    constr_pair_id[w, c] = b_lo * (n_bodies + 1) + b_hi


@wp.kernel
def _build_pair_member_lists_kernel(
    constr_pair_id: wp.array(dtype=wp.int32, ndim=2),  # (W, N_c)
    # Outputs:
    pair_member_count: wp.array(dtype=wp.int32, ndim=2),     # (W, n_pairs_max)
    pair_member_list: wp.array(dtype=wp.int32, ndim=3),       # (W, n_pairs_max, MAX_MEMBERS)
    pair_overflow_flag: wp.array(dtype=wp.int32, ndim=1),     # (W,)
):
    """Populate pair → constraint member lists by scanning every
    constraint and atomically inserting it into its pair's list.

    Overflow protection: if a pair has more constraints than
    MAX_MEMBERS_PER_PAIR, we silently clip the insertion and flag
    the world via pair_overflow_flag[w] = 1. Phase 2 callers should
    sync-and-check this flag before assuming the lists are complete.
    """
    w, c = wp.tid()
    if c >= constr_pair_id.shape[1]:
        return

    pid = constr_pair_id[w, c]
    if pid < 0:
        return

    # Atomically claim a slot in this pair's member list. The slot
    # value returned is the *previous* count, i.e. our slot index.
    slot = wp.atomic_add(pair_member_count, w, pid, 1)
    if slot < pair_member_list.shape[2]:
        pair_member_list[w, pid, slot] = c
    else:
        # Overflow — slot index is past our buffer. Flag the world.
        # (Multiple overflowing constraints will all set the flag to
        # 1; that's fine, the flag is just a "did anything overflow"
        # indicator.)
        pair_overflow_flag[w] = 1


class PerBodyPairPreconditioner:
    """Block-Jacobi preconditioner with per-body-pair blocks.

    Phase 1: pair-assignment only. Subsequent phases will add block
    extraction (phase 2), per-pair dense factorization (phase 3),
    and matvec (phase 4) — at which point the class will subclass
    `warp.optim.linear.LinearOperator` and become a drop-in
    replacement for `JacobiPreconditioner`.

    Lifecycle:
        precond = PerBodyPairPreconditioner(engine)
        # Per simulation step (or when active constraint set changes):
        precond.update_pair_assignments()
        # Phases 2+ will add an .update() (block factorization) and
        # a .matvec() (apply preconditioner).
    """

    def __init__(self, engine):
        self.engine = engine
        self.device = engine.device

        N_w = engine.dims.num_worlds
        N_c = engine.dims.num_constraints
        N_b = engine.dims.body_count

        # Pair encoding: ground sentinel = N_b, so max pair_id is
        # (N_b+1)² - 1. Storage cost grows quadratically with body
        # count; for N_b=16 → 289 pair slots, comfortably small.
        self.n_pairs_max = (N_b + 1) * (N_b + 1)
        self.n_bodies = N_b
        self.MAX_MEMBERS_PER_PAIR = MAX_MEMBERS_PER_PAIR

        with wp.ScopedDevice(self.device):
            # Per-constraint pair id (or -1 for inactive).
            self.constr_pair_id = wp.zeros((N_w, N_c), dtype=wp.int32)
            # Per-pair member count and member list.
            self.pair_member_count = wp.zeros(
                (N_w, self.n_pairs_max), dtype=wp.int32
            )
            self.pair_member_list = wp.zeros(
                (N_w, self.n_pairs_max, self.MAX_MEMBERS_PER_PAIR),
                dtype=wp.int32,
            )
            self.pair_overflow_flag = wp.zeros((N_w,), dtype=wp.int32)

    def update_pair_assignments(self):
        """Recompute pair IDs and member lists.

        Should be called whenever the active constraint set changes —
        in practice once per simulation step (after collision detection
        + reduction). The per-NR-iter J_values do change inside a step,
        but the pair assignment itself only depends on which constraint
        indices are active and which bodies they touch, both of which
        are static within a step.
        """
        # Reset: zero member counts, clear overflow flag. The per-
        # constraint pair_id is fully overwritten by the kernel so no
        # need to clear it. The pair_member_list values are stale for
        # empty slots but won't be read (pair_member_count gates how
        # far we scan).
        self.pair_member_count.zero_()
        self.pair_overflow_flag.zero_()

        N_w = self.engine.dims.num_worlds
        N_c = self.engine.dims.num_constraints

        wp.launch(
            kernel=_compute_pair_ids_kernel,
            dim=(N_w, N_c),
            inputs=[
                self.engine.data._constr_body_idx,
                self.engine.data._constr_active_mask,
                self.n_bodies,
            ],
            outputs=[self.constr_pair_id],
            device=self.device,
        )

        wp.launch(
            kernel=_build_pair_member_lists_kernel,
            dim=(N_w, N_c),
            inputs=[self.constr_pair_id],
            outputs=[
                self.pair_member_count,
                self.pair_member_list,
                self.pair_overflow_flag,
            ],
            device=self.device,
        )

    # ---- Phase 1 introspection helpers (CPU-side, sync required) ----

    def num_active_pairs(self, world_idx: int = 0) -> int:
        """Number of distinct pairs with at least one active constraint
        in `world_idx`. Forces a sync — debug/diagnostic use only."""
        counts = self.pair_member_count.numpy()[world_idx]
        return int((counts > 0).sum())

    def active_pair_member_counts(self, world_idx: int = 0):
        """List of (pair_id, member_count) for each active pair."""
        counts = self.pair_member_count.numpy()[world_idx]
        nz = counts.nonzero()[0]
        return [(int(p), int(counts[p])) for p in nz]

    def overflowed(self, world_idx: int = 0) -> bool:
        """True iff any pair in `world_idx` exceeded MAX_MEMBERS_PER_PAIR."""
        return bool(self.pair_overflow_flag.numpy()[world_idx])

    def decode_pair_id(self, pair_id: int):
        """Decode a pair_id back into (b_lo, b_hi). The ground sentinel
        is `n_bodies`; -1 (no body) was mapped to that during encoding.
        Returns (-1, b_hi) if the pair includes the static reference."""
        b_lo = pair_id // (self.n_bodies + 1)
        b_hi = pair_id % (self.n_bodies + 1)
        if b_lo == self.n_bodies:
            b_lo = -1
        if b_hi == self.n_bodies:
            b_hi = -1
        return (b_lo, b_hi)
