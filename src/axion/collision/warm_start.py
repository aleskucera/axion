"""Cross-step contact warm-start.

Predicted-position matching:
  At each new simulation step, for every contact in the (post-reducer)
  active set, find the closest contact from the previous step's
  converged state where "closest" is measured against a position
  predicted from body velocities. Copy the matched contact's converged
  λ_n and λ_t (with friction-basis projection) into
  ``_constr_force_prev_iter`` so the friction kernel sees real values
  at NR iter 0 instead of starting from zero.

Phase 0 (this commit) ships only the scaffolding: persistent buffers,
lifecycle hooks, and disabled-by-default no-op apply/snapshot kernels.
The matching and snapshot logic lands in subsequent phases.

The component lives parallel to ``ContactReducer`` because:
  - both run between Newton's narrow-phase output and Axion's
    constraint kernels,
  - the warm-starter relies on the reducer's stable per-pair structure
    (matching is done within (b0, b1) groups, with K survivors per
    pair after reduction),
  - keeping them as siblings under ``axion/collision/`` makes the
    contact-pipeline pre-processing obvious from the package layout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from axion.core.contacts import AxionContacts
    from axion.core.engine_data import EngineData
    from axion.core.engine_dims import EngineDimensions
    from axion.core.model import AxionModel


@wp.func
def _resolve_body(
    world_idx: int,
    shape_idx: int,
    shape_body: wp.array(dtype=wp.int32, ndim=2),
) -> int:
    if shape_idx < 0:
        return -1
    return shape_body[world_idx, shape_idx]


@wp.func
def _world_midpoint(
    world_idx: int,
    c_idx: int,
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    body_a: int,
    body_b: int,
    body_pose: wp.array(dtype=wp.transform, ndim=2),
) -> wp.vec3:
    """Contact midpoint in world frame, after thickness offsets along
    the normal. Mirrors the geometry used by the contact constraint
    kernel."""
    n = contact_normal[world_idx, c_idx]
    p_a_local = contact_point0[world_idx, c_idx]
    p_b_local = contact_point1[world_idx, c_idx]
    t_a = contact_thickness0[world_idx, c_idx]
    t_b = contact_thickness1[world_idx, c_idx]

    pose_a = wp.transform_identity()
    if body_a >= 0:
        pose_a = body_pose[world_idx, body_a]
    pose_b = wp.transform_identity()
    if body_b >= 0:
        pose_b = body_pose[world_idx, body_b]

    p_a_world = wp.transform_point(pose_a, p_a_local) - t_a * n
    p_b_world = wp.transform_point(pose_b, p_b_local) + t_b * n
    return 0.5 * (p_a_world + p_b_world)


@wp.kernel
def _snapshot_kernel(
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    offset_n: wp.int32,
    offset_f: wp.int32,
    # Outputs
    prev_b0: wp.array(dtype=wp.int32, ndim=2),
    prev_b1: wp.array(dtype=wp.int32, ndim=2),
    prev_p_world: wp.array(dtype=wp.vec3, ndim=2),
    prev_normal: wp.array(dtype=wp.vec3, ndim=2),
    prev_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    prev_lambda_t: wp.array(dtype=wp.vec2, ndim=2),
):
    """One thread per (world, contact slot). Slots beyond
    contact_count[w] are untouched (stale; the next step's apply will
    iterate only [0, prev_count) so it doesn't matter what's there)."""
    world_idx, c_idx = wp.tid()
    n_count = contact_count[world_idx]
    if c_idx >= n_count:
        return

    s0 = contact_shape0[world_idx, c_idx]
    s1 = contact_shape1[world_idx, c_idx]
    if s0 == s1:
        return

    b0 = _resolve_body(world_idx, s0, shape_body)
    b1 = _resolve_body(world_idx, s1, shape_body)

    midpoint = _world_midpoint(
        world_idx, c_idx,
        contact_point0, contact_point1, contact_normal,
        contact_thickness0, contact_thickness1,
        b0, b1, body_pose,
    )

    prev_b0[world_idx, c_idx] = b0
    prev_b1[world_idx, c_idx] = b1
    prev_p_world[world_idx, c_idx] = midpoint
    prev_normal[world_idx, c_idx] = contact_normal[world_idx, c_idx]
    prev_lambda_n[world_idx, c_idx] = constr_force[world_idx, offset_n + c_idx]
    prev_lambda_t[world_idx, c_idx] = wp.vec2(
        constr_force[world_idx, offset_f + 2 * c_idx],
        constr_force[world_idx, offset_f + 2 * c_idx + 1],
    )


# Adaptive match-distance threshold parameters. Used by the phase-2 match
# kernel; collected here so phase 0 already exposes them on the config.
DEFAULT_ALPHA = 0.1     # fraction of v·dt allowed as prediction error
DEFAULT_MIN_THRESHOLD = 5.0e-3   # 5 mm absolute floor (contact-discovery noise)


class ContactWarmStarter:
    """Cross-step warm-start of contact normal/friction forces.

    Lifecycle, per simulation step:
      1. ``apply()`` runs in ``load_data``, after contact reduction has
         settled the active contact set. It populates
         ``data._constr_force_prev_iter`` so the friction model has
         non-degenerate ``f_n_prev`` at NR iter 0.
      2. Newton-Raphson runs.
      3. ``snapshot()`` runs at the end of ``_solve``, after
         ``perform_backtracking`` has gathered the picked iterate into
         ``data._constr_force``. It saves the converged contact set
         and forces into per-instance ``_prev_*`` buffers so the next
         step can match against them.

    Phase 0: both methods are no-ops. The buffers are allocated and
    the lifecycle hooks are wired so disabling/enabling the feature is
    just a config flip in later phases.
    """

    def __init__(
        self,
        enabled: bool,
        axion_model: "AxionModel",
        data: "EngineData",
        dims: "EngineDimensions",
        device: wp.Device,
        alpha: float = DEFAULT_ALPHA,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
    ):
        self._enabled = bool(enabled)
        self._device = device
        self._alpha = float(alpha)
        self._min_threshold = float(min_threshold)

        # References needed by apply/snapshot.
        self._shape_body = axion_model.shape_body
        # body_pose_prev is the right reference for BOTH lifecycle hooks:
        #   - At apply time (in step N+1's load_data, after body_pose_prev
        #     was just set to state_in.body_q): equals start-of-step N+1
        #     pose, which is what collide() ran at, what contact_point0/1
        #     are anchored to, and what we want for current midpoints.
        #   - At snapshot time (end of step N's _solve, after
        #     perform_backtracking): body_pose_prev still holds step N's
        #     start-of-step pose because nothing in _solve writes to it.
        #     This is where step N's contacts were anchored, so the
        #     stored midpoint is the actual contact location at step N.
        # Using the same reference keeps both code paths simple AND
        # makes the (apply, snapshot) midpoints comparable across steps
        # via the v·dt prediction we compute in apply.
        self._body_pose = data.body_pose_prev
        self._body_vel = data.body_vel_prev

        # Constraint-vector offsets so the kernels can index
        # ``data._constr_force`` correctly.
        self._offset_n = int(dims.offset_n)
        self._offset_f = int(dims.offset_f)

        N_w = dims.num_worlds
        N_c = dims.contact_count

        # Persistent state: previous step's converged contact set + forces.
        # Indexed by (world, contact_slot) — same shape as AxionContacts
        # arrays so we can reuse the indexing math. Slots beyond
        # ``_prev_count[w]`` are stale and must be ignored on read.
        with wp.ScopedDevice(device):
            self._prev_count = wp.zeros((N_w,), dtype=wp.int32)
            self._prev_b0 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._prev_b1 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._prev_p_world = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._prev_normal = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._prev_lambda_n = wp.zeros((N_w, N_c), dtype=wp.float32)
            self._prev_lambda_t = wp.zeros((N_w, N_c), dtype=wp.vec2)

            # Per-step diagnostics (reset at start of each apply()).
            self._diag_attempts = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_matched = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_max_dist = wp.zeros((N_w,), dtype=wp.float32)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply(self, contacts: "AxionContacts", data: "EngineData", dt: float) -> None:
        """Phase 0: no-op. Phase 2 will read ``_prev_*`` and write
        warm-started values into ``data._constr_force_prev_iter``."""
        if not self._enabled:
            return
        # Diagnostics reset for this step.
        self._diag_attempts.zero_()
        self._diag_matched.zero_()
        self._diag_max_dist.zero_()
        # Match-and-write kernel lands in phase 2.

    def snapshot(self, contacts: "AxionContacts", data: "EngineData") -> None:
        """Save the post-backtrack converged contact set + forces into
        the persistent ``_prev_*`` buffers so the next step's ``apply``
        can match against them.

        Reads ``data._constr_force`` (post-backtrack picked iter's
        forces), ``data.body_pose_prev`` (pre-step pose, the frame in
        which contact_point0/1 are anchored), and the current contact
        set from ``contacts``.
        """
        if not self._enabled:
            return

        # Mirror the active count so apply knows how many slots to scan.
        wp.copy(self._prev_count, contacts.contact_count)

        wp.launch(
            kernel=_snapshot_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
            inputs=[
                contacts.contact_count,
                contacts.contact_point0,
                contacts.contact_point1,
                contacts.contact_normal,
                contacts.contact_shape0,
                contacts.contact_shape1,
                contacts.contact_thickness0,
                contacts.contact_thickness1,
                self._shape_body,
                self._body_pose,
                data._constr_force,
                self._offset_n,
                self._offset_f,
            ],
            outputs=[
                self._prev_b0,
                self._prev_b1,
                self._prev_p_world,
                self._prev_normal,
                self._prev_lambda_n,
                self._prev_lambda_t,
            ],
            device=self._device,
        )
