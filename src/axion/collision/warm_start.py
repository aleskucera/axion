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

        # References needed by apply/snapshot (used in later phases).
        self._shape_body = axion_model.shape_body
        # body_pose at warm-start time is body_pose_prev (load_data has
        # populated it; engine.step copies body_pose itself only after
        # load_data returns).
        self._body_pose = data.body_pose_prev
        self._body_vel = data.body_vel_prev

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
        """Phase 0: no-op. Phase 1 will copy the post-backtrack state
        from ``contacts`` and ``data._constr_force`` into the
        ``_prev_*`` buffers."""
        if not self._enabled:
            return
        # Snapshot kernel lands in phase 1.
