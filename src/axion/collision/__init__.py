"""Contact-pipeline preprocessing.

This subpackage sits between Newton's narrow-phase output and Axion's
constraint kernels. Today it provides per-pair contact reduction policies;
in the future it can host other contact-pipeline preprocessing (warm
starting, contact-graph analysis, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ContactReducer, NoOpReducer
from .config import ContactReductionConfig, ContactReductionPolicy

if TYPE_CHECKING:
    from axion.core.engine_dims import EngineDimensions
    import warp as wp


def build_reducer(
    cfg: ContactReductionConfig,
    dims: "EngineDimensions",
    device: "wp.Device",
) -> ContactReducer:
    """Construct a reducer instance from a config.

    Args:
        cfg: User-provided reduction settings. ``cfg.policy="none"``
            yields a no-op reducer.
        dims: Engine dimensions (for kernels that allocate per-world
            scratch buffers).
        device: Warp device that the reducer's kernels will launch on.
    """
    policy = cfg.policy
    if policy == "none":
        return NoOpReducer()
    raise NotImplementedError(
        f"Contact reduction policy {policy!r} is not implemented yet. "
        "Phase 0 ships only the no-op path; subsequent phases add "
        "'top_k', 'fps', 'cluster', and 'hull'."
    )


__all__ = [
    "ContactReducer",
    "NoOpReducer",
    "ContactReductionConfig",
    "ContactReductionPolicy",
    "build_reducer",
]
