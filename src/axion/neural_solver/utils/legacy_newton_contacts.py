"""Legacy Newton contact convention helpers for neural inference.

Newton PR #2069 standardized ``rigid_contact_normal`` as shape0 -> shape1 (A-to-B).
Axion's ``batch_contact_data_kernel`` (bd620db) negates once at ingest so NR
constraint code keeps the older B-to-A convention. Neural checkpoints trained
before that change expect unflipped normals through reorder / depth / mask;
callers pass ``apply_legacy_convention=True`` to undo the flip on
``AxionContacts`` buffers before NN feature extraction.
"""

from __future__ import annotations

import newton
import warp as wp


@wp.kernel
def _restore_legacy_contact_normals_kernel(
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
):
    """Negate batched contact normals in place (undo Axion Newton-1.2 flip)."""
    world_idx, contact_idx = wp.tid()
    n = contact_normal[world_idx, contact_idx]
    contact_normal[world_idx, contact_idx] = wp.vec3(-n[0], -n[1], -n[2])


def restore_legacy_newton_contact_convention(axion_contacts) -> None:
    """Restore pre-bd620db contact normals on AxionContacts for legacy NN inference."""
    wp.launch(
        kernel=_restore_legacy_contact_normals_kernel,
        dim=axion_contacts.contact_normal.shape,
        inputs=[axion_contacts.contact_normal],
        device=axion_contacts.device,
    )


def restore_legacy_newton_contacts_if(apply_legacy: bool, axion_contacts) -> None:
    """Apply legacy contact convention when ``apply_legacy`` is True."""
    if apply_legacy:
        restore_legacy_newton_contact_convention(axion_contacts)


def create_axion_contacts_for_nn(
    nn_predictor,
    newton_contacts: newton.Contacts,
    *,
    apply_legacy_convention: bool,
):
    """Load Newton contacts and optionally restore pre-Newton-1.2 normals for NN input."""
    axion_contacts = nn_predictor.create_axion_contacts(newton_contacts)
    restore_legacy_newton_contacts_if(apply_legacy_convention, axion_contacts)
    return axion_contacts
