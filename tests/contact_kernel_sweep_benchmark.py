import csv
import os
import statistics
import time
from itertools import product

import numpy as np
import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.types import ContactInteraction
from axion.types import SpatialInertia

wp.init()


def setup_data(
    num_worlds,
    num_bodies_per_world,
    num_contacts_per_world,
    device,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Generates random batched input data for the kernel benchmark."""

    N_w = num_worlds
    N_b = num_bodies_per_world
    N_c = num_contacts_per_world

    np.random.seed(42)  # for reproducibility

    # --- Generate Penetration Depths ---
    penetration_depths = np.random.rand(N_w, N_c).astype(np.float32) * 0.1
    num_inactive = int(N_c * inactive_contact_ratio)
    if num_inactive > 0:
        inactive_indices = np.random.choice(N_c, num_inactive, replace=False)
        penetration_depths[:, inactive_indices] = -0.01  # Make them non-penetrating

    # --- Generate Body Indices ---
    body_a_indices = np.random.randint(0, N_b, size=(N_w, N_c), dtype=np.int32)
    body_b_indices = np.random.randint(0, N_b, size=(N_w, N_c), dtype=np.int32)
    mask = body_a_indices == body_b_indices
    while np.any(mask):
        body_b_indices[mask] = np.random.randint(0, N_b, size=np.sum(mask), dtype=np.int32)
        mask = body_a_indices == body_b_indices

    num_fixed = int(N_c * fixed_body_ratio)
    if num_fixed > 0:
        fixed_indices = np.random.choice(N_c, num_fixed, replace=False)
        chooser = np.random.randint(0, 2, size=num_fixed)
        body_a_indices[:, fixed_indices[chooser == 0]] = -1
        body_b_indices[:, fixed_indices[chooser == 1]] = -1

    # --- Generate Random Jacobians, coeffs, etc. ---
    J_a = (np.random.rand(N_w, N_c, 3, 6) - 0.5).astype(np.float32)
    J_b = (np.random.rand(N_w, N_c, 3, 6) - 0.5).astype(np.float32)
    restitution_coeffs = (np.random.rand(N_w, N_c) * 0.5).astype(np.float32)
    friction_coeffs = (np.random.rand(N_w, N_c) * 0.5 + 0.5).astype(np.float32)

    # --- Create Interactions Array ---
    interactions_list = []
    for w in range(N_w):
        for i in range(N_c):
            inter = ContactInteraction()
            inter.is_active = penetration_depths[w, i] > 0.0
            inter.body_a_idx = body_a_indices[w, i]
            inter.body_b_idx = body_b_indices[w, i]
            inter.penetration_depth = penetration_depths[w, i]
            inter.restitution_coeff = restitution_coeffs[w, i]
            inter.friction_coeff = friction_coeffs[w, i]

            inter.basis_a.normal = wp.spatial_vector(*J_a[w, i, 0])
            inter.basis_a.tangent1 = wp.spatial_vector(*J_a[w, i, 1])
            inter.basis_a.tangent2 = wp.spatial_vector(*J_a[w, i, 2])
            inter.basis_b.normal = wp.spatial_vector(*J_b[w, i, 0])
            inter.basis_b.tangent1 = wp.spatial_vector(*J_b[w, i, 1])
            inter.basis_b.tangent2 = wp.spatial_vector(*J_b[w, i, 2])
            interactions_list.append(inter)

    # --- Create Inverse Mass Array ---
    spatial_inertia_list = []
    for w in range(N_w):
        for i in range(N_b):
            inertia = SpatialInertia()
            inertia.m = np.random.rand() + 0.5  # mass from 0.5 to 1.5
            inertia.inertia = np.diag(np.random.rand(3) * 0.1 + 0.05)  # simple diagonal inertia
            spatial_inertia_list.append(inertia)

    # --- Assemble final data dictionary for Warp ---
    data = {
        "body_u": wp.from_numpy(
            (np.random.rand(N_w, N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_u_prev": wp.from_numpy(
            (np.random.rand(N_w, N_b, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_lambda_n": wp.from_numpy(
            (np.random.rand(N_w, N_c) * 0.1).astype(np.float32), device=device, dtype=wp.float32
        ),
        "interactions": wp.array(
            interactions_list, dtype=ContactInteraction, device=device
        ).reshape((N_w, N_c)),
        "body_M_inv": wp.array(spatial_inertia_list, dtype=SpatialInertia, device=device).reshape(
            (N_w, N_b)
        ),
        "dt": 0.01,
        "stabilization_factor": 0.2,
        "fb_alpha": 0.25,
        "fb_beta": 0.25,
        "compliance": 1e-6,
        # Outputs
        "h_d": wp.zeros((N_w, N_b), dtype=wp.spatial_vector, device=device),
        "h_n": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
        "J_hat_n_values": wp.zeros((N_w, N_c, 2), dtype=wp.spatial_vector, device=device),
        "C_n_values": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
        "s_n": wp.zeros((N_w, N_c), dtype=wp.float32, device=device),
    }
    return data


def time_kernel_once(device, kernel, kernel_args, dim):
    # single launch with synchronization
    wp.launch(kernel=kernel, dim=dim, inputs=kernel_args, device=device)
    wp.synchronize()


def measure(
    num_worlds,
    num_bodies,
    num_contacts,
    kernel,
    setup_fn,
    device,
    repeats=5,
    iters_per_repeat=100,
    warmup_iters=20,
    inactive_contact_ratio=0.0,
    fixed_body_ratio=0.0,
):
    """Return dict with mean/median/std etc. (no CUDA graph)."""
    # prepare data once per configuration
    data = setup_fn(
        num_worlds,
        num_bodies,
        num_contacts,
        device,
        inactive_contact_ratio=inactive_contact_ratio,
        fixed_body_ratio=fixed_body_ratio,
    )
    kernel_args = [
        data["body_u"],
        data["body_u_prev"],
        data["body_lambda_n"],
        data["interactions"],
        data["body_M_inv"],
        data["dt"],
        data["stabilization_factor"],
        data["fb_alpha"],
        data["fb_beta"],
        data["compliance"],
        data["h_d"],
        data["h_n"],
        data["J_hat_n_values"],
        data["C_n_values"],
        data["s_n"],
    ]

    dim = (num_worlds, num_contacts)

    # warm-up
    for _ in range(warmup_iters):
        data["h_d"].zero_()
        time_kernel_once(device, kernel, kernel_args, dim)

    # timed repeats
    per_repeat_avg_ms = []
    per_repeat_median_ms = []
    for r in range(repeats):
        # zero output to mimic solver pipeline
        # measure bundles of iters to reduce timer noise
        start = time.perf_counter()
        for _ in range(iters_per_repeat):
            data["h_d"].zero_()
            wp.launch(kernel=kernel, dim=dim, inputs=kernel_args, device=device)
        wp.synchronize()
        delta = time.perf_counter() - start
        avg_ms = delta / iters_per_repeat * 1000.0
        per_repeat_avg_ms.append(avg_ms)

    # robust summary stats
    mean_ms = statistics.mean(per_repeat_avg_ms)
    median_ms = statistics.median(per_repeat_avg_ms)
    stdev_ms = statistics.stdev(per_repeat_avg_ms) if len(per_repeat_avg_ms) > 1 else 0.0

    # normalized metrics
    contacts_total = num_worlds * num_contacts
    ns_per_contact = (mean_ms * 1e6) / contacts_total if contacts_total else float("inf")
    ms_per_world = mean_ms / num_worlds if num_worlds else float("inf")
    contacts_per_sec = (
        1.0 / (mean_ms / 1000.0) * (num_contacts)
    )  # contacts per second per world? careful interpretation

    return {
        "N_WORLDS": num_worlds,
        "N_BODIES": num_bodies,
        "N_CONTACTS": num_contacts,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "stdev_ms": stdev_ms,
        "ns_per_contact": ns_per_contact,
        "ms_per_world": ms_per_world,
        "contacts_total": contacts_total,
        "contacts_per_sec": contacts_per_sec,
        "repeats": repeats,
        "iters_per_repeat": iters_per_repeat,
    }


def sweep_and_save(
    kernel,
    setup_fn,
    device,
    sweep_ranges,
    out_csv="bench_sweep_results.csv",
    repeats=5,
    iters_per_repeat=100,
):
    """sweep_ranges is dict: {'N_WORLDS':[...], 'N_BODIES':[...], 'N_CONTACTS':[...]}"""
    keys = ["N_WORLDS", "N_BODIES", "N_CONTACTS"]
    fieldnames = keys + [
        "mean_ms",
        "median_ms",
        "stdev_ms",
        "ns_per_contact",
        "ms_per_world",
        "contacts_total",
        "contacts_per_sec",
        "repeats",
        "iters_per_repeat",
    ]

    # create CSV output
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for nw, nb, nc in product(
            sweep_ranges["N_WORLDS"], sweep_ranges["N_BODIES"], sweep_ranges["N_CONTACTS"]
        ):
            print(f"Measuring: Worlds={nw}, Bodies={nb}, Contacts={nc} ...")
            res = measure(
                nw,
                nb,
                nc,
                kernel,
                setup_fn,
                device,
                repeats=repeats,
                iters_per_repeat=iters_per_repeat,
            )
            writer.writerow({k: res[k] for k in fieldnames})
            f.flush()  # safe write in case it runs long
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    device = wp.get_device()
    print("Device:", device.name, "CUDA:", device.is_cuda)
    # Example sweep ranges (tweak them to cover the regime you care about)
    sweep_ranges = {
        "N_WORLDS": [1, 2, 4, 8, 16, 32, 64],
        "N_BODIES": [8, 16, 32, 64, 128],
        "N_CONTACTS": [16, 64, 256, 512, 1024],
    }

    # smaller defaults for quick iter; raise repeats/iters for higher confidence
    sweep_and_save(
        kernel=contact_constraint_kernel,
        setup_fn=setup_data,
        device=device,
        sweep_ranges=sweep_ranges,
        out_csv="bench_sweep_results.csv",
        repeats=5,
        iters_per_repeat=100,
    )
