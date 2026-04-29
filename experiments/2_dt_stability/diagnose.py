"""Phase 1 diagnostic for the dt-dependence problem.

Reproduces the symptom in the simplest possible setting: a 1 kg sphere
resting on a frictionless ground plane. Runs the AxionEngine with HDF5
logging on, then plots per-step `iter_count` and `res_norm_sq`. Compare
runs at large dt (e.g. 0.05) vs small dt (e.g. 0.001) — at small dt we
expect `iter_count` to saturate at `max_newton_iters` and `res_norm_sq`
to fail to drop below `newton_atol²`.

Usage:
    python experiments/2_dt_stability/diagnose.py --dt 0.05  --out results/diag_dt0.05.h5
    python experiments/2_dt_stability/diagnose.py --dt 0.001 --out results/diag_dt0.001.h5
    python experiments/2_dt_stability/diagnose.py --plot results/diag_dt0.05.h5 results/diag_dt0.001.h5
"""
import argparse
import pathlib

import warp

warp.config.quiet = True
import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.logging.hdf5_reader import HDF5Reader

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Defaults match AxionEngineConfig defaults — the no-flag run exercises
# the "shipped" behavior. Override --compliance to A/B against alternatives.
SIM_DURATION = 0.3
MAX_NEWTON_ITERS = 16
NEWTON_ATOL = 1e-3
CONTACT_COMPLIANCE = 1e-4   # interpreted as dt-adaptive: effective e = compliance/dt^2
FRICTION_COMPLIANCE = 1e-6


def build_model() -> newton.Model:
    """1 kg sphere of radius 0.1 m, resting just above a frictionless ground."""
    builder = AxionModelBuilder()

    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    )

    radius = 0.1
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()),
        label="ball",
    )
    # density chosen so that (4/3) pi r^3 * rho = 1 kg
    target_mass = 1.0
    volume = (4.0 / 3.0) * 3.141592653589793 * radius**3
    density = target_mass / volume

    builder.add_shape_sphere(
        body=body,
        radius=radius,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=density, mu=0.0, restitution=0.0
        ),
    )

    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def run(dt: float, out_path: pathlib.Path, contact_compliance: float = CONTACT_COMPLIANCE) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    num_steps = int(round(SIM_DURATION / dt))

    config = AxionEngineConfig(
        max_newton_iters=MAX_NEWTON_ITERS,
        newton_atol=NEWTON_ATOL,
        contact_compliance=contact_compliance,
        friction_compliance=FRICTION_COMPLIANCE,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig(
        enable_hdf5_logging=True,
        hdf5_log_file=str(out_path),
        max_simulation_steps=num_steps,
    )

    engine = AxionEngine(
        model=model, sim_steps=num_steps, config=config, logging_config=logging_config
    )

    state_in = model.state()
    state_out = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    print(f"Running diagnostic at dt={dt}, {num_steps} steps "
          f"(max_newton_iters={MAX_NEWTON_ITERS}, newton_atol={NEWTON_ATOL}, "
          f"contact_compliance={contact_compliance})")
    for step in range(num_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    engine.save_logs()
    print(f"Wrote {out_path}")


def plot(paths: list[pathlib.Path]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    for path in paths:
        with HDF5Reader(str(path)) as r:
            iter_count = r.get_dataset("iter_count")  # (N_steps, num_worlds)
            res_norm_sq = r.get_dataset("res_norm_sq")  # (N_steps, num_worlds)
            dt = r.get_attribute("/", "dt") if "dt" in r.list_attributes("/") else None

        # Use world 0 only.
        ic = np.asarray(iter_count)[:, 0] if iter_count.ndim > 1 else np.asarray(iter_count)
        rn = np.asarray(res_norm_sq)[:, 0] if res_norm_sq.ndim > 1 else np.asarray(res_norm_sq)

        label = f"{path.name}" + (f"  (dt={dt:g})" if dt else "")
        axes[0].plot(ic, label=label)
        axes[1].semilogy(np.maximum(rn, 1e-20), label=label)

    axes[0].set_ylabel("iter_count")
    axes[0].axhline(MAX_NEWTON_ITERS, color="k", linestyle="--",
                    alpha=0.4, label=f"max_newton_iters={MAX_NEWTON_ITERS}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("res_norm_sq (log)")
    axes[1].set_xlabel("step")
    axes[1].axhline(NEWTON_ATOL**2, color="k", linestyle="--",
                    alpha=0.4, label=f"newton_atol²={NEWTON_ATOL**2:g}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out = RESULTS_DIR / "diagnose_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dt", type=float, default=None, help="Run a diagnostic at this dt")
    parser.add_argument("--compliance", type=float, default=CONTACT_COMPLIANCE,
                        help=f"contact_compliance (default {CONTACT_COMPLIANCE})")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="HDF5 output path")
    parser.add_argument("--plot", nargs="+", type=pathlib.Path, default=None,
                        help="Plot one or more HDF5 logs")
    args = parser.parse_args()

    if args.plot:
        plot(args.plot)
    elif args.dt is not None:
        if args.out is not None:
            out = args.out
        elif args.compliance != CONTACT_COMPLIANCE:
            out = RESULTS_DIR / f"diag_dt{args.dt:g}_e{args.compliance:g}.h5"
        else:
            out = RESULTS_DIR / f"diag_dt{args.dt:g}.h5"
        run(args.dt, out, contact_compliance=args.compliance)
    else:
        parser.error("specify either --dt or --plot")


if __name__ == "__main__":
    main()
