"""Friction timestep-independence test — Helhest differential turn.

Drives the Helhest robot with a differential turn (left=4, right=2, rear=3 rad/s)
for 3s at multiple timesteps. If the impulse-level friction formulation is correct,
all trajectories should overlap regardless of dt.

Usage:
    python -m examples.comparison_friction.friction_dt_independence
    python -m examples.comparison_friction.friction_dt_independence --save results/friction_dt.json
    python -m examples.comparison_friction.friction_dt_independence --save results/friction_dt.json --plot
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import warp as wp

wp.init()

import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from helhest.common import create_helhest_model


def run_drive(dt, drive_time=3.0, settle_time=0.5,
              wheel_speeds=(4.0, 2.0, 3.0), k_p=250.0, friction=0.8):
    """Drive Helhest with given wheel speeds and return the xy trajectory."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane()
    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()),
        is_visible=False,
        control_mode="velocity",
        k_p=k_p,
        k_d=0.0,
        friction_left_right=friction,
        friction_rear=friction * 0.5,
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    settle_steps = max(1, int(settle_time / dt))
    drive_steps = max(1, int(drive_time / dt))
    total_steps = settle_steps + drive_steps

    config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
    )
    engine = AxionEngine(
        model=model, sim_steps=total_steps, config=config,
        logging_config=LoggingConfig(),
    )

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    # Controls
    zero_ctrl = model.control()
    drive_ctrl = model.control()
    tv = np.zeros(model.joint_dof_count, dtype=np.float32)
    tv[6] = wheel_speeds[0]  # left
    tv[7] = wheel_speeds[1]  # right
    tv[8] = wheel_speeds[2]  # rear
    wp.copy(
        drive_ctrl.joint_target_vel,
        wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device),
    )

    # Record trajectory (chassis body = index 0)
    xs, ys = [], []

    for step in range(total_steps):
        ctrl = zero_ctrl if step < settle_steps else drive_ctrl
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        state_in, state_out = state_out, state_in

        if step >= settle_steps:
            q = state_in.body_q.numpy().reshape(-1, 7)
            xs.append(float(q[0, 0]))
            ys.append(float(q[0, 1]))

    q_final = state_in.body_q.numpy().reshape(-1, 7)
    qd_final = state_in.body_qd.numpy().reshape(-1, 6)

    return {
        "x": xs,
        "y": ys,
        "final_x": float(q_final[0, 0]),
        "final_y": float(q_final[0, 1]),
        "final_vx": float(qd_final[0, 0]),
        "final_vy": float(qd_final[0, 1]),
        "dt": dt,
        "drive_steps": drive_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    dt_values = [0.005, 0.01, 0.02, 0.05, 0.1]

    print("=" * 60)
    print("  Helhest friction timestep-independence test")
    print("  Differential turn: left=4.0, right=2.0, rear=3.0 rad/s")
    print("  Duration: 3.0s, settle: 0.5s")
    print("=" * 60)

    results = {"dt_values": dt_values, "runs": []}

    for dt in dt_values:
        print(f"  dt={dt:.3f}s ({int(3.0/dt)} drive steps)...", end=" ", flush=True)
        run = run_drive(dt)
        results["runs"].append(run)
        print(f"final=({run['final_x']:.4f}, {run['final_y']:.4f})")

    # Compute deviations from finest dt
    ref = results["runs"][0]
    print(f"\n  Reference: dt={ref['dt']}s")
    print(f"  {'dt':>8} {'steps':>6} {'x (m)':>8} {'y (m)':>8} {'Δx (%)':>8} {'Δy (%)':>8}")
    print(f"  {'-'*50}")
    for run in results["runs"]:
        dx = abs(run["final_x"] - ref["final_x"]) / max(abs(ref["final_x"]), 1e-6) * 100
        dy = abs(run["final_y"] - ref["final_y"]) / max(abs(ref["final_y"]), 1e-6) * 100
        marker = "ref" if run["dt"] == ref["dt"] else f"{max(dx,dy):.1f}"
        print(f"  {run['dt']:8.3f} {run['drive_steps']:6d} {run['final_x']:8.4f} "
              f"{run['final_y']:8.4f} {dx:8.1f} {dy:8.1f}  {marker}")

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\n  Saved to {args.save}")

    if args.plot or args.show:
        plot_results(results, show=args.show)


def plot_results(results, show=False):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    colors = ["#1565C0", "#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for i, run in enumerate(results["runs"]):
        dt = run["dt"]
        label = f"$\\Delta t = {dt}$\\,s ({run['drive_steps']} steps)"
        ax.plot(run["x"], run["y"], color=colors[i], linewidth=2.0, label=label)

    ax.plot(run["x"][0], run["y"][0], "o", color="black", markersize=6, zorder=5)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", fontsize=11, framealpha=0.85)

    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "friction_dt_independence.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved plot to {out_path}")

    paper_dir = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"
    if paper_dir.resolve().is_dir():
        paper_path = paper_dir.resolve() / "friction_dt_independence.png"
        fig.savefig(paper_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {paper_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    main()
