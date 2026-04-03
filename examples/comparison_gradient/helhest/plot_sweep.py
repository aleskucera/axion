"""Plot best sweep trajectories vs ground truth for all simulators.

Reads sweep result JSONs, re-simulates each best config, and plots
XY trajectory + X(t) + Y(t) comparison.

Usage:
    python examples/comparison_gradient/helhest/plot_sweep.py \
        --ground-truth results/helhest_chrono.json \
        --axion results/sweep_axion.json \
        --mujoco results/sweep_mujoco.json \
        --dojo results/sweep_dojo.json \
        --tinydiffsim results/sweep_tinydiffsim.json \
        -o results/sweep_all_vs_chrono.png
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DURATION = 3.0


# ── Trajectory generators ──

def simulate_mujoco(params):
    sys.path.insert(0, os.path.dirname(__file__))
    from sweep_mujoco import HELHEST_XML_TEMPLATE, TARGET_CTRL
    import mujoco

    xml = HELHEST_XML_TEMPLATE.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T = int(DURATION / params["dt"])
    traj = [[0.0, 0.0]]
    for _ in range(T):
        data.ctrl[:] = TARGET_CTRL
        mujoco.mj_step(model, data)
        traj.append([float(data.qpos[0]), float(data.qpos[1])])
    return np.array(traj)


def simulate_tinydiffsim(params):
    sys.path.insert(0, os.path.dirname(__file__))
    from sweep_tinydiffsim import simulate
    return np.array(simulate(**params))


def simulate_axion(params):
    """Run in subprocess to isolate GPU memory."""
    tmp = tempfile.mktemp(suffix=".json")
    worker = f'''
import os, json, sys, numpy as np
os.environ["PYOPENGL_PLATFORM"] = "glx"
import newton, warp as wp
from axion import AxionDifferentiableSimulator, AxionEngineConfig, ExecutionConfig, LoggingConfig, RenderingConfig, SimulationConfig
from axion.simulation.sim_config import SyncMode
from examples.helhest.common import create_helhest_model, HelhestConfig

class Sim(AxionDifferentiableSimulator):
    def build_model(self):
        self.builder.rigid_gap = 0.1
        self.builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0))
        create_helhest_model(self.builder,
            xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p={params["k_p"]}, k_d=HelhestConfig.TARGET_KD,
            friction_left_right={params["friction_lr"]},
            friction_rear={params["friction_rear"]})
        return self.builder.finalize_replicated(num_worlds=1, requires_grad=False)
    def compute_loss(self): pass
    def update(self): pass

sim = Sim(
    SimulationConfig(duration_seconds=3.0, target_timestep_seconds={params["dt"]},
                     num_worlds=1, sync_mode=SyncMode.ALIGN_FPS_TO_DT),
    RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False),
    ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1),
    AxionEngineConfig(max_newton_iters=12, max_linear_iters=12, backtrack_min_iter=8,
        newton_atol=1e-1, linear_atol=1e-3, linear_tol=1e-3, enable_linesearch=False,
        joint_compliance=6e-8, contact_compliance=1e-6, friction_compliance=1e-6,
        regularization=1e-6, contact_fb_alpha=0.5, contact_fb_beta=1.0,
        friction_fb_alpha=1.0, friction_fb_beta=1.0, max_contacts_per_world=8),
    LoggingConfig(enable_timing=False, enable_hdf5_logging=False))

model = sim.model
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.states[0])
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.target_states[0])
T = sim.clock.total_sim_steps
num_dofs = sim.trajectory.joint_target_vel.shape[-1]
for i in range(T):
    ctrl = np.zeros(num_dofs, dtype=np.float32)
    ctrl[6] = 1.0; ctrl[7] = 6.0; ctrl[8] = 0.0
    wp.copy(sim.target_controls[i].joint_target_vel,
            wp.array(ctrl, dtype=wp.float32, device=model.device))
sim.run_target_episode()
traj = []
for t in range(sim.trajectory.target_body_pose.shape[0]):
    bp = sim.trajectory.target_body_pose[t].numpy()[0, 0]
    traj.append([float(bp[0]), float(bp[1])])
import pathlib
pathlib.Path("{tmp}").write_text(json.dumps(traj))
'''
    result = subprocess.run([sys.executable, "-c", worker],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Axion subprocess failed: {result.stderr[-500:]}")
    with open(tmp) as f:
        traj = json.load(f)
    os.unlink(tmp)
    return np.array(traj)


def simulate_dojo(params):
    """Run in subprocess via Julia."""
    tmp_json = tempfile.mktemp(suffix=".json")
    tmp_jl = tempfile.mktemp(suffix=".jl")

    jl_code = f"""
using Dojo, JSON, LinearAlgebra

const WHEEL_RADIUS = 0.36
const WHEEL_WIDTH = 0.11
const WHEEL_MASS = 5.5

function build_helhest(; timestep, kv, friction_front, friction_rear)
    origin = Origin{{Float64}}()
    chassis_mass = 85.0 + 2.0 + 7.0 + 7.0 + 7.0 + 3.0 + 3.0
    chassis = Box(0.26, 0.6, 0.18, chassis_mass; color=RGBA(0.5, 0.5, 0.5), name=:chassis)
    wo = Quaternion(cos(pi/4), sin(pi/4), 0.0, 0.0)
    mk = (nm) -> Capsule(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS; orientation_offset=wo, name=nm)
    lw, rw, rew = mk(:lw), mk(:rw), mk(:rew)
    j_base = JointConstraint(Floating(origin, chassis; spring=0.0, damper=0.0), name=:bj)
    j_l = JointConstraint(Revolute(chassis, lw, Y_AXIS; parent_vertex=[0,0.36,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jl)
    j_r = JointConstraint(Revolute(chassis, rw, Y_AXIS; parent_vertex=[0,-0.36,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jr)
    j_re = JointConstraint(Revolute(chassis, rew, Y_AXIS; parent_vertex=[-0.697,0,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jre)
    n = [0.0, 0.0, 1.0]
    function wc(w, fr, nms)
        origins = [[0.0, WHEEL_WIDTH/2, 0.0], [0.0, -WHEEL_WIDTH/2, 0.0]]
        contact_constraint(w, fill(n, 2); friction_coefficients=fill(fr, 2), contact_origins=origins, contact_radii=fill(WHEEL_RADIUS, 2), contact_type=:nonlinear, names=nms)
    end
    cl = wc(lw, friction_front, [:lc1,:lc2])
    cr = wc(rw, friction_front, [:rc1,:rc2])
    cre = wc(rew, friction_rear, [:rec1,:rec2])
    Mechanism(origin, [chassis, lw, rw, rew], [j_base, j_l, j_r, j_re], [cl; cr; cre]; gravity=-9.81, timestep)
end

function initial_z()
    local z = zeros(52)
    for (i, pos) in enumerate([[0,0,WHEEL_RADIUS],[0,0.36,WHEEL_RADIUS],[0,-0.36,WHEEL_RADIUS],[-0.697,0,WHEEL_RADIUS]])
        idx = 13*(i-1)
        z[idx+1:idx+3] = pos
        z[idx+7:idx+10] = [1,0,0,0]
    end
    return z
end

function main()
    mech = build_helhest(timestep={params['dt']}, kv={params['kv']},
        friction_front={params['friction_front']}, friction_rear={params['friction_rear']})
    local z = initial_z()
    local T = Int(3.0 / {params['dt']})
    local kv = {params['kv']}
    local ctrl = [1.0, 6.0, 0.0]
    local traj = [[z[1], z[2]]]
    for t in 1:T
        local u = zeros(9)
        u[7:9] = kv .* ctrl
        z = step!(mech, z, u)
        push!(traj, [z[1], z[2]])
    end
    open("{tmp_json}", "w") do io
        JSON.print(io, traj)
    end
end

main()
"""
    with open(tmp_jl, "w") as f:
        f.write(jl_code)

    julia = os.path.expanduser("~/.juliaup/bin/julia")
    result = subprocess.run([julia, "+1.10", "--startup-file=no", tmp_jl],
                            capture_output=True, text=True, timeout=300)
    os.unlink(tmp_jl)
    if result.returncode != 0:
        raise RuntimeError(f"Dojo subprocess failed: {result.stderr[-500:]}")
    with open(tmp_json) as f:
        traj = json.load(f)
    os.unlink(tmp_json)
    return np.array(traj)


# ── Plotting ──

def main():
    parser = argparse.ArgumentParser(description="Plot sweep trajectories vs ground truth")
    parser.add_argument("--ground-truth", required=True, help="Chrono ground truth JSON")
    parser.add_argument("--axion", help="Axion sweep result JSON")
    parser.add_argument("--mujoco", help="MuJoCo sweep result JSON")
    parser.add_argument("--dojo", help="Dojo sweep result JSON")
    parser.add_argument("--tinydiffsim", help="TinyDiffSim sweep result JSON")
    parser.add_argument("-o", "--output", default="results/sweep_all_vs_chrono.png",
                        help="Output image path (default: results/sweep_all_vs_chrono.png)")
    args = parser.parse_args()

    # Ground truth
    with open(args.ground_truth) as f:
        gt = json.load(f)
    chrono_traj = np.array(gt["target_trajectory"])
    print(f"Ground truth: {gt.get('simulator', '?')}, "
          f"final=({chrono_traj[-1, 0]:.3f}, {chrono_traj[-1, 1]:.3f})")

    # Collect (name, traj, error, dt, color, linestyle, linewidth)
    sims = [("Chrono (GT)", chrono_traj, None, gt["dt"], "k", "-", 2.5)]

    simulator_args = [
        ("Axion", args.axion, simulate_axion, "tab:blue", "-", 1.8),
        ("Dojo", args.dojo, simulate_dojo, "tab:green", "--", 1.8),
        ("MuJoCo", args.mujoco, simulate_mujoco, "tab:red", "--", 1.5),
        ("TinyDiffSim", args.tinydiffsim, simulate_tinydiffsim, "tab:orange", ":", 1.5),
    ]

    for name, sweep_path, sim_fn, color, ls, lw in simulator_args:
        if sweep_path is None:
            continue
        with open(sweep_path) as f:
            sweep = json.load(f)
        params = sweep["best_params"]
        err = sweep["best_error"]
        dt = params.get("dt", params.get("timestep", "?"))
        print(f"Simulating {name}: err={err:.4f}, dt={dt}, params={params}")
        try:
            traj = sim_fn(params)
            print(f"  -> {len(traj)} pts, final=({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")
            sims.append((name, traj, err, dt, color, ls, lw))
        except Exception as e:
            print(f"  -> FAILED: {e}")

    # ── Paper-quality plot with TeX fonts ──
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))

    for name, traj, err, dt, color, ls, lw in sims:
        t = np.linspace(0, DURATION, len(traj))
        if err is not None:
            label_xy = rf"{name} ($\bar{{e}}$={err:.2f}\,m)"
        else:
            label_xy = name

        axes[0].plot(traj[:, 0], traj[:, 1], color=color, ls=ls, lw=lw, label=label_xy)
        axes[1].plot(t, traj[:, 0], color=color, ls=ls, lw=lw, label=name)
        axes[2].plot(t, traj[:, 1], color=color, ls=ls, lw=lw, label=name)

    axes[0].set_xlabel(r"$x$ (m)")
    axes[0].set_ylabel(r"$y$ (m)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel(r"Time (s)")
    axes[1].set_ylabel(r"$x$ (m)")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_xlabel(r"Time (s)")
    axes[2].set_ylabel(r"$y$ (m)")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.subplots_adjust(bottom=0.28)

    # Single legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sims),
               frameon=True, framealpha=0.9, fontsize=6.5,
               handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.01))

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
