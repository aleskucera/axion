"""
Rigorous friction accuracy tests.

Compares Axion's implicit friction against analytical Coulomb friction
across multiple scenarios. Measures exact errors and their dependence
on timestep, solver iterations, and problem stiffness.
"""

import sys
from pathlib import Path

import warp as wp
wp.init()

import numpy as np
import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder


# ─── Helpers ────────────────────────────────────────────────────────────────

def build_box_on_slope(slope_deg, mu=0.5, box_mass=10.0, rigid_gap=0.05):
    """Single box on a tilted plane. Returns (model, engine_fn)."""
    slope_rad = np.radians(slope_deg)
    builder = AxionModelBuilder()
    builder.rigid_gap = rigid_gap

    # Tilted ground
    ground_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-slope_rad))
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), ground_rot),
        hx=20.0, hy=20.0, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
    )

    # Box on the slope surface
    # Place it above the slope so it drops and settles
    box_z = 1.0
    link = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, box_z), wp.quat_identity()),
        mass=box_mass,
    )
    builder.add_shape_box(
        body=link, xform=wp.transform_identity(),
        hx=0.2, hy=0.2, hz=0.2,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu),
    )
    j = builder.add_joint_free(parent=-1, child=link, label="box")
    builder.add_articulation([j], label="box")

    return builder


def simulate(builder, dt, total_time, newton_iters=16, linear_iters=16):
    """Run simulation, return list of (time, pos, vel) tuples."""
    num_steps = max(1, int(total_time / dt))
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)
    config = AxionEngineConfig(
        max_newton_iters=newton_iters, max_linear_iters=linear_iters,
    )
    engine = AxionEngine(
        model=model, sim_steps=num_steps, config=config,
        logging_config=LoggingConfig(),
    )
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    trajectory = []
    for step in range(num_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, model.control(), contacts, dt)
        state_in, state_out = state_out, state_in

        q = state_in.body_q.numpy().reshape(-1, 7)[0]
        qd = state_in.body_qd.numpy().reshape(-1, 6)[0]
        t = (step + 1) * dt
        trajectory.append((t, q[:3].copy(), qd[:3].copy()))

    return trajectory


# ─── Test 1: Static friction threshold ──────────────────────────────────────

def test_static_threshold():
    print("=" * 80)
    print("TEST 1: Static friction threshold")
    print("   Box on slope, mu=0.5. Coulomb: should stick for θ < atan(0.5) = 26.57°")
    print("   Measure displacement after 1s at various angles.")
    print("=" * 80)

    mu = 0.5
    critical_angle = np.degrees(np.arctan(mu))
    dt = 0.01
    sim_time = 1.0
    settle_time = 0.5

    print(f"   Critical angle (analytical): {critical_angle:.2f}°")
    print()
    print(f"{'angle':>8} {'disp (m)':>12} {'speed (m/s)':>12} {'expected':>10} {'actual':>10} {'correct':>8}")
    print("-" * 65)

    for angle in [5, 10, 15, 20, 25, 26, 26.5, 27, 28, 30, 35, 40]:
        builder = build_box_on_slope(angle, mu=mu)
        traj = simulate(builder, dt, settle_time + sim_time)

        # Take displacement from settle point to end
        settle_idx = int(settle_time / dt) - 1
        end_idx = -1
        pos_settle = traj[settle_idx][1]
        pos_end = traj[end_idx][1]
        vel_end = traj[end_idx][2]

        disp = np.linalg.norm(pos_end - pos_settle)
        speed = np.linalg.norm(vel_end)

        expected = "STICK" if angle < critical_angle else "SLIDE"
        actual = "STICK" if disp < 0.01 else "SLIDE"
        correct = "OK" if expected == actual else "WRONG"

        print(f"{angle:8.1f} {disp:12.6f} {speed:12.6f} {expected:>10} {actual:>10} {correct:>8}")


# ─── Test 2: Static friction leakage vs dt ──────────────────────────────────

def test_static_leakage():
    print()
    print("=" * 80)
    print("TEST 2: Static friction leakage (velocity residual)")
    print("   Box on 20° slope, mu=0.5 (well within Coulomb cone).")
    print("   Measure residual velocity after settling — should be exactly 0.")
    print("=" * 80)

    mu = 0.5
    angle = 20.0
    sim_time = 2.0

    print(f"{'dt':>8} {'iters':>6} {'residual_vel':>14} {'displacement':>14}")
    print("-" * 45)

    for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        for niters in [16, 64]:
            builder = build_box_on_slope(angle, mu=mu)
            traj = simulate(builder, dt, sim_time, newton_iters=niters, linear_iters=niters)

            pos_start = traj[0][1]
            pos_end = traj[-1][1]
            vel_end = traj[-1][2]
            disp = np.linalg.norm(pos_end - pos_start)
            res_vel = np.linalg.norm(vel_end)

            print(f"{dt:8.4f} {niters:6d} {res_vel:14.8f} {disp:14.8f}")


# ─── Test 3: Kinetic friction force ────────────────────────────────────────

def test_kinetic_friction():
    print()
    print("=" * 80)
    print("TEST 3: Kinetic friction (sliding box)")
    print("   Box on 35° slope, mu=0.3 (above Coulomb limit, must slide).")
    print("   Analytical: a = g*(sin(35°) - 0.3*cos(35°)) = 3.22 m/s²")
    print("   After 1s: v = 3.22 m/s")
    print("=" * 80)

    mu = 0.3
    angle = 35.0
    g = 9.81
    slope_rad = np.radians(angle)
    a_analytical = g * (np.sin(slope_rad) - mu * np.cos(slope_rad))
    sim_time = 1.5  # 0.5s settle + 1.0s slide

    print(f"   Analytical acceleration: {a_analytical:.4f} m/s²")
    print()
    print(f"{'dt':>8} {'v_sim (m/s)':>14} {'v_ana (m/s)':>14} {'rel_error':>12}")
    print("-" * 50)

    for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        builder = build_box_on_slope(angle, mu=mu)
        traj = simulate(builder, dt, sim_time)

        # After settling, the box should be sliding
        # Measure velocity along the slope direction at the end
        vel_end = traj[-1][2]
        # The sliding direction is along the slope in the xz plane
        slide_dir = np.array([-np.cos(slope_rad), 0, -np.sin(slope_rad)])
        v_slide = np.dot(vel_end, slide_dir)
        v_slide = abs(v_slide)

        # Analytical velocity after ~1s of sliding (approximate — settling time varies)
        # Find when the box started sliding (velocity > 0.1)
        slide_start = None
        for i, (t, p, v) in enumerate(traj):
            if np.linalg.norm(v) > 0.1:
                slide_start = t
                break

        if slide_start is not None:
            slide_duration = traj[-1][0] - slide_start
            v_analytical = a_analytical * slide_duration
            err = abs(v_slide - v_analytical) / max(v_analytical, 1e-10)
        else:
            v_analytical = 0.0
            err = float('inf')

        print(f"{dt:8.4f} {v_slide:14.4f} {v_analytical:14.4f} {err:12.4f}")


# ─── Test 4: Friction consistency across dt ─────────────────────────────────

def test_dt_consistency():
    print()
    print("=" * 80)
    print("TEST 4: Friction consistency across timestep")
    print("   Same scenario at different dt. Report spread in final state.")
    print("=" * 80)

    mu = 0.5
    sim_time = 2.0

    for label, angle in [("Sticking (20°)", 20.0), ("Sliding (35°, mu=0.3)", 35.0)]:
        test_mu = 0.5 if angle == 20.0 else 0.3
        print(f"\n  {label}:")
        print(f"  {'dt':>8} {'pos_x':>10} {'pos_z':>10} {'vel_x':>10} {'vel_z':>10}")
        print("  " + "-" * 45)

        positions = []
        for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
            builder = build_box_on_slope(angle, mu=test_mu)
            traj = simulate(builder, dt, sim_time)
            pos = traj[-1][1]
            vel = traj[-1][2]
            positions.append(pos)
            print(f"  {dt:8.4f} {pos[0]:10.4f} {pos[2]:10.4f} {vel[0]:10.4f} {vel[2]:10.4f}")

        positions = np.array(positions)
        spread_pos = np.max(positions, axis=0) - np.min(positions, axis=0)
        print(f"  {'SPREAD':>8} {spread_pos[0]:10.4f} {spread_pos[2]:10.4f}")


# ─── Test 5: Convergence study ─────────────────────────────────────────────

def test_convergence():
    print()
    print("=" * 80)
    print("TEST 5: Convergence with Newton iterations")
    print("   Box on 20° slope (sticking). Does more iterations reduce leakage?")
    print("=" * 80)

    mu = 0.5
    angle = 20.0
    dt = 0.03
    sim_time = 1.0

    print(f"  {'newton_iters':>14} {'linear_iters':>14} {'displacement':>14} {'residual_vel':>14}")
    print("  " + "-" * 60)

    for ni in [8, 16, 32, 64, 128]:
        for li in [16, 64]:
            builder = build_box_on_slope(angle, mu=mu)
            traj = simulate(builder, dt, sim_time, newton_iters=ni, linear_iters=li)
            pos_start = traj[0][1]
            pos_end = traj[-1][1]
            disp = np.linalg.norm(pos_end - pos_start)
            vel = np.linalg.norm(traj[-1][2])
            print(f"  {ni:14d} {li:14d} {disp:14.8f} {vel:14.8f}")


# ─── Run all ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_static_threshold()
    test_static_leakage()
    test_kinetic_friction()
    test_dt_consistency()
    test_convergence()
