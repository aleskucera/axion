"""Helhest trajectory optimization using Nimble Physics (DART, implicit AD).

Comparable to examples/comparison/helhest/helhest_mjx.py.

Optimizes K spline control points (linearly interpolated to per-timestep wheel
reference velocities) to match a target trajectory. Uses gradient descent with
analytic gradients from Nimble's BackpropSnapshot API (implicit differentiation
through the DART constraint solver).

Setup:
    uv venv .venv-nimble --python 3.9
    uv pip install nimblephysics==0.10.35 "numpy<2.0" --python .venv-nimble/bin/python
    .venv-nimble/bin/python examples/comparison/helhest/helhest_nimble.py
    .venv-nimble/bin/python examples/comparison/helhest/helhest_nimble.py --plot
    .venv-nimble/bin/python examples/comparison/helhest/helhest_nimble.py --save examples/comparison/results/helhest_nimble.json

-------------------------------------------------------------------------------
KNOWN LIMITATION: DART rolling-contact friction is broken for sphere-on-plane
-------------------------------------------------------------------------------

Nimble/DART 0.10.35 has a fundamental limitation: the LCP contact solver does
not generate reliable rolling traction for sphere-ground contacts. In practice,
regardless of the friction coefficient (tested 0.7–10.0), the friction forces
are near zero and the robot barely moves (~0.08 m/s instead of the expected
~1.26 m/s). This is a known DART limitation for this geometry.

A direct workaround of setting world.setAction() with setActionSpace() also
does not work: despite reporting the correct generalized force (e.g. 1944 N on
the x-DOF), the chassis position does not change. The fix is to use
world.setControlForces() directly, which correctly maps to generalized forces.

Physics workaround — explicit rolling-traction model:
  We bypass DART's contact solver entirely (no ground plane in the world) and
  model rolling propulsion with a soft no-slip constraint:

    tau_wheel[i]    = KV * (v_ref[i] - omega[i])    velocity servo torque
    F_traction[i]   = K_TR * (omega[i]*R - vx)      soft no-slip constraint force
    tau_chassis[x] += F_traction[i] * cos(theta)    propulsive force, world x
    tau_chassis[z] -= F_traction[i] * sin(theta)    propulsive force, world z
    tau_chassis[y] += ±F_traction[i] * half_track   differential yaw torque
    tau_wheel[i]   -= F_traction[i] * R             Newton 3rd law reaction

  At steady state: omega[i]*R = vx → F_traction = 0 (correct: no net force
  needed at constant speed). A height spring (K=1e4 N/m) holds the chassis at
  wheel-radius height, and stiff pitch/roll springs prevent tipping.

  Stability: with KV=5, K_TR=500 the coupled wheel-chassis system has a slow
  time constant of ~0.9 s and a fast time constant of ~5 ms; both are stable
  with DT=5 ms. Higher KV/K_TR cause the wheel to spin up in a single step,
  producing traction forces large enough to explode the simulation.

Coordinate system: y-up (gravity = [0, -9.81, 0])
  forward = +X, up = +Y, lateral = +Z
  left wheel at z = +HALF_TRACK, right wheel at z = -HALF_TRACK
  FreeJoint DOFs: q[0:3] = rotation (exp map), q[3:6] = translation

Gradient computation:
  Nimble's BackpropSnapshot.backpropState(world, next_state_grad) computes
  dL/d(state_t) treating the control forces as fixed. Because our control law
  depends on the state (omega, vx, theta, y), we add manual chain-rule
  corrections for every state-dependent term in tau:
    - Traction forces (depend on omega[i] and vx)
    - Yaw torque (depends on theta and omega[i])
    - Height spring (depends on q[y] and qdot[y])
    - Pitch/roll pins (depend on q[rx/rz] and qdot[rx/rz])
  The gradient w.r.t. v_ref[t] is simply la[6:9] * KV, where la =
  lossWrtAction; the traction forces do not depend on v_ref directly.

-------------------------------------------------------------------------------
FAIRNESS NOTE for the simulator comparison
-------------------------------------------------------------------------------

This script is NOT a fully fair comparison to the other helhest benchmarks
(helhest_axion.py, helhest_mjx.py, helhest_dojo.jl) because:

  1. Different physics: others use real ground-contact friction (mu=0.7);
     this uses an explicit spring-damper traction model with no ground plane.

  2. Scaled-down dynamics: KV=5 here vs KV=100 in all other simulators.
     The robot barely accelerates → trajectories span ~0.2 m instead of
     several metres, making the initial loss ~100× smaller (0.08 vs 6–16).

  3. Shorter duration: 1.5 s here vs 3.0 s in all other simulators.

  These differences mean the loss convergence curves are NOT comparable across
  simulators. The per-iteration wall-clock time (ms/iter) IS comparable and
  is the only metric that should be used in the timing subplot.

  Benchmark results (50 iterations each):
    Axion       T=60,   dt=50ms,  loss 6.64→0.07,  median  333 ms/iter
    Dojo        T=100,  dt=30ms,  loss 15.98→3.29,  median  521 ms/iter
    Nimble      T=300,  dt=5ms,   loss 0.08→0.002, median   34 ms/iter  ← this script
    MJX         T=1500, dt=2ms,   loss 8.38→0.018, median 5334 ms/iter
"""
import argparse
import json
import pathlib
import time
from math import cos, sin

import nimblephysics as nimble
import numpy as np

DT       = 5e-3        # timestep (s)
DURATION = 1.5         # total simulation duration (s)
T        = int(DURATION / DT)  # number of steps
K        = 10          # number of spline control points

WHEEL_RADIUS = 0.36    # m
HALF_TRACK   = 0.36    # m  (lateral distance from chassis centre to wheel)
KV           = 5.0     # velocity servo gain (N·m / (rad/s))
K_TR         = 500.0   # rolling traction stiffness (N·s/m)

# Spring keeping chassis at ground height
HEIGHT_KP = 1e4        # N/m
HEIGHT_KD = 500.0      # N·s/m

# Pitch/roll pinning springs
K_PIN = 2e3
D_PIN = 100.0

TARGET_CTRL = np.array([1.0, 6.0, 0.0])   # target wheel angular velocities (rad/s)
INIT_CTRL   = np.array([2.0, 5.0, 0.0])   # initial guess

TRAJECTORY_WEIGHT     = 10.0
SMOOTHNESS_WEIGHT     = 1e-2
REGULARIZATION_WEIGHT = 1e-7

# FreeJoint DOF layout (DART):  q[0:3]=rot_exp, q[3:6]=translation
# Velocity layout: q_dot[0:3]=ang_vel, q_dot[3:6]=lin_vel
# State layout: [q(9), q_dot(9)]  (18 elements total)
_RX, _RY, _RZ = 0, 1, 2          # rotation DOF indices
_TX, _TY, _TZ = 3, 4, 5          # translation DOF indices (= position indices in q)
# State indices (offset by 9 for velocities):
_S_RX, _S_RY, _S_RZ       = 0, 1, 2
_S_TX, _S_TY, _S_TZ        = 3, 4, 5
_S_OMX, _S_OMY, _S_OMZ    = 9, 10, 11
_S_VX, _S_VY, _S_VZ        = 12, 13, 14
_S_WL, _S_WR, _S_WRear     = 15, 16, 17  # wheel angular velocities in state

TOTAL_MASS = 85.0 + 3 * 5.5      # chassis + 3 wheels


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def build_world() -> nimble.simulation.World:
    """Build Helhest world: chassis + 3 wheels, y-up."""
    world = nimble.simulation.World()
    world.setGravity(np.array([0.0, -9.81, 0.0]))
    world.setTimeStep(DT)

    helhest = nimble.dynamics.Skeleton()
    helhest.setName("helhest")

    # Chassis
    _, chassis = helhest.createFreeJointAndBodyNodePair()
    chassis.setName("chassis")
    chassis.setMass(85.0)
    chassis.setInertia(
        nimble.dynamics.Inertia(85.0, np.array([-0.047, 0.0, 0.0]),
                                np.diag([0.6213, 0.1583, 0.677]))
    )

    # Wheels — revolute around -Z axis (positive vel → forward in +X)
    wheel_specs = [
        ("left_wheel",  np.array([0.0,   0.0,  HALF_TRACK])),   # DOF 6
        ("right_wheel", np.array([0.0,   0.0, -HALF_TRACK])),   # DOF 7
        ("rear_wheel",  np.array([-0.697, 0.0, 0.0])),           # DOF 8
    ]
    for name, pos_chassis in wheel_specs:
        j, b = helhest.createRevoluteJointAndBodyNodePair(chassis)
        j.setName(f"{name}_j")
        b.setName(name)
        b.setMass(5.5)
        b.setInertia(
            nimble.dynamics.Inertia(5.5, np.zeros(3),
                                    np.diag([0.20045, 0.20045, 0.3888]))
        )
        j.setAxis(np.array([0.0, 0.0, -1.0]))
        tf = nimble.math.Isometry3()
        tf.set_translation(pos_chassis)
        j.setTransformFromParentBodyNode(tf)

    world.addSkeleton(helhest)
    return world


# ---------------------------------------------------------------------------
# Spline
# ---------------------------------------------------------------------------

def make_interp_matrix(T: int, K: int) -> np.ndarray:
    W = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low  = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha  = k_float - k_low
        W[t, k_low]  += 1.0 - alpha
        W[t, k_high] += alpha
    return W


# ---------------------------------------------------------------------------
# Control law
# ---------------------------------------------------------------------------

def compute_control(q, qdot, v_ref_t):
    """Compute 9-DOF control force vector.

    Returns tau (9,) and auxiliary values needed for the backward pass.
    """
    omega = qdot[6:9]          # [left, right, rear] angular vels
    vx    = qdot[3]            # chassis forward velocity (world x)
    theta = q[1]               # chassis yaw
    cos_t = cos(theta)
    sin_t = sin(theta)

    # Rolling traction forces (soft no-slip constraint)
    R = WHEEL_RADIUS
    F_left  = K_TR * (omega[0] * R - vx)
    F_right = K_TR * (omega[1] * R - vx)
    F_rear  = K_TR * (omega[2] * R - vx)
    F_fwd   = F_left + F_right + F_rear

    tau = np.zeros(9)

    # Chassis: forward force (world frame, projected from body frame)
    tau[_TX] += F_fwd * cos_t
    tau[_TZ] -= F_fwd * sin_t
    # Yaw torque from differential traction
    tau[_RY] += HALF_TRACK * (F_left - F_right)
    # Wheel reaction torques (Newton's 3rd law) + servo
    tau[6] += -F_left  * R + KV * (v_ref_t[0] - omega[0])
    tau[7] += -F_right * R + KV * (v_ref_t[1] - omega[1])
    tau[8] += -F_rear  * R + KV * (v_ref_t[2] - omega[2])

    # Height spring
    tau[_TY] = HEIGHT_KP * (WHEEL_RADIUS - q[_TY]) - HEIGHT_KD * qdot[4] + TOTAL_MASS * 9.81

    # Pin pitch and roll to prevent tipping
    tau[_RX] = -K_PIN * q[_RX] - D_PIN * qdot[0]
    tau[_RZ] = -K_PIN * q[_RZ] - D_PIN * qdot[2]

    aux = dict(F_left=F_left, F_right=F_right, F_rear=F_rear, F_fwd=F_fwd,
               cos_t=cos_t, sin_t=sin_t, omega=omega.copy(), vx=vx)
    return tau, aux


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def reset_world(world: nimble.simulation.World):
    q = np.zeros(9)
    q[_TY] = WHEEL_RADIUS
    world.setPositions(q)
    world.setVelocities(np.zeros(9))


def rollout_forward(world: nimble.simulation.World,
                    v_ref_traj: np.ndarray,
                    record: bool = False):
    """Run T steps. Returns (xz_traj (T,2), snapshots, aux_list)."""
    reset_world(world)
    xz_traj   = np.zeros((T, 2))
    snapshots  = [] if record else None
    aux_list   = [] if record else None

    for t in range(T):
        q    = world.getPositions()
        qdot = world.getVelocities()
        tau, aux = compute_control(q, qdot, v_ref_traj[t])
        world.setControlForces(tau)

        if record:
            snap = nimble.neural.forwardPass(world)
            snapshots.append(snap)
            aux_list.append(aux)
        else:
            world.step()

        q_post = world.getPositions()
        xz_traj[t, 0] = q_post[_TX]
        xz_traj[t, 1] = q_post[_TZ]

    return xz_traj, snapshots, aux_list


# ---------------------------------------------------------------------------
# Loss & gradient
# ---------------------------------------------------------------------------

def trajectory_loss(xz_traj, target_xz, ctrl_traj):
    delta  = xz_traj - target_xz
    track  = TRAJECTORY_WEIGHT / T * float(np.sum(delta ** 2))
    smooth = SMOOTHNESS_WEIGHT  * float(np.sum((ctrl_traj[1:] - ctrl_traj[:-1]) ** 2))
    reg    = REGULARIZATION_WEIGHT * float(np.sum(ctrl_traj ** 2))
    return track + smooth + reg


def compute_loss_and_grad(world, W, params, target_xz):
    ctrl_traj = W @ params                         # (T, 3) per-step v_ref
    xz_traj, snapshots, aux_list = rollout_forward(world, ctrl_traj, record=True)

    loss = trajectory_loss(xz_traj, target_xz, ctrl_traj)

    # Direct tracking gradient
    dl_dxz = 2.0 * TRAJECTORY_WEIGHT / T * (xz_traj - target_xz)  # (T, 2)

    # Smoothness + regularisation gradient on ctrl_traj
    dl_dctrl = np.zeros_like(ctrl_traj)
    dl_dctrl += 2.0 * REGULARIZATION_WEIGHT * ctrl_traj
    diff = ctrl_traj[1:] - ctrl_traj[:-1]
    dl_dctrl[1:]  += 2.0 * SMOOTHNESS_WEIGHT * diff
    dl_dctrl[:-1] -= 2.0 * SMOOTHNESS_WEIGHT * diff

    # --- Backward pass ---
    next_state_grad = np.zeros(18)
    dl_dvref = np.zeros((T, 3))

    R  = WHEEL_RADIUS
    HT = HALF_TRACK

    for t in reversed(range(T)):
        # Accumulate direct tracking gradient (x=state[3], z=state[5])
        next_state_grad[_S_TX] += dl_dxz[t, 0]
        next_state_grad[_S_TZ] += dl_dxz[t, 1]

        # Backprop through nimble's dynamics (treats tau as fixed input)
        result = snapshots[t].backpropState(world, next_state_grad)
        next_state_grad = np.array(result.lossWrtState)   # (18,) dL/d(state_t)
        la = result.lossWrtAction                          # (9,)  dL/d(tau_t)

        # Retrieve pre-step kinematics saved during forward pass
        aux   = aux_list[t]
        F_left  = aux["F_left"]
        F_right = aux["F_right"]
        F_rear  = aux["F_rear"]
        F_fwd   = aux["F_fwd"]
        cos_t   = aux["cos_t"]
        sin_t   = aux["sin_t"]

        # --- Chain-rule correction: tau depends on state ---
        # tau[TY] = HEIGHT_KP*(R - q[TY]) - HEIGHT_KD*qdot[4] + M*g
        #   d_tau[TY]/dq[TY]    = -HEIGHT_KP  → state index _S_TY = 4
        #   d_tau[TY]/dqdot[4] = -HEIGHT_KD  → state index _S_VY = 13
        next_state_grad[_S_TY] += la[_TY] * (-HEIGHT_KP)
        next_state_grad[_S_VY] += la[_TY] * (-HEIGHT_KD)

        # tau[RX] = -K_PIN*q[0] - D_PIN*qdot[0]
        next_state_grad[_S_RX]  += la[_RX] * (-K_PIN)
        next_state_grad[_S_OMX] += la[_RX] * (-D_PIN)

        # tau[RZ] = -K_PIN*q[2] - D_PIN*qdot[2]
        next_state_grad[_S_RZ]  += la[_RZ] * (-K_PIN)
        next_state_grad[_S_OMZ] += la[_RZ] * (-D_PIN)

        # Traction force corrections (tau[TX], tau[TZ], tau[RY], tau[6], tau[7], tau[8]
        # all depend on omega_i and vx):

        # w.r.t. yaw theta = q[1] (state index _S_RY = 1):
        #   d_tau[TX]/dtheta = -F_fwd * sin_t
        #   d_tau[TZ]/dtheta = -F_fwd * cos_t
        next_state_grad[_S_RY] += la[_TX] * (-F_fwd * sin_t) + la[_TZ] * (-F_fwd * cos_t)

        # w.r.t. vx = qdot[3] (state index _S_VX = 12):
        #   dF_i/dvx = -K_TR for each i (3 wheels)
        #   d_tau[TX]/dvx = cos_t * (-3*K_TR)
        #   d_tau[TZ]/dvx = -sin_t * (-3*K_TR)
        #   d_tau[RY]/dvx = HT*(−K_TR − (−K_TR)) = 0  (cancels)
        #   d_tau[6]/dvx = K_TR*R  (from -F_left*R, F_left = K_TR*(omega*R-vx))
        #   d_tau[7]/dvx = K_TR*R
        #   d_tau[8]/dvx = K_TR*R
        next_state_grad[_S_VX] += (la[_TX] * (-3.0 * K_TR * cos_t)
                                   + la[_TZ] * (3.0 * K_TR * sin_t)
                                   + la[6] * (K_TR * R)
                                   + la[7] * (K_TR * R)
                                   + la[8] * (K_TR * R))

        # w.r.t. omega_left = qdot[6] (state index _S_WL = 15):
        #   d_tau[TX]/domega_L = K_TR*R * cos_t
        #   d_tau[TZ]/domega_L = -K_TR*R * sin_t
        #   d_tau[RY]/domega_L = HT * K_TR * R  (from +F_left)
        #   d_tau[6]/domega_L  = -K_TR*R² - KV
        next_state_grad[_S_WL] += (la[_TX] * (K_TR * R * cos_t)
                                   + la[_TZ] * (-K_TR * R * sin_t)
                                   + la[_RY] * (HT * K_TR * R)
                                   + la[6] * (-(K_TR * R * R + KV)))

        # w.r.t. omega_right = qdot[7] (state index _S_WR = 16):
        #   d_tau[TX]/domega_R = K_TR*R * cos_t
        #   d_tau[TZ]/domega_R = -K_TR*R * sin_t
        #   d_tau[RY]/domega_R = -HT * K_TR * R  (from -F_right)
        #   d_tau[7]/domega_R  = -K_TR*R² - KV
        next_state_grad[_S_WR] += (la[_TX] * (K_TR * R * cos_t)
                                   + la[_TZ] * (-K_TR * R * sin_t)
                                   + la[_RY] * (-HT * K_TR * R)
                                   + la[7] * (-(K_TR * R * R + KV)))

        # w.r.t. omega_rear = qdot[8] (state index _S_WRear = 17):
        #   d_tau[TX]/domega_rear = K_TR*R * cos_t
        #   d_tau[TZ]/domega_rear = -K_TR*R * sin_t
        #   d_tau[RY]/domega_rear = 0 (rear wheel at centre, no yaw contribution)
        #   d_tau[8]/domega_rear  = -K_TR*R² - KV
        next_state_grad[_S_WRear] += (la[_TX] * (K_TR * R * cos_t)
                                      + la[_TZ] * (-K_TR * R * sin_t)
                                      + la[8] * (-(K_TR * R * R + KV)))

        # --- Gradient w.r.t. v_ref_t (only via wheel servo: d_tau[6,7,8]/dv_ref = KV) ---
        dl_dvref[t, 0] += la[6] * KV
        dl_dvref[t, 1] += la[7] * KV
        dl_dvref[t, 2] += la[8] * KV

        # Smoothness/regularisation gradient
        dl_dvref[t] += dl_dctrl[t]

    grad_params = W.T @ dl_dvref   # (K, 3)
    return loss, grad_params


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, shape, lr=0.05, betas=(0.9, 0.999), eps=1e-8, clip=50.0):
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.clip = clip
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, params, grad):
        self.t += 1
        grad = np.clip(grad, -self.clip, self.clip)
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_trajectory(target_xz, init_xz, opt_xz=None):
    """Plot 2D (x, z) trajectories. z is the lateral axis (world right = +Z)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(target_xz[:, 0], target_xz[:, 1], "b-",  linewidth=2, label="Target")
    ax.plot(init_xz[:, 0],   init_xz[:, 1],   "r--", linewidth=2, label="Initial guess")
    if opt_xz is not None:
        ax.plot(opt_xz[:, 0], opt_xz[:, 1], "g-", linewidth=2, label="Optimized")
    # Mark start
    ax.plot(0, 0, "ko", markersize=8, label="Start")
    # Mark ends
    ax.plot(target_xz[-1, 0], target_xz[-1, 1], "b*", markersize=12)
    ax.plot(init_xz[-1, 0],   init_xz[-1, 1],   "r*", markersize=12)
    if opt_xz is not None:
        ax.plot(opt_xz[-1, 0], opt_xz[-1, 1], "g*", markersize=12)

    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("z (m, lateral)")
    ax.set_title(f"Helhest trajectory (Nimble, T={T}, dt={DT}s)\n"
                 f"Target ctrl={TARGET_CTRL}, Init ctrl={INIT_CTRL}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--plot", action="store_true", help="Plot trajectories after optimizing")
    args = parser.parse_args()

    W     = make_interp_matrix(T, K)
    world = build_world()
    print(f"World: {world.getNumDofs()} DOFs, T={T}, dt={DT}, K={K}")

    # Target trajectory
    target_v_traj = np.tile(TARGET_CTRL, (T, 1))
    target_xz, _, _ = rollout_forward(world, target_v_traj)
    print(f"Target final xz: ({target_xz[-1, 0]:.3f}, {target_xz[-1, 1]:.3f})")
    print(f"Target max |x|={np.abs(target_xz[:,0]).max():.3f}, "
          f"max |z|={np.abs(target_xz[:,1]).max():.3f}")

    # Initial trajectory (for plot)
    init_v_traj = np.tile(INIT_CTRL, (T, 1))
    init_xz, _, _ = rollout_forward(world, init_v_traj)

    params    = np.tile(INIT_CTRL, (K, 1)).astype(np.float64)
    optimizer = Adam(params.shape, lr=0.05)

    print(f"\nOptimizing: T={T}, dt={DT}, K={K}, lr=0.05 (Adam, Nimble implicit AD)")
    results = {
        "simulator": "Nimble",
        "problem": "helhest",
        "dt": DT,
        "T": T,
        "K": K,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()
        loss, grad = compute_loss_and_grad(world, W, params, target_xz)
        t_ms = (time.perf_counter() - t0) * 1000

        p0, pm, pN = params[0], params[K // 2], params[-1]
        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"cp[0]=({p0[0]:.2f},{p0[1]:.2f}) "
            f"cp[{K//2}]=({pm[0]:.2f},{pm[1]:.2f}) "
            f"cp[-1]=({pN[0]:.2f},{pN[1]:.2f}) | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_ms)

        params = optimizer.step(params, grad)

        if loss < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")

    if args.plot:
        opt_v_traj = W @ params
        opt_xz, _, _ = rollout_forward(world, opt_v_traj)
        plot_trajectory(target_xz, init_xz, opt_xz)


if __name__ == "__main__":
    main()
