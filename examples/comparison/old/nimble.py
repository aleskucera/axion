"""Ball throw optimization using Nimble Physics (DART-based differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial 3D linear velocity of a free-floating ball to match a
target trajectory. Uses gradient descent with analytic gradients from Nimble's
BackpropSnapshot API (implicit differentiation through the LCP contact solver).

Setup:
    uv venv .venv-nimble --python 3.9
    uv pip install nimblephysics==0.10.35 "numpy<2.0" --python .venv-nimble/bin/python
"""
import argparse
import json
import pathlib
import time

import nimblephysics as nimble
import numpy as np

DT = 3e-2
DURATION = 1.5
T = int(DURATION / DT)  # 50 steps

INIT_VEL = np.array([0.0, 2.0, 1.0])    # initial guess  (vx, vy, vz)
TARGET_VEL = np.array([0.0, 4.0, 7.0])  # target initial velocity

BALL_RADIUS = 0.2
BALL_MASS = 1.0
BALL_START_HEIGHT = 1.0

LEARNING_RATE = 1e-2
MAX_GRAD = 100.0


def build_world() -> nimble.simulation.World:
    """Create a World with a free-floating ball and a ground plane."""
    world = nimble.simulation.World()
    world.setGravity(np.array([0.0, -9.81, 0.0]))
    world.setTimeStep(DT)

    # Free-floating ball: FreeJoint (6-DOF) + sphere shape
    ball = nimble.dynamics.Skeleton()
    ball.setName("ball")

    joint, body = ball.createFreeJointAndBodyNodePair()
    joint.setName("freejoint")
    body.setName("ball_body")

    # Mass and inertia
    inertia_val = 0.4 * BALL_MASS * BALL_RADIUS**2  # solid sphere: 2/5 m r²
    inertia = nimble.dynamics.Inertia(
        BALL_MASS,
        np.zeros(3),
        np.diag([inertia_val, inertia_val, inertia_val]),
    )
    body.setInertia(inertia)

    # Collision shape
    shape = nimble.dynamics.SphereShape(BALL_RADIUS)
    shape_node = body.createShapeNode(shape)
    shape_node.createCollisionAspect()
    shape_node.createDynamicsAspect()

    world.addSkeleton(ball)

    # Ground plane
    ground = nimble.dynamics.Skeleton()
    ground.setName("ground")
    weld_joint, ground_body = ground.createWeldJointAndBodyNodePair()
    ground_body.setName("ground_body")
    ground_shape = nimble.dynamics.BoxShape(np.array([100.0, 0.1, 100.0]))
    ground_node = ground_body.createShapeNode(ground_shape)
    ground_node.createCollisionAspect()
    ground_body.setMass(1e6)  # effectively static (weld joint)

    # Position ground so its top surface is at y=0
    world.addSkeleton(ground)

    return world


def set_ball_state(world: nimble.simulation.World, pos_xyz: np.ndarray, vel_xyz: np.ndarray):
    """Set the ball position (xyz) and linear velocity (xyz). Angles/angular vel = 0."""
    # FreeJoint DOFs: [rx, ry, rz, tx, ty, tz] (rotation then translation)
    q = np.zeros(6)
    q[3] = pos_xyz[0]
    q[4] = pos_xyz[1]
    q[5] = pos_xyz[2]

    dq = np.zeros(6)
    dq[3] = vel_xyz[0]
    dq[4] = vel_xyz[1]
    dq[5] = vel_xyz[2]

    world.setPositions(q)
    world.setVelocities(dq)


def get_ball_xyz(world: nimble.simulation.World) -> np.ndarray:
    q = world.getPositions()  # [rx, ry, rz, tx, ty, tz]
    return q[3:6].copy()


def rollout(world: nimble.simulation.World, init_vel: np.ndarray):
    """Run T forward steps, returning list of xyz positions and BackpropSnapshots."""
    set_ball_state(world, np.array([0.0, BALL_START_HEIGHT, 0.0]), init_vel)
    snapshots = []
    xyz_traj = []
    for _ in range(T):
        snap = nimble.neural.forwardPass(world)
        snapshots.append(snap)
        xyz_traj.append(get_ball_xyz(world))
    return xyz_traj, snapshots


def compute_loss_and_grad(
    world: nimble.simulation.World,
    init_vel: np.ndarray,
    target_xyz_traj: list,
):
    """One forward + backward pass. Returns (loss, grad wrt init_vel).

    Uses backpropState which takes dL/d(state_t+1) and returns a
    LossGradientHighLevelAPI whose lossWrtState is dL/d(state_t).
    State layout: [positions(6), velocities(6)].
    """
    xyz_traj, snapshots = rollout(world, init_vel)

    # Loss = sum of squared position errors over all timesteps
    loss = 0.0
    # dL/d(state) at each step: shape (T, 12) — positions are indices 0:6, vels 6:12
    dl_dstate = np.zeros((T, 12))
    for t in range(T):
        diff = xyz_traj[t] - target_xyz_traj[t]
        loss += float(np.dot(diff, diff))
        # xyz are translation DOFs (indices 3:6 in positions, i.e. state indices 3:6)
        dl_dstate[t, 3:6] = 2.0 * diff

    # Backprop through all timesteps (reverse order)
    # nextTimestepStateLossGrad starts at zero after the last step
    next_state_grad = np.zeros(12)
    for t in reversed(range(T)):
        # Accumulate direct loss contribution at this timestep
        next_state_grad += dl_dstate[t]
        result = snapshots[t].backpropState(world, next_state_grad)
        next_state_grad = np.array(result.lossWrtState)

    # next_state_grad now holds dL/d(state_0) = dL/d([q0, dq0])
    # Gradient w.r.t. initial linear velocity: translation DOFs 3:6 of velocity part
    # Velocity is at indices 6:12 in state; translation DOFs within that are 9:12
    grad_vel = next_state_grad[9:12].copy()
    return loss, grad_vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and run headless")
    parser.add_argument("--iters", type=int, default=30, help="Number of gradient steps")
    args = parser.parse_args()

    world = build_world()
    print(f"World: {world.getNumDofs()} DOFs, T={T}, dt={DT}")

    # --- Compute target trajectory (forward only) ---
    target_xyz_traj, _ = rollout(world, TARGET_VEL)
    print(
        f"Target final xyz: ({target_xyz_traj[-1][0]:.3f}, "
        f"{target_xyz_traj[-1][1]:.3f}, {target_xyz_traj[-1][2]:.3f})"
    )

    # --- Optimize ---
    vel = INIT_VEL.copy().astype(float)
    print(
        f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} "
        f"(gradient descent, Nimble implicit AD)"
    )
    results = {
        "simulator": "Nimble",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()
        loss, grad = compute_loss_and_grad(world, vel, target_xyz_traj)
        t_ms = (time.perf_counter() - t0) * 1000

        grad_clamped = np.clip(grad, -MAX_GRAD, MAX_GRAD)
        vel = vel - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) | "
            f"grad=({grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}) | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_ms)

        if loss < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")
    else:
        print(f"\nOptimized velocity: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
        print(f"Target velocity:    ({TARGET_VEL[0]:.3f}, {TARGET_VEL[1]:.3f}, {TARGET_VEL[2]:.3f})")


if __name__ == "__main__":
    main()
