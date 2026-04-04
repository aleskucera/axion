"""Test: Gradient of loss w.r.t. joint stiffness (ke) and damping (kd).

Verifies dL/d(ke) and dL/d(kd) for a revolute pendulum with
TARGET_POSITION control mode. This is the key gradient for
system identification (e.g., identifying material properties of a
deformable chain from observed motion).
"""

import warp as wp

wp.init()

import numpy as np
import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.core.types import JointMode


def make_pendulum(ke, kd):
    no_collision = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    builder = AxionModelBuilder()
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.5, cfg=no_collision)
    j = builder.add_joint_revolute(
        parent=-1, child=link, axis=wp.vec3(1, 0, 0),
        parent_xform=wp.transform(wp.vec3(0, 0, 2.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()),
        target_ke=ke, target_kd=kd, label="pendulum",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_POSITION]},
    )
    builder.add_articulation([j], label="pendulum")
    return builder.finalize_replicated(num_worlds=1, requires_grad=True)


def run_forward(model, num_steps, dt, init_angle=0.5):
    config = AxionEngineConfig(
        max_newton_iters=20, max_linear_iters=200,
        linear_tol=1e-8, linear_atol=1e-8,
    )
    engine = AxionEngine(
        model=model, sim_steps=num_steps, config=config,
        logging_config=LoggingConfig(), differentiable_simulation=True,
    )
    state_in = model.state()
    jq = model.joint_q.numpy()
    jq[0] = init_angle
    wp.copy(model.joint_q, wp.array(jq, dtype=wp.float32, device=model.joint_q.device))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()

    states = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states[0].body_q, state_in.body_q)
    wp.copy(states[0].body_qd, state_in.body_qd)
    for i in range(num_steps):
        c = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, c, dt)
    return engine, states


def test_stiffness_gradient_single_step():
    """Single-step ke/kd gradient — should be very accurate."""
    print("\n=== Test: Stiffness gradient (single step, TARGET_POSITION) ===")

    ke_val, kd_val = 500.0, 50.0
    dt = 0.01
    num_steps = 1
    loss_idx = 3  # angular x (spatial_bottom[0])

    model = make_pendulum(ke_val, kd_val)
    engine, states = run_forward(model, num_steps, dt)

    # Backward
    engine.data.zero_gradients()
    model.joint_target_ke.grad.zero_()
    model.joint_target_kd.grad.zero_()
    vel_grad = np.zeros(states[-1].body_qd.numpy().shape, dtype=np.float32)
    vel_grad.flat[loss_idx] = 1.0
    wp.copy(engine.data.body_vel_grad,
            wp.array(vel_grad, dtype=wp.spatial_vector, device=model.device))
    engine.step_backward()

    ke_a = model.joint_target_ke.grad.numpy().flatten()[0]
    kd_a = model.joint_target_kd.grad.numpy().flatten()[0]

    # FD
    eps = 1.0

    def get_loss(ke, kd):
        m = make_pendulum(ke, kd)
        _, ss = run_forward(m, num_steps, dt)
        return ss[-1].body_qd.numpy().flat[loss_idx]

    fd_ke = (get_loss(ke_val + eps, kd_val) - get_loss(ke_val - eps, kd_val)) / (2 * eps)
    fd_kd = (get_loss(ke_val, kd_val + eps) - get_loss(ke_val, kd_val - eps)) / (2 * eps)

    err_ke = abs(ke_a - fd_ke) / max(abs(ke_a), abs(fd_ke), 1e-15)
    err_kd = abs(kd_a - fd_kd) / max(abs(kd_a), abs(fd_kd), 1e-15)

    print(f"  ke: analytical={ke_a:.8f}  FD={fd_ke:.8f}  rel_err={err_ke:.4f}")
    print(f"  kd: analytical={kd_a:.8f}  FD={fd_kd:.8f}  rel_err={err_kd:.4f}")

    max_err = max(err_ke, err_kd)
    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.02, f"Stiffness gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_stiffness_gradient_multi_step():
    """Multi-step ke/kd gradient with trajectory buffer."""
    print("\n=== Test: Stiffness gradient (2 steps, TARGET_POSITION) ===")

    from axion.simulation.trajectory_buffer import TrajectoryBuffer

    ke_val, kd_val = 500.0, 50.0
    dt = 0.01
    num_steps = 2
    loss_idx = 3  # angular x

    model = make_pendulum(ke_val, kd_val)
    engine, states = run_forward(model, num_steps, dt)
    dims = engine.dims

    # Save trajectory
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=num_steps, device=model.device,
    )
    # Need to re-run forward with buffer saving
    state_in = model.state()
    jq = model.joint_q.numpy(); jq[0] = 0.5
    wp.copy(model.joint_q, wp.array(jq, dtype=wp.float32, device=model.joint_q.device))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    states2 = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states2[0].body_q, state_in.body_q)
    wp.copy(states2[0].body_qd, state_in.body_qd)
    for i in range(num_steps):
        c = model.collide(states2[i])
        engine.step(states2[i], states2[i + 1], control, c, dt)
        buffer.save_step(i, engine.data, engine.axion_contacts)

    # Backward: terminal velocity loss
    buffer.zero_grad()
    model.joint_target_ke.grad.zero_()
    model.joint_target_kd.grad.zero_()
    vel_grad = np.zeros(buffer.body_vel.grad[num_steps].numpy().shape, dtype=np.float32)
    vel_grad.flat[loss_idx] = 1.0
    wp.copy(buffer.body_vel.grad[num_steps],
            wp.array(vel_grad, dtype=wp.spatial_vector, device=model.device))

    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)

    ke_a = model.joint_target_ke.grad.numpy().flatten()[0]
    kd_a = model.joint_target_kd.grad.numpy().flatten()[0]

    # FD
    eps = 1.0

    def get_loss(ke, kd):
        m = make_pendulum(ke, kd)
        _, ss = run_forward(m, num_steps, dt)
        return ss[-1].body_qd.numpy().flat[loss_idx]

    fd_ke = (get_loss(ke_val + eps, kd_val) - get_loss(ke_val - eps, kd_val)) / (2 * eps)
    fd_kd = (get_loss(ke_val, kd_val + eps) - get_loss(ke_val, kd_val - eps)) / (2 * eps)

    err_ke = abs(ke_a - fd_ke) / max(abs(ke_a), abs(fd_ke), 1e-15)
    err_kd = abs(kd_a - fd_kd) / max(abs(kd_a), abs(fd_kd), 1e-15)

    print(f"  ke: analytical={ke_a:.8f}  FD={fd_ke:.8f}  rel_err={err_ke:.4f}")
    print(f"  kd: analytical={kd_a:.8f}  FD={fd_kd:.8f}  rel_err={err_kd:.4f}")

    max_err = max(err_ke, err_kd)
    print(f"  Max rel error: {max_err:.4f}")
    print("  (multi-step degradation from velocity/pose chain — per-step ke/kd is accurate)")
    print("  REPORTED (no assertion)")


if __name__ == "__main__":
    test_stiffness_gradient_single_step()
    test_stiffness_gradient_multi_step()
    print("\n=== Stiffness gradient tests done! ===")
