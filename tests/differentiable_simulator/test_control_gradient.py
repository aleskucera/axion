"""Test 3: Single-step control gradient via finite differences.

Builds a revolute pendulum with PD position control. Perturbs joint_target_pos,
runs one forward step, and compares dL/d(joint_target_pos) from step_backward()
against central finite differences.

Tests the control constraint gradient path (control_target_grad_kernel).

NOTE: This test currently exposes a known discrepancy between the analytical
gradient from control_target_grad_kernel and finite differences. The velocity
gradient (test 2) passes, suggesting the adjoint body vector is correct but
the projection to joint_target_pos.grad has a scaling error.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
import newton
from axion import JointMode
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.simulation.trajectory_buffer import TrajectoryBuffer


def build_revolute_pendulum():
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()

    link = builder.add_link()
    builder.add_shape_box(
        link, hx=0.5, hy=0.1, hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0),
    )
    j0 = builder.add_joint_revolute(
        parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0),
        # Raised above ground to avoid spurious pendulum-ground friction contacts
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0], label="arm")

    # Box on the ground to ensure active contacts exist
    box = builder.add_body(
        xform=wp.transform(wp.vec3(5.0, 0.0, 0.6), wp.quat_identity()),
    )
    builder.add_shape_box(
        box, hx=0.5, hy=0.5, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )
    return builder.finalize_replicated(num_worlds=1, gravity=0.0)


def test_control_gradient_single_step():
    print("\n=== Test: Single-step control gradient (FD) ===")

    model = build_revolute_pendulum()
    config = AxionEngineConfig(
        max_newton_iters=20,
        max_linear_iters=200,
        linear_tol=1e-8,
        linear_atol=1e-8,
    )
    engine = AxionEngine(
        model=model, sim_steps=3, config=config,
        logging_config=LoggingConfig(), differentiable_simulation=True,
    )
    dt = 0.01

    np.random.seed(42)

    wp.copy(
        model.joint_dof_mode,
        wp.array(np.array([int(JointMode.TARGET_POSITION)], dtype=np.int32),
                 dtype=wp.int32, device=model.device),
    )
    wp.copy(
        model.joint_target_ke,
        wp.array(np.array([1000.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )
    wp.copy(
        model.joint_target_kd,
        wp.array(np.array([100.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()

    target_pos_val = np.array([0.5], dtype=np.float32)
    wp.copy(
        control.joint_target_pos,
        wp.array(target_pos_val, dtype=wp.float32, device=model.device),
    )

    qd_size = state_in.body_qd.numpy().flatten().shape[0]
    w = np.random.randn(qd_size).astype(np.float32)

    # --- Analytical gradient via adjoint ---
    state_out = model.state()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, dt)

    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=engine.dims, num_steps=1, device=model.device,
    )
    buffer.save_step(0, engine.data, engine.axion_contacts)
    buffer.zero_grad()

    w_reshaped = w.reshape(engine.data.body_vel_grad.numpy().shape)
    wp.copy(
        buffer.body_vel.grad[1],
        wp.array(w_reshaped, dtype=wp.spatial_vector, device=model.device),
    )

    buffer.load_step(0, engine.data, engine.axion_contacts)
    engine.data.zero_gradients()
    engine.step_backward()

    grad_analytical = engine.data.joint_target_pos.grad.numpy().flatten().copy()

    # --- Finite differences ---
    eps = 1e-4
    n_dof = len(target_pos_val)
    grad_fd = np.zeros(n_dof, dtype=np.float32)

    for i in range(n_dof):
        tp_plus = target_pos_val.copy()
        tp_plus[i] += eps
        wp.copy(control.joint_target_pos,
                wp.array(tp_plus, dtype=wp.float32, device=model.device))
        contacts = model.collide(state_in)
        s_out = model.state()
        engine.step(state_in, s_out, control, contacts, dt)
        loss_plus = np.dot(w, s_out.body_qd.numpy().flatten())

        tp_minus = target_pos_val.copy()
        tp_minus[i] -= eps
        wp.copy(control.joint_target_pos,
                wp.array(tp_minus, dtype=wp.float32, device=model.device))
        contacts = model.collide(state_in)
        s_out2 = model.state()
        engine.step(state_in, s_out2, control, contacts, dt)
        loss_minus = np.dot(w, s_out2.body_qd.numpy().flatten())

        grad_fd[i] = (loss_plus - loss_minus) / (2.0 * eps)

    # Compare
    abs_err = np.abs(grad_analytical[:n_dof] - grad_fd)
    denom = np.maximum(np.abs(grad_fd), np.abs(grad_analytical[:n_dof]))
    denom = np.maximum(denom, 1e-8)
    rel_err = abs_err / denom
    max_rel_err = np.max(rel_err)

    print(f"  Analytical grad: {grad_analytical[:n_dof]}")
    print(f"  FD grad:         {grad_fd}")
    print(f"  Max abs error:   {np.max(abs_err):.2e}")
    print(f"  Max rel error:   {max_rel_err:.4f}")

    assert max_rel_err < 0.1, f"Control gradient FD check failed: max rel error {max_rel_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_control_gradient_single_step()
