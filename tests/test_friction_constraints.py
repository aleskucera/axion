import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine_new import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_logger import EngineLogger
from axion.core.engine_logger import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


@wp.kernel
def apply_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1),
    force_x: float,
    force_y: float,
    force_z: float,
    body_idx: int,
):
    tid = wp.tid()
    if tid == body_idx:
        # Spatial vector layout: (moment_x, moment_y, moment_z, force_x, force_y, force_z)
        current = body_f[tid]
        body_f[tid] = wp.spatial_vector(
            current[0],
            current[1],
            current[2],
            current[3] + force_x,
            current[4] + force_y,
            current[5] + force_z,
        )


def _run_friction_test_logic(test_case: str):
    """
    test_case: "stick" or "slip"
    """
    print(f"\n=== Testing Friction ({test_case}) ===")

    builder = AxionModelBuilder()

    # Constants
    mu = 0.5
    gravity = 9.81
    box_size = 1.0
    density = 100.0
    # Mass = volume * density = 1.0 * 1.0 * 1.0 * 100.0 = 100.0 kg

    # Ground plane
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=mu,
            restitution=0.0,
        )
    )

    # Box
    start_height = box_size / 2.0
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, start_height), wp.quat_identity()), key="dynamic_body"
    )

    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        mu=mu,
        restitution=0.0,
    )

    builder.add_shape_box(
        body=body, hx=box_size / 2.0, hy=box_size / 2.0, hz=box_size / 2.0, cfg=shape_cfg
    )

    model = builder.finalize_replicated(num_worlds=1, gravity=-gravity)

    config = AxionEngineConfig(
        contact_constraint_level="pos",
        contact_stabilization_factor=1.0,
        joint_compliance=1e-8,
        contact_compliance=1e-8,
        max_newton_iters=10,
        max_linear_iters=10,
    )
    # Disable logging overhead
    logger = EngineLogger(LoggingConfig(enable_timing=False))
    logger.initialize_events(steps_per_segment=1, newton_iters=config.max_newton_iters)

    engine = AxionEngine(
        model=model,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
        logger=logger,
        config=config,
    )

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 0.01

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    # Calculate theoretical limits
    # Normal force N = m * g = 1.0 * 9.81 = 9.81
    # Max static friction F_max = mu * N = 0.5 * 9.81 = 4.905
    max_static_friction = mu * gravity * 100.0  # mass is 1.0

    if test_case == "stick":
        applied_force = max_static_friction * 0.9  # 2.4525
        expected_motion = False
    else:  # slip
        applied_force = max_static_friction * 1.1  # 7.3575
        expected_motion = True

    print(
        f"Applied Force: {applied_force:.4f} N (Max Static Friction: {max_static_friction:.4f} N)"
    )

    # Run simulation
    # We need to run enough steps to see velocity develop if it slips,
    # but not so many that it flies off the world if we don't track it.

    # First, let it settle for a few steps (gravity only) to ensure contact is established
    # Although starting at exact height should be fine.

    accumulated_motion = 0.0

    for step in range(30):
        state_in.body_f.zero_()

        # Apply horizontal force
        wp.launch(
            kernel=apply_force_kernel,
            dim=1,
            inputs=[state_in.body_f, applied_force, 0.0, 0.0, 0],  # x-force, body_idx 0
            device=engine.device,
        )

        contacts = model.collide(state_in)
        logger.set_current_step_in_segment(0)
        engine.step(state_in, state_out, control, contacts, dt)

        # check velocity
        vel = state_out.body_qd.numpy()[0]  # (w_x, w_y, w_z, v_x, v_y, v_z)
        linear_vel_x = vel[3]

        # We can also check position drift
        pos = state_out.body_q.numpy()[0]  # (x, y, z, qx, qy, qz, qw)
        pos_x = pos[0]

        if step > 10:  # Check after some settling
            accumulated_motion += abs(linear_vel_x)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    final_vel_x = state_out.body_qd.numpy()[0][3]
    print(f"Final X Velocity: {final_vel_x:.6f}")

    if expected_motion:
        # Should be moving significantly
        assert final_vel_x > 0.1, f"Test Failed: Expected slip, but velocity is low ({final_vel_x})"
        print("SUCCESS: Box slipped as expected.")
    else:
        # Should be effectively stationary
        # There might be micro-slip due to compliance/solver tolerance, but should be very small
        assert (
            final_vel_x < 0.01
        ), f"Test Failed: Expected stick, but velocity is high ({final_vel_x})"
        print("SUCCESS: Box stuck as expected.")


def test_friction_stick():
    _run_friction_test_logic("stick")


def test_friction_slip():
    _run_friction_test_logic("slip")


if __name__ == "__main__":
    try:
        test_friction_stick()
        test_friction_slip()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
