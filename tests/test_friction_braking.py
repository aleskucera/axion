import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_logger import EngineLogger
from axion.core.engine_logger import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


@wp.kernel
def set_velocity_kernel(body_qd: wp.array(dtype=wp.spatial_vector), v_x: float, body_idx: int):
    tid = wp.tid()
    if tid == body_idx:
        # (vx, vy, vz, wx, wy, wz)
        body_qd[tid] = wp.spatial_vector(v_x, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_friction_braking():
    builder = AxionModelBuilder()
    mu = 0.5
    gravity = 9.81
    box_size = 1.0
    density = 100.0

    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu, restitution=0.0))

    start_height = box_size / 2.0
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, start_height), wp.quat_identity()), key="dynamic_body"
    )

    shape_cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=mu, restitution=0.0)
    builder.add_shape_box(
        body=body, hx=box_size / 2.0, hy=box_size / 2.0, hz=box_size / 2.0, cfg=shape_cfg
    )

    model = builder.finalize_replicated(num_worlds=1, gravity=-gravity)

    config = AxionEngineConfig(
        max_newton_iters=10,
        max_linear_iters=10,
        friction_compliance=0.0,
    )

    logger = EngineLogger(LoggingConfig())
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

    # Set initial velocity
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    initial_velocity = 5.0
    wp.launch(
        kernel=set_velocity_kernel,
        dim=1,
        inputs=[state_in.body_qd, initial_velocity, 0],
        device=engine.device,
    )

    expected_decel = mu * gravity
    print(f"Expected Deceleration: {expected_decel:.4f} m/s^2")

    for step in range(100):
        state_in.body_f.zero_()
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)

        # Warm start
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    final_vel_x = state_out.body_qd.numpy()[0][0]
    print(f"Final Velocity: {final_vel_x:.4f} m/s")

    # Expected final velocity is approx 5.0 - 0.5 * 9.81 * 1.0 = 0.095
    # Allow some margin, but it should be definitely less than 1.0
    assert final_vel_x < 1.0, f"Friction too weak! Final vel {final_vel_x} (Expected < 1.0)"


if __name__ == "__main__":
    test_friction_braking()
