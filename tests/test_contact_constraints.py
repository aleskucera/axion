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
def apply_downward_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1), force_mag: float
):
    body_idx = wp.tid()
    if body_idx == 0:
        # Spatial vector layout: (moment_x, moment_y, moment_z, force_x, force_y, force_z)
        body_f[body_idx] = body_f[body_idx] + wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, -force_mag)


def run_contact_test(shape_type):
    print(f"\n=== Testing {shape_type} Contact Penetration (mu=0.0, level=pos) ===")

    builder = AxionModelBuilder()

    # Ground plane
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            restitution=0.0,
        )
    )

    radius = 0.5
    box_size = 1.0

    if shape_type == "Sphere":
        start_height = radius
    elif shape_type == "Box":
        start_height = box_size / 2.0
    else:
        start_height = 1.0

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, start_height), wp.quat_identity()), key="dynamic_body"
    )

    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=100.0,
        mu=0.0,
        restitution=0.0,
    )

    if shape_type == "Sphere":
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        expected_z = radius
    elif shape_type == "Box":
        builder.add_shape_box(
            body=body, hx=box_size / 2.0, hy=box_size / 2.0, hz=box_size / 2.0, cfg=shape_cfg
        )
        expected_z = box_size / 2.0

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    config = AxionEngineConfig(
        max_newton_iters=20,
        max_linear_iters=20,
    )

    logger = EngineLogger(LoggingConfig())
    logger.initialize_events(steps_per_segment=1, newton_iters=config.max_newton_iters)

    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, logger=logger, config=config)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 0.01

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    # Force to see penetration
    applied_force = 1e5

    print(f"Applying downward force: {applied_force}")

    for step in range(50):
        state_in.body_f.zero_()

        wp.launch(
            kernel=apply_downward_force_kernel,
            dim=1,
            inputs=[state_in.body_f, applied_force],
            device=engine.device,
        )

        contacts = model.collide(state_in)
        logger.set_current_step_in_segment(0)
        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    final_z = state_out.body_q.numpy()[0][2]
    signed_distance = final_z - expected_z

    print(f"Final Z: {final_z:.6f}, Expected Z (surface): {expected_z:.6f}")
    print(f"Signed Distance: {signed_distance:.6f}")

    # Check penetration is positive (sinking)
    assert np.abs(
        signed_distance < 1e-3
    ), f"{shape_type} didn't satisfy the contact constraint! Signed Distance={signed_distance}"

    print(f"SUCCESS: {shape_type} contact penetration test satisfied.")


@pytest.mark.parametrize("shape_type", ["Sphere", "Box"])
def test_contact_penetration(shape_type):
    run_contact_test(shape_type)


if __name__ == "__main__":
    try:
        run_contact_test("Sphere")
        run_contact_test("Box")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
