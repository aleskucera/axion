import newton
import numpy as np
import warp as wp
from axion.core.control_utils import JointMode
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_logger import EngineLogger
from axion.core.engine_logger import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


def setup_test_engine():
    # Setup Engine
    config = AxionEngineConfig(
        joint_constraint_level="pos",
        contact_constraint_level="pos",
        joint_compliance=0.0,
        max_newton_iters=20,
        max_linear_iters=50,
    )
    logger = EngineLogger(LoggingConfig(enable_timing=False))
    logger.initialize_events(steps_per_segment=1, newton_iters=config.max_newton_iters)

    return config, logger


def create_revolute_model():
    builder = AxionModelBuilder()

    # Create a link (box)
    link_1 = builder.add_link()
    builder.add_shape_box(
        link_1, hx=0.5, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=100.0)
    )

    # Joint connected to world (parent=-1)
    # Z-axis rotation
    axis = wp.vec3(0.0, 0.0, 1.0)

    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(-0.5, 0.0, 0.0), q=wp.quat_identity())

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_1,
        axis=axis,
        parent_xform=parent_local_xform,
        child_xform=child_local_xform,
    )

    builder.add_articulation([j0], key="arm")

    # Finalize model
    model = builder.finalize_replicated(num_worlds=1, gravity=0.0)
    return model, j0


def test_revolute_position_control():
    print("\n=== Testing Revolute Position Control ===")
    model, joint_id = create_revolute_model()
    config, logger = setup_test_engine()

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166  # 60Hz

    # Set Mode and Gains BEFORE engine initialization
    wp.copy(
        model.joint_dof_mode,
        wp.array(
            np.array([int(JointMode.TARGET_POSITION)], dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        ),
    )
    wp.copy(
        model.joint_target_ke,
        wp.array(np.array([1000.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )
    wp.copy(
        model.joint_target_kd,
        wp.array(np.array([100.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )

    def init_state_fn(state_in, state_out, contacts, dt):
        # Correctly initialize state_out for the first step
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, logger=logger, config=config)

    # Initialize FK
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    # Setup Control
    target_pos = np.pi / 2.0  # 90 degrees

    print(f"Target: {target_pos:.4f}")

    for step in range(100):
        # Clear external forces
        state_in.body_f.zero_()

        # Set target in control object
        wp.copy(
            control.joint_target,
            wp.array(
                np.array([target_pos], dtype=np.float32), dtype=wp.float32, device=model.device
            ),
        )

        engine.step(state_in, state_out, control, contacts, dt)

        # Swap states
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        # Update model.joint_q via IK
        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)

        current_q = model.joint_q.numpy()[0]

        if step % 10 == 0 or step == 99:
            print(f"Step {step}: q={current_q:.4f}")

    final_error = abs(current_q - target_pos)
    print(f"Final Position Error: {final_error:.4f}")

    assert final_error < 0.05, f"Position control failed to converge. Error: {final_error}"


if __name__ == "__main__":
    test_revolute_position_control()

