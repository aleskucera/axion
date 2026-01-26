import newton
import numpy as np
import pytest
import warp as wp
import warp.autograd
from axion.constraints.positional_joint_constraint import positional_joint_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


def run_joint_test(joint_type_str):
    print(f"\n=== Testing {joint_type_str} Joint (level=pos) ===")

    builder = AxionModelBuilder()

    # Fixed base body
    parent = builder.add_link(xform=wp.transform_identity(), key="parent")

    # Dynamic child body
    child_pos = wp.vec3(0.0, 0.0, 1.0)
    child = builder.add_link(xform=wp.transform(child_pos, wp.quat_identity()), key="child")

    if joint_type_str == "Revolute":
        joint = builder.add_joint_revolute(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
        )
    elif joint_type_str == "Prismatic":
        joint = builder.add_joint_prismatic(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
        )
    elif joint_type_str == "Fixed":
        joint = builder.add_joint_fixed(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    elif joint_type_str == "Ball":
        joint = builder.add_joint_ball(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    else:
        raise ValueError(f"Unknown joint type: {joint_type_str}")

    builder.add_articulation([joint])

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    config = AxionEngineConfig(
        joint_constraint_level="pos",
        contact_constraint_level="pos",
        max_newton_iters=10,
        max_linear_iters=10,
    )

    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, config=config)

    # Detect collisions (none expected but needed for initialization)
    state_in = model.state()
    state_out = model.state()
    contacts = model.collide(state_in)

    # Initialize the internal data
    engine.data.set_dt(0.01)
    engine._initialize_variables(state_in, state_out, contacts)
    engine._initialize_constraints(contacts)
    engine._update_mass_matrix()

    # Enable gradients for differentiable inputs
    body_q = wp.clone(engine.data.body_q)
    body_q.requires_grad = True

    # Joint impulses (lambda_j)
    body_lambda_j = wp.clone(engine.data.body_lambda.j)
    body_lambda_j.fill_(1.0)
    body_lambda_j.requires_grad = True

    # Static / non-differentiable inputs
    body_com = wp.clone(engine.axion_model.body_com)
    body_com.requires_grad = False

    # Outputs
    h_j = wp.clone(engine.data.h.c.j)
    h_j.requires_grad = True

    h_d = wp.clone(engine.data.h.d_spatial)
    h_d.requires_grad = True

    passed = wp.autograd.gradcheck(
        positional_joint_residual_kernel,
        dim=(engine.dims.N_w, engine.axion_model.joint_count),
        inputs=[
            body_q,
            body_lambda_j,
            body_com,
            engine.axion_model.joint_type,
            engine.axion_model.joint_parent,
            engine.axion_model.joint_child,
            engine.axion_model.joint_X_p,
            engine.axion_model.joint_X_c,
            engine.axion_model.joint_axis,
            engine.axion_model.joint_qd_start,
            engine.axion_model.joint_enabled,
            engine.data.joint_constraint_offsets,
            engine.axion_model.joint_compliance,
            engine.data.dt,
            engine.config.joint_compliance,
        ],
        outputs=[
            h_d,
            h_j,
        ],
        plot_relative_error=False,
        plot_absolute_error=False,
        raise_exception=True,
        show_summary=True,
    )

    assert passed


@pytest.mark.parametrize("joint_type", ["Revolute", "Prismatic", "Fixed", "Ball"])
def test_joint_gradient(joint_type):
    run_joint_test(joint_type)


if __name__ == "__main__":
    try:
        run_joint_test("Revolute")
        run_joint_test("Prismatic")
        run_joint_test("Fixed")
        run_joint_test("Ball")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
