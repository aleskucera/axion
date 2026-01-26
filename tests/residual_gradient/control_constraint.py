import newton
import numpy as np
import pytest
import warp as wp
import warp.autograd
from axion.constraints.control_constraint import control_constraint_residual_kernel
from axion.core.control_utils import JointMode
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


def run_control_test(joint_type_str, mode_str):
    print(f"\n=== Testing {joint_type_str} Control ({mode_str}) (level=pos) ===")

    builder = AxionModelBuilder()

    # Fixed base body
    parent = builder.add_link(xform=wp.transform_identity(), key="parent")

    # Dynamic child body
    child_pos = wp.vec3(0.0, 0.0, 1.0)
    child_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)
    child = builder.add_link(xform=wp.transform(child_pos, child_rot), key="child")

    mode = JointMode.NONE
    if mode_str == "Position":
        mode = JointMode.TARGET_POSITION
    elif mode_str == "Velocity":
        mode = JointMode.TARGET_VELOCITY

    if joint_type_str == "Revolute":
        joint = builder.add_joint_revolute(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=1e4,
            target_kd=1e2,
        )
    elif joint_type_str == "Prismatic":
        joint = builder.add_joint_prismatic(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=1e4,
            target_kd=1e2,
        )
    else:
        raise ValueError(f"Unknown joint type: {joint_type_str}")

    builder.add_articulation([joint])

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    # Set the control mode for the joint DOF
    # Newton stores joint DOF data. Revolute/Prismatic have 1 DOF.
    joint_dof_mode_np = np.zeros((1, model.joint_dof_count), dtype=np.int32)
    joint_dof_mode_np[0, model.joint_qd_start.numpy()[joint]] = mode
    
    # We need to pass this to the engine or set it in the builder before finalize if it supports it.
    # AxionModelBuilder adds custom attributes for this.
    # Actually, model.joint_dof_mode is a custom attribute added by AxionModelBuilder.
    # It's reachable via model.joint_dof_mode (if finalized with it) or we can set it in builder.
    
    # Let's recreate with builder setting
    builder = AxionModelBuilder()
    parent = builder.add_link(xform=wp.transform_identity(), key="parent")
    child = builder.add_link(xform=wp.transform(child_pos, wp.quat_identity()), key="child")
    
    custom_attrs = {"joint_dof_mode": [mode]}
    
    if joint_type_str == "Revolute":
        joint = builder.add_joint_revolute(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=1e4,
            target_kd=1e2,
            custom_attributes=custom_attrs
        )
    elif joint_type_str == "Prismatic":
        joint = builder.add_joint_prismatic(
            parent,
            child,
            parent_xform=wp.transform(child_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=1e4,
            target_kd=1e2,
            custom_attributes=custom_attrs
        )
    
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

    # Detect collisions
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

    body_u = wp.clone(engine.data.body_u)
    # Fill with some velocity to test velocity-dependent terms
    body_u.fill_(wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    body_u.requires_grad = True

    # Control impulses (lambda_ctrl)
    body_lambda_ctrl = wp.clone(engine.data.body_lambda.ctrl)
    body_lambda_ctrl.fill_(1.0)
    body_lambda_ctrl.requires_grad = True

    # Joint targets
    joint_target = wp.clone(engine.data.joint_target)
    joint_target.requires_grad = True

    # Static / non-differentiable inputs
    body_com = wp.clone(engine.axion_model.body_com)
    body_com.requires_grad = False

    # Outputs
    h_ctrl = wp.clone(engine.data.h.c.ctrl)
    h_ctrl.requires_grad = True

    h_d = wp.clone(engine.data.h.d_spatial)
    h_d.requires_grad = True

    passed = wp.autograd.gradcheck(
        control_constraint_residual_kernel,
        dim=(engine.dims.N_w, engine.axion_model.joint_count),
        inputs=[
            body_q,
            body_u,
            body_lambda_ctrl,
            body_com,
            engine.axion_model.joint_type,
            engine.axion_model.joint_parent,
            engine.axion_model.joint_child,
            engine.axion_model.joint_X_p,
            engine.axion_model.joint_X_c,
            engine.axion_model.joint_axis,
            engine.axion_model.joint_qd_start,
            engine.axion_model.joint_enabled,
            engine.axion_model.joint_dof_mode,
            engine.data.control_constraint_offsets,
            joint_target,
            engine.axion_model.joint_target_ke,
            engine.axion_model.joint_target_kd,
            engine.data.dt,
        ],
        outputs=[
            h_d,
            h_ctrl,
        ],
        plot_relative_error=False,
        plot_absolute_error=False,
        raise_exception=True,
        show_summary=True,
    )

    assert passed


@pytest.mark.parametrize("joint_type", ["Revolute", "Prismatic"])
@pytest.mark.parametrize("mode", ["Position", "Velocity"])
def test_control_gradient(joint_type, mode):
    run_control_test(joint_type, mode)


if __name__ == "__main__":
    try:
        run_control_test("Revolute", "Position")
        run_control_test("Revolute", "Velocity")
        run_control_test("Prismatic", "Position")
        run_control_test("Prismatic", "Velocity")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
