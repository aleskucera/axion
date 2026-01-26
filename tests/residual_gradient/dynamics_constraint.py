import numpy as np
import pytest
import warp as wp
import warp.autograd
from axion.constraints.dynamics_constraint import unconstrained_dynamics_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


def run_dynamics_gradient_test():
    print(f"\n=== Testing Dynamics Constraint Gradients ===")

    builder = AxionModelBuilder()

    # Dynamic body
    builder.add_link(xform=wp.transform(wp.vec3(0.0, 5.0, 0.0), wp.quat_identity()), key="body")

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    config = AxionEngineConfig()

    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, config=config)

    # Detect collisions (none expected but needed for initialization)
    state_in = model.state()
    state_out = model.state()
    contacts = model.collide(state_in)

    # Initialize the internal data
    dt = 0.01
    engine.data.set_dt(dt)
    engine._initialize_variables(state_in, state_out, contacts)

    # Set anisotropic inertia in the model before cloning
    engine.axion_model.body_inertia.fill_(wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0))

    # Enable gradients for differentiable inputs
    body_q = wp.clone(engine.data.body_q)
    # Set a non-identity rotation to ensure world inertia depends on q
    q_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 2.0, 3.0), 0.7)
    body_q.fill_(wp.transform(wp.vec3(0.0, 5.0, 0.0), q_rot))
    body_q.requires_grad = True

    body_u = wp.clone(engine.data.body_u)
    # Give some initial velocity to check gyroscopic terms etc
    body_u.fill_(wp.spatial_vector(1.0, 2.0, 3.0, 0.5, 0.8, 1.2))
    body_u.requires_grad = True

    body_u_prev = wp.clone(engine.data.body_u_prev)
    body_u_prev.fill_(wp.spatial_vector(0.1, 0.2, 0.3, 0.1, 0.1, 0.1))
    body_u_prev.requires_grad = True

    # Static inputs
    body_f = wp.clone(engine.data.body_f)
    body_f.fill_(wp.spatial_vector(0.5, 0.5, 0.5, 0.1, 0.1, 0.1))
    body_f.requires_grad = False

    body_mass = wp.clone(engine.axion_model.body_mass)
    body_mass.requires_grad = False

    body_inertia = wp.clone(engine.axion_model.body_inertia)
    body_inertia.requires_grad = False

    g_accel = wp.clone(engine.data.g_accel)
    g_accel.requires_grad = False

    # Outputs
    h_d = wp.zeros_like(engine.data.h.d_spatial)
    h_d.requires_grad = True

    print("Running gradcheck...")
    passed = wp.autograd.gradcheck(
        unconstrained_dynamics_kernel,
        dim=(engine.dims.N_w, engine.axion_model.body_count),
        inputs=[
            body_q,
            body_u,
            body_u_prev,
            body_f,
            body_mass,
            body_inertia,
            dt,
            g_accel,
        ],
        outputs=[
            h_d,
        ],
        plot_relative_error=False,
        plot_absolute_error=False,
        raise_exception=True,
        show_summary=True,
    )

    assert passed


def test_dynamics_gradient():
    run_dynamics_gradient_test()


if __name__ == "__main__":
    try:
        run_dynamics_gradient_test()
        print("\nDynamics gradient test PASSED")
    except Exception as e:
        print(f"\nDynamics gradient test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
