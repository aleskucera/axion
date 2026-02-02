import newton
import numpy as np
import pytest
import warp as wp
import warp.autograd
from axion.constraints.friction_constraint import friction_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


def run_friction_test(shape_type):
    print(f"\n=== Testing {shape_type} Friction (level=pos) ===")

    builder = AxionModelBuilder()

    # Ground plane
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            restitution=0.0,
        )
    )

    radius = 0.5
    box_size = 1.0

    if shape_type == "Sphere":
        start_height = radius - 0.01
    elif shape_type == "Box":
        start_height = box_size / 2.0 - 0.01
    else:
        raise ValueError("Shape type can be only 'Sphere' or  'Box'")

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, start_height), wp.quat_identity()), key="dynamic_body"
    )

    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=100.0,
        mu=0.5,
        restitution=0.0,
    )

    if shape_type == "Sphere":
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
    elif shape_type == "Box":
        builder.add_shape_box(
            body=body, hx=box_size / 2.0, hy=box_size / 2.0, hz=box_size / 2.0, cfg=shape_cfg
        )
    else:
        raise ValueError("Shape type can be only 'Sphere' or  'Box'")

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    config = AxionEngineConfig(
        joint_constraint_level="pos",
        contact_constraint_level="pos",
        max_newton_iters=10,
        max_linear_iters=10,
        contact_compliance=1e-8,
        contact_stabilization_factor=0.0,
    )

    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, config=config)

    # Detect collisions
    state_in = model.state()
    state_out = model.state()
    contacts = model.collide(state_in)

    # Initialize the internal data needed for computation of contact constraint
    engine.data.set_dt(0.01)
    engine._initialize_variables(state_in, state_out, contacts)
    engine._initialize_constraints(contacts)
    engine._update_mass_matrix()

    # Enable gradients for differentiable inputs
    body_q = wp.clone(engine.data.body_q)
    body_q.requires_grad = True

    # Velocities (important for friction)
    body_u = wp.clone(engine.data.body_u)
    body_u.fill_(wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
    body_u.requires_grad = True

    # Friction impulses (lambda_f)
    body_lambda_f = wp.clone(engine.data.body_lambda.f)
    body_lambda_f.fill_(100.0)
    body_lambda_f.requires_grad = True

    # Previous normal impulses (crucial for friction cone)
    body_lambda_n_prev = wp.clone(engine.data.body_lambda.n)
    body_lambda_n_prev.fill_(300.0)
    body_lambda_n_prev.requires_grad = False

    # Static / non-differentiable inputs
    body_u_prev = wp.clone(engine.data.body_u_prev)
    body_u_prev.requires_grad = False

    body_lambda_f_prev = wp.clone(engine.data.body_lambda_prev.f)
    body_lambda_f_prev.requires_grad = False

    s_n_prev = wp.clone(engine.data.s_n_prev)
    s_n_prev.requires_grad = False

    body_inv_inertia = wp.clone(engine.axion_model.body_inv_inertia)
    body_inv_inertia.requires_grad = True

    body_inv_mass = wp.clone(engine.axion_model.body_inv_mass)
    body_inv_mass.requires_grad = True

    # Outputs
    h_f = wp.clone(engine.data.h.c.f)
    h_f.requires_grad = True

    h_d = wp.clone(engine.data.h.d_spatial)
    h_d.requires_grad = True

    passed = wp.autograd.gradcheck(
        friction_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            body_q,
            body_u,
            body_lambda_f,
            body_lambda_f_prev,
            body_lambda_n_prev,
            s_n_prev,
            engine.data.contact_body_a,
            engine.data.contact_body_b,
            engine.data.contact_friction_coeff,
            engine.data.contact_basis_t1_a,
            engine.data.contact_basis_t2_a,
            engine.data.contact_basis_t1_b,
            engine.data.contact_basis_t2_b,
            body_inv_mass,
            body_inv_inertia,
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[
            h_d,
            h_f,
        ],
        plot_relative_error=False,
        plot_absolute_error=False,
        raise_exception=True,
        show_summary=True,
    )

    assert passed


@pytest.mark.parametrize("shape_type", ["Sphere", "Box"])
def test_friction_gradient(shape_type):
    run_friction_test(shape_type)


if __name__ == "__main__":
    try:
        run_friction_test("Sphere")
        run_friction_test("Box")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
