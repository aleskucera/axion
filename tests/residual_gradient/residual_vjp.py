import newton
import numpy as np
import pytest
import warp as wp
from axion.constraints.control_constraint import control_constraint_residual_kernel
from axion.constraints.dynamics_constraint import unconstrained_dynamics_kernel
from axion.constraints.friction_constraint import friction_residual_kernel
from axion.constraints.positional_contact_constraint import positional_contact_residual_kernel
from axion.constraints.positional_joint_constraint import positional_joint_residual_kernel
from axion.constraints.velocity_contact_constraint import velocity_contact_residual_kernel
from axion.constraints.velocity_joint_constraint import velocity_joint_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.core.model_builder import AxionModelBuilder
# Import constraint kernels

wp.init()


# -----------------------------------------------------------------------------
# 1. Residual Computation Function (From your snippet)
# -----------------------------------------------------------------------------
def compute_residual(
    model: AxionModel,
    data: EngineData,
    config: AxionEngineConfig,
    dims: EngineDimensions,
    dt: float,
):
    device = data.device
    data.h.zero_()

    # 1. Dynamics
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.N_w, dims.N_b),
        inputs=[
            data.body_q,
            data.body_u,
            data.body_u_prev,
            data.body_f,
            model.body_mass,
            model.body_inertia,
            dt,
            data.g_accel,
        ],
        outputs=[data.h.d_spatial],
        device=device,
    )

    # 2. Joint Constraints (Positional)
    if dims.N_j > 0 and config.joint_constraint_level == "pos":
        wp.launch(
            kernel=positional_joint_residual_kernel,
            dim=(dims.N_w, dims.joint_count),
            inputs=[
                data.body_q,
                data.body_lambda.j,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                model.joint_enabled,
                data.joint_constraint_offsets,
                model.joint_compliance,
                dt,
                config.joint_compliance,
            ],
            outputs=[data.h.d_spatial, data.h.c.j],
            device=device,
        )

    # 3. Contact Constraints (Positional)
    if dims.N_n > 0 and config.contact_constraint_level == "pos":
        wp.launch(
            kernel=positional_contact_residual_kernel,
            dim=(dims.N_w, dims.N_n),
            inputs=[
                data.body_q,
                data.body_u,
                data.body_u_prev,
                data.body_lambda.n,
                data.contact_body_a,
                data.contact_body_b,
                data.contact_point_a,
                data.contact_point_b,
                data.contact_thickness_a,
                data.contact_thickness_b,
                data.contact_dist,
                data.contact_basis_n_a,
                data.contact_basis_n_b,
                model.body_inv_mass,
                model.body_inv_inertia,
                dt,
                config.contact_compliance,
            ],
            outputs=[data.h.d_spatial, data.h.c.n],
            device=device,
        )


# -----------------------------------------------------------------------------
# 2. Test Setup
# -----------------------------------------------------------------------------
def setup_simulation():
    """Sets up a simple Sphere-Ground scene."""
    builder = AxionModelBuilder()

    # Ground
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0))

    # Sphere
    radius = 0.5
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, radius * 0.9), wp.quat_identity()), key="dynamic_body"
    )
    builder.add_shape_sphere(
        body=body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.0)
    )

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    config = AxionEngineConfig(
        joint_constraint_level="pos", contact_constraint_level="pos", contact_compliance=1e-5
    )

    # Dummy init function
    def init_fn(state_in, state_out, contacts, dt):
        pass

    engine = AxionEngine(model=model, init_state_fn=init_fn, config=config)

    # Run one collision detection step to populate contact data
    state_in = model.state()
    contacts = model.collide(state_in)
    engine.data.set_dt(0.01)
    engine._initialize_variables(state_in, state_in, contacts)  # u=0
    engine._initialize_constraints(contacts)
    engine._update_mass_matrix()

    return engine


# -----------------------------------------------------------------------------
# 3. The VJP Test
# -----------------------------------------------------------------------------
def test_residual_vjp_manual():
    print("\n=== Testing Manual VJP (Tape Injection) ===")

    engine = setup_simulation()
    data = engine.data
    model = engine.axion_model
    dt = 0.01

    # --- A. Enable Gradients on Parameters to Test ---
    # We will test the gradient w.r.t 'body_q' (positions) and 'body_inv_mass' (parameter)

    # 1. Clone inputs to ensure they are tracked
    body_q_in = wp.clone(data.body_q)
    body_q_in.requires_grad = True

    # 2. Ensure parameter has grad enabled
    inv_mass_in = wp.clone(model.body_inv_mass)
    inv_mass_in.requires_grad = True

    # Inject these tracked variables back into data/model for the function call
    data.body_q = body_q_in
    model.body_inv_mass = inv_mass_in

    # --- Ensure outputs have gradients enabled for manual injection ---
    if data._h_spatial is not None:
        data._h_spatial.requires_grad = True
    data._h.requires_grad = True

    # --- B. Create Random "Adjoint" Vectors 'w' ---
    # We need a random vector for every output of the residual function.
    # Outputs are: h.d_spatial (Spatial Vector) AND h.c.n (Float)

    rng = np.random.default_rng(42)

    # Random w for dynamics residual (N_w, N_b, 6)
    w_d_np = rng.standard_normal(size=(data.dims.N_w, data.dims.N_b, 6)).astype(np.float32)
    w_d = wp.from_numpy(w_d_np, dtype=wp.spatial_vector, device=data.device)

    # Random w for contact residual (N_w, N_c) (Note: _h stores all constraints)
    w_c_np = rng.standard_normal(size=(data.dims.N_w, data.dims.N_u + data.dims.N_c)).astype(
        np.float32
    )
    w_c = wp.from_numpy(w_c_np, dtype=wp.float32, device=data.device)

    # --- C. Run AD (Automatic Differentiation) with Tape Injection ---
    tape = wp.Tape()
    with tape:
        compute_residual(model, data, engine.config, engine.dims, dt)

    # INJECTION TRICK:
    # 1. Dynamics output is in `data._h_spatial`
    # 2. Constraint output is in `data._h` (specifically the contact slice)

    # We assume w_d and w_c are the "adjoint solutions" coming from the linear solver.
    # We populate the gradients manually.

    if data._h_spatial is not None:
        wp.copy(dest=data._h_spatial.grad, src=w_d)

    wp.copy(dest=data._h.grad, src=w_c)

    # Backpropagate!
    tape.backward()

    # Capture AD Gradients
    grad_ad_q = body_q_in.grad.numpy().copy()
    grad_ad_m = inv_mass_in.grad.numpy().copy()

    print("AD Backward Pass Complete.")

    # --- D. Run Finite Differences (FD) to Verify ---
    # We treat L = dot(w_d, h_d) + dot(w_c, h_c) as a scalar loss function.
    # We verify that grad_ad == gradient(L)

    epsilon = 1e-4

    def compute_scalar_loss(q_array, m_array):
        # Helper to compute w . R for perturbed inputs

        # Backup original data
        orig_q = data.body_q
        orig_m = model.body_inv_mass

        # Assign perturbed
        data.body_q = q_array
        model.body_inv_mass = m_array

        # Run
        compute_residual(model, data, engine.config, engine.dims, dt)

        # Compute dot product
        # dot_spatial
        h_d_np = data.h.d_spatial.numpy()  # (N_w, N_b, 6)
        dot_d = np.sum(h_d_np * w_d_np)

        # dot_constraint
        # Note: data.h.c.n is a View, let's just dot the whole _h buffer for simplicity
        # since compute_residual zeros it out anyway.
        h_c_np = data._h.numpy()
        dot_c = np.sum(h_c_np * w_c_np)

        # Restore
        data.body_q = orig_q
        model.body_inv_mass = orig_m

        return dot_d + dot_c

    print("\nVerifying 'body_inv_mass' (Scalar Parameter)...")
    # FD for Mass
    val_orig = inv_mass_in.numpy().copy()

    # Perturb +
    val_plus = val_orig.copy()
    val_plus[0, 0] += epsilon
    loss_plus = compute_scalar_loss(
        body_q_in, wp.array(val_plus, dtype=wp.float32, device=data.device)
    )

    # Perturb -
    val_minus = val_orig.copy()
    val_minus[0, 0] -= epsilon
    loss_minus = compute_scalar_loss(
        body_q_in, wp.array(val_minus, dtype=wp.float32, device=data.device)
    )

    grad_fd_m = (loss_plus - loss_minus) / (2 * epsilon)

    print(f"AD Grad: {grad_ad_m[0,0]:.6f}")
    print(f"FD Grad: {grad_fd_m:.6f}")

    assert np.isclose(grad_ad_m[0, 0], grad_fd_m, rtol=1e-3, atol=1e-3), "Mass gradient mismatch!"
    print("✅ Mass Gradient Matches!")

    print("\nVerifying 'body_q' (Transform Parameter)...")
    # Test one component (e.g., Z position of body 0)
    q_orig_np = body_q_in.numpy().copy()  # (N_w, N_b, 7)

    # Perturb Z pos (index 2)
    q_plus = q_orig_np.copy()
    q_plus[0, 0, 2] += epsilon
    loss_plus = compute_scalar_loss(
        wp.array(q_plus, dtype=wp.transform, device=data.device), inv_mass_in
    )

    q_minus = q_orig_np.copy()
    q_minus[0, 0, 2] -= epsilon
    loss_minus = compute_scalar_loss(
        wp.array(q_minus, dtype=wp.transform, device=data.device), inv_mass_in
    )

    grad_fd_q_z = (loss_plus - loss_minus) / (2 * epsilon)

    # Note: Warp transform gradient is stored as [p_x, p_y, p_z, r_x, r_y, r_z] (spatial vector)??
    # OR [p_x, p_y, p_z, q_x, q_y, q_z, q_w]?
    # WP 1.0+: wp.transform gradient is usually 7 floats (pos + quat).
    # Let's check the shape.
    print(f"Grad Shape: {grad_ad_q.shape}")

    # Assuming standard layout [px, py, pz, qx, qy, qz, qw]
    grad_ad_q_z = grad_ad_q[0, 0, 2]

    print(f"AD Grad (Z-pos): {grad_ad_q_z:.6f}")
    print(f"FD Grad (Z-pos): {grad_fd_q_z:.6f}")

    assert np.isclose(grad_ad_q_z, grad_fd_q_z, rtol=1e-2, atol=1e-2), "Position gradient mismatch!"
    print("✅ Position Gradient Matches!")


if __name__ == "__main__":
    test_residual_vjp_manual()
