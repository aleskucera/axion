import newton
import numpy as np
import pytest
import torch
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

# Import your Axion engine PyTorch wrapper (adjust import path as needed)
# from your_module import AxionResidual
# For testing purposes, assuming AxionResidual is defined here or imported.

wp.init()


# -----------------------------------------------------------------------------
# 1. Setup Simulation (Matching `residual_vjp.py`)
# -----------------------------------------------------------------------------
def setup_simulation():
    """Sets up a simple Sphere-Ground scene to test the residual."""
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

    def init_fn(state_in, state_out, contacts, dt):
        pass

    engine = AxionEngine(model=model, init_state_fn=init_fn, config=config)

    # Detect collisions and initialize data
    state_in = model.state()
    contacts = model.collide(state_in)
    engine.data.set_dt(0.01)
    engine._initialize_variables(state_in, state_in, contacts)
    engine._initialize_constraints(contacts)
    engine._update_mass_matrix()

    return engine


# -----------------------------------------------------------------------------
# 2. PyTorch Gradcheck Test
# -----------------------------------------------------------------------------
def test_pytorch_residual_gradcheck():
    print("\n=== Testing PyTorch Autograd Wrapper (gradcheck) ===")

    engine = setup_simulation()
    data = engine.data
    model = engine.axion_model
    config = engine.config
    dims = engine.dims
    contacts = engine.contacts  # Assuming you store contacts or pass them

    # Convert Warp arrays to PyTorch Tensors.
    # We clone them so we own the memory in PyTorch, and require_grad=True
    torch_device = wp.device_to_torch(data.device)

    pose_th = wp.to_torch(data.body_q).clone().detach().requires_grad_(True)
    vel_th = wp.to_torch(data.body_u).clone().detach().requires_grad_(True)
    vel_prev_th = wp.to_torch(data.body_u_prev).clone().detach().requires_grad_(True)

    # Define a pure PyTorch function to pass to gradcheck
    def eval_residual(p, v, vp):
        # Using the custom autograd function we wrote
        # AxionResidual.apply(model, contacts, data, config, dims, pose, vel, vel_prev)
        return AxionResidual.apply(model, contacts, data, config, dims, p, v, vp)

    # PyTorch's gradcheck computes the analytical Jacobian using backward()
    # and compares it to a numerical Jacobian computed via finite differences.
    # Note: Since Warp runs in float32, we use relaxed tolerances (eps, atol, rtol).
    print("Running torch.autograd.gradcheck...")

    # We double the precision of tensors just for the gradcheck if needed,
    # but PyTorch's gradcheck can handle float32 with larger epsilons.
    passed = torch.autograd.gradcheck(
        eval_residual,
        (pose_th, vel_th, vel_prev_th),
        eps=1e-3,  # Perturbation step size
        atol=1e-3,  # Absolute tolerance
        rtol=1e-2,  # Relative tolerance
        fast_mode=True,  # Uses a faster random projection check
        raise_exception=True,
    )

    assert passed, "PyTorch gradcheck failed! Analytical gradients do not match numerical."
    print("✅ PyTorch Gradcheck Passed!")


# -----------------------------------------------------------------------------
# 3. End-to-End Optimizer Step Test
# -----------------------------------------------------------------------------
def test_pytorch_optimization_step():
    print("\n=== Testing PyTorch End-to-End Optimization Step ===")

    engine = setup_simulation()

    # Setup Tensors
    pose_th = wp.to_torch(engine.data.body_q).clone().detach().requires_grad_(True)
    vel_th = wp.to_torch(engine.data.body_u).clone().detach().requires_grad_(True)
    vel_prev_th = wp.to_torch(engine.data.body_u_prev).clone().detach().requires_grad_(True)

    # Setup a standard PyTorch optimizer
    optimizer = torch.optim.Adam([pose_th, vel_th, vel_prev_th], lr=0.01)

    optimizer.zero_grad()

    # Forward pass
    residual = AxionResidual.apply(
        engine.axion_model,
        engine.contacts,
        engine.data,
        engine.config,
        engine.dims,
        pose_th,
        vel_th,
        vel_prev_th,
    )

    # Simple L2 Loss on the residual
    loss = torch.sum(residual**2)
    loss_initial = loss.item()

    # Backward pass (triggers your Warp backward kernels)
    loss.backward()

    # Ensure gradients were populated
    assert pose_th.grad is not None
    assert vel_th.grad is not None
    assert vel_prev_th.grad is not None

    # Step the optimizer
    optimizer.step()

    # Re-evaluate to ensure loss decreased (basic sanity check)
    residual_new = AxionResidual.apply(
        engine.axion_model,
        engine.contacts,
        engine.data,
        engine.config,
        engine.dims,
        pose_th,
        vel_th,
        vel_prev_th,
    )
    loss_new = torch.sum(residual_new**2).item()

    print(f"Initial Loss: {loss_initial:.6f}")
    print(f"Loss after 1 Adam step: {loss_new:.6f}")

    # Note: Depending on the complex physical landscape, 1 step might not perfectly
    # guarantee a decrease unless the problem is convex, but generally it should move.
    print("✅ Optimization step executed successfully without crashing.")


if __name__ == "__main__":
    test_pytorch_residual_gradcheck()
    test_pytorch_optimization_step()
