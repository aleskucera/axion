import numpy as np
import warp as wp
from axion.constraints.dynamics_constraint import (
    batch_unconstrained_dynamics_kernel,
    fused_batch_unconstrained_dynamics_kernel,
    unconstrained_dynamics_kernel,
)
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import SceneGenerator
from axion.types import SpatialInertia, world_spatial_inertia_kernel

wp.init()


def print_stats(name, array):
    """Helper to print min/max/mean/zeros of a Warp array."""
    np_arr = array.numpy()
    # Handle structured arrays (like SpatialInertia)
    if np_arr.dtype.names:
        print(f"  > [{name}]")
        print(f"      Shape: {np_arr.shape}")
        for field in np_arr.dtype.names:
            flat_field = np_arr[field].flatten()
            min_val = np.min(flat_field)
            max_val = np.max(flat_field)
            print(f"      Field '{field}': Range: [{min_val:.4e}, {max_val:.4e}]")
        return

    flat = np_arr.flatten()



def test_dynamics_residual_consistency():
    print("========================================================")
    print("      TESTING DYNAMICS RESIDUAL CONSISTENCY             ")
    print("========================================================")

    # --- 1. Setup Scene ---
    builder = AxionModelBuilder()
    gen = SceneGenerator(builder, seed=42)
    # Generate some bodies
    gen.generate_chain(length=3, start_pos=(0, 5, 0))
    gen.generate_chain(length=3, start_pos=(2, 5, 0))

    model = builder.finalize_replicated(num_worlds=1)

    # --- 2. Engine & Inputs ---
    config = AxionEngineConfig()
    engine = AxionEngine(
        model=model,
        config=config,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
    )

    rng = np.random.default_rng(42)
    N_w = engine.dims.N_w
    N_b = engine.dims.N_b

    # Random Body States
    # body_q
    q_np = rng.uniform(-1.0, 1.0, size=(N_w, N_b, 7)).astype(np.float32)
    # Normalize quaternions
    q_np[:, :, 3:] /= np.linalg.norm(q_np[:, :, 3:], axis=2, keepdims=True)
    wp.copy(engine.data.body_q, wp.array(q_np, dtype=wp.transform, device=engine.device))

    # body_u and body_u_prev
    u_np = rng.uniform(-5.0, 5.0, size=(N_w, N_b, 6)).astype(np.float32)
    u_prev_np = rng.uniform(-5.0, 5.0, size=(N_w, N_b, 6)).astype(np.float32)
    wp.copy(engine.data.body_u, wp.array(u_np, dtype=wp.spatial_vector, device=engine.device))
    wp.copy(engine.data.body_u_prev, wp.array(u_prev_np, dtype=wp.spatial_vector, device=engine.device))

    # body_f
    f_np = rng.uniform(-10.0, 10.0, size=(N_w, N_b, 6)).astype(np.float32)
    wp.copy(engine.data.body_f, wp.array(f_np, dtype=wp.spatial_vector, device=engine.device))

    engine.data.set_dt(0.01)

    # Update world_M based on body_q
    wp.launch(
        kernel=world_spatial_inertia_kernel,
        dim=(N_w, N_b),
        inputs=[
            engine.data.body_q,
            engine.axion_model.body_mass,
            engine.axion_model.body_inertia,
        ],
        outputs=[engine.data.world_M],
    )

    print("\n[Input Verification]")
    print_stats("Body Q", engine.data.body_q)
    print_stats("Body U", engine.data.body_u)
    print_stats("Body U Prev", engine.data.body_u_prev)
    print_stats("Body F", engine.data.body_f)
    print_stats("World M", engine.data.world_M)

    # --- 3. Reference Kernel (Single World) ---
    print("\n[Running Reference Kernel]")
    ref_h_d = wp.zeros((N_w, N_b), dtype=wp.spatial_vector, device=engine.device)

    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(N_w, N_b),
        inputs=[
            engine.data.body_q,
            engine.data.body_u,
            engine.data.body_u_prev,
            engine.data.body_f,
            engine.axion_model.body_mass,
            engine.axion_model.body_inertia,
            engine.data.dt,
            engine.data.g_accel,
        ],
        outputs=[ref_h_d],
    )

    print_stats("Ref h_d", ref_h_d)

    # --- 4. Batched Check ---
    B = 4
    print(f"\n[Batched Check B={B}]")

    # Tile Inputs
    q_batch_np = np.tile(q_np[np.newaxis, ...], (B, 1, 1, 1))
    u_batch_np = np.tile(u_np[np.newaxis, ...], (B, 1, 1, 1))

    batch_body_q = wp.array(q_batch_np, dtype=wp.transform, device=engine.device)
    batch_body_u = wp.array(u_batch_np, dtype=wp.spatial_vector, device=engine.device)

    batch_h_d = wp.zeros((B, N_w, N_b), dtype=wp.spatial_vector, device=engine.device)

    wp.launch(
        kernel=batch_unconstrained_dynamics_kernel,
        dim=(B, N_w, N_b),
        inputs=[
            batch_body_q,
            batch_body_u,
            engine.data.body_u_prev,
            engine.data.body_f,
            engine.axion_model.body_mass,
            engine.axion_model.body_inertia,
            engine.data.dt,
            engine.data.g_accel,
        ],
        outputs=[batch_h_d],
    )

    batch_diff = np.max(np.abs(batch_h_d.numpy()[0] - ref_h_d.numpy()))
    if batch_diff < 1e-5:
        print(f"✅ Standard Batch Match (Err: {batch_diff:.1e})")
    else:
        print(f"❌ Standard Batch Fail (Err: {batch_diff:.1e})")

    # --- 5. Fused Batch Check ---
    print(f"\n[Fused Batch Check B={B}]")
    fused_h_d = wp.zeros((B, N_w, N_b), dtype=wp.spatial_vector, device=engine.device)

    wp.launch(
        kernel=fused_batch_unconstrained_dynamics_kernel,
        dim=(N_w, N_b),
        inputs=[
            batch_body_q,
            batch_body_u,
            engine.data.body_u_prev,
            engine.data.body_f,
            engine.axion_model.body_mass,
            engine.axion_model.body_inertia,
            engine.data.dt,
            engine.data.g_accel,
            B,
        ],
        outputs=[fused_h_d],
    )

    fused_diff = np.max(np.abs(fused_h_d.numpy()[0] - ref_h_d.numpy()))
    if fused_diff < 1e-5:
        print(f"✅ Fused Batch Match (Err: {fused_diff:.1e})")
    else:
        print(f"❌ Fused Batch Fail (Err: {fused_diff:.1e})")


if __name__ == "__main__":
    test_dynamics_residual_consistency()
