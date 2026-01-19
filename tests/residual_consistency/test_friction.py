import newton
import numpy as np
import warp as wp
from axion.constraints.friction_constraint import batch_friction_residual_kernel
from axion.constraints.friction_constraint import friction_constraint_kernel
from axion.constraints.friction_constraint import friction_residual_kernel
from axion.constraints.friction_constraint import fused_batch_friction_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import SceneGenerator
# Add the batched kernels to imports

wp.init()


def print_stats(name, array):
    """Helper to print min/max/mean/zeros of a Warp array."""
    np_arr = array.numpy()
    # Handle spatial vectors or multi-dim arrays by flattening
    flat = np_arr.flatten()

    min_val = np.min(flat)
    max_val = np.max(flat)
    mean_abs = np.mean(np.abs(flat))
    non_zeros = np.count_nonzero(flat)
    total = flat.size

    print(f"  > [{name}]")
    print(f"      Shape: {np_arr.shape}")
    print(f"      Range: [{min_val:.4e}, {max_val:.4e}]")
    print(f"      Mean Abs: {mean_abs:.4e}")
    print(f"      Non-Zero: {non_zeros} / {total} ({100*non_zeros/total:.1f}%)")

    if non_zeros == 0:
        print(f"      ⚠️  WARNING: Array {name} is all ZEROS!")


def test_friction_residual_consistency():
    print("========================================================")
    print("      TESTING FRICTION RESIDUAL CONSISTENCY             ")
    print("========================================================")

    # --- 1. Setup Scene ---
    builder = AxionModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    gen = SceneGenerator(builder, seed=42)

    # Generate grounded objects to ensure contacts
    for i in range(5):
        gen.generate_random_ground_touching()

    model = builder.finalize_replicated(num_worlds=1)

    # --- 2. Initialize Engine ---
    config = AxionEngineConfig(friction_compliance=0.0)
    engine = AxionEngine(
        model=model,
        config=config,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
    )

    # --- 3. Populate Random State ---
    print("\n[Step 1] Initializing Random State...")
    rng = np.random.default_rng(42)

    # Initialize Lambda (Forces)
    lambda_f = rng.uniform(-2000.0, 2000.0, size=engine.data.body_lambda.f.shape).astype(np.float32)
    lambda_f_prev = rng.uniform(-20.0, 20.0, size=engine.data.body_lambda_prev.f.shape).astype(
        np.float32
    )
    lambda_n_prev = rng.uniform(1.0, 20.0, size=engine.data.body_lambda_prev.n.shape).astype(
        np.float32
    )  # Positive Normal Force

    wp.copy(engine.data.body_lambda.f, wp.array(lambda_f, device=engine.device))
    wp.copy(engine.data.body_lambda_prev.f, wp.array(lambda_f_prev, device=engine.device))
    wp.copy(engine.data.body_lambda_prev.n, wp.array(lambda_n_prev, device=engine.device))

    # Initialize Velocity
    u_np = rng.uniform(-10.0, 10.0, size=(engine.dims.N_w, engine.dims.N_b, 6)).astype(np.float32)
    wp.copy(engine.data.body_u, wp.array(u_np, dtype=wp.spatial_vector, device=engine.device))

    engine.data.set_dt(0.01)

    # Initialize Contacts
    state_in = model.state()
    contacts = model.collide(state_in)
    engine._update_mass_matrix()
    engine._initialize_constraints(contacts)

    # Check Inputs
    print_stats("Input: Body U", engine.data.body_u)
    print_stats("Input: Lambda F", engine.data.body_lambda.f)
    print_stats("Input: Lambda N Prev", engine.data.body_lambda_prev.n)

    # Check Interaction Count
    num_contacts = engine.data.contact_interaction.shape[1]
    print(f"\n  > Active Contacts Found: {num_contacts}")
    if num_contacts == 0:
        print("  ❌ FATAL: No contacts generated. Test will produce trivial zeros.")
        return

    # --- 4. Run REFERENCE Kernel (Single World) ---
    print("\n[Step 2] Running Reference Kernel (Full Solver)...")
    ref_h_d = wp.zeros_like(engine.data.h.d_spatial)
    ref_h_f = wp.zeros_like(engine.data.h.c.f)

    # Dummy outputs
    dummy_J = wp.zeros_like(engine.data.J_values.f)
    dummy_C = wp.zeros_like(engine.data.C_values.f)
    dummy_mask = wp.zeros_like(engine.data.constraint_active_mask.f)

    wp.launch(
        kernel=friction_constraint_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            engine.data.body_q,
            engine.data.body_u,
            engine.data.body_lambda.f,
            engine.data.body_lambda_prev.f,
            engine.data.body_lambda_prev.n,
            engine.data.s_n_prev,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[dummy_mask, ref_h_d, ref_h_f, dummy_J, dummy_C],
    )

    print_stats("Reference h_d (Force)", ref_h_d)
    print_stats("Reference h_f (Constraint)", ref_h_f)

    # --- 5. Run TEST Kernel (Single World Residual) ---
    print("\n[Step 3] Running Test Kernel (Residual Only)...")
    test_h_d = wp.zeros_like(engine.data.h.d_spatial)
    test_h_f = wp.zeros_like(engine.data.h.c.f)

    wp.launch(
        kernel=friction_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            engine.data.body_q,
            engine.data.body_u,
            engine.data.body_lambda.f,
            engine.data.body_lambda_prev.f,
            engine.data.body_lambda_prev.n,
            engine.data.s_n_prev,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[test_h_d, test_h_f],
    )

    # --- 6. Compare Single World ---
    diff_d = ref_h_d.numpy() - test_h_d.numpy()
    diff_f = ref_h_f.numpy() - test_h_f.numpy()

    print(f"  > Max Diff Force:      {np.max(np.abs(diff_d)):.2e}")
    print(f"  > Max Diff Constraint: {np.max(np.abs(diff_f)):.2e}")

    if np.max(np.abs(diff_d)) < 1e-5 and np.max(np.abs(diff_f)) < 1e-5:
        print("✅ Single-World Residual Match")
    else:
        print(f"❌ Single-World Mismatch!")

    # =========================================================================
    # 7. BATCHED KERNEL CHECK
    # =========================================================================
    print("\n[Step 4] Checking Batched Kernels...")

    B = 4  # Batch size
    print(f"  > Testing with Batch Size B={B}")

    # -- A. Create Batched Inputs (Repeated Tiling) --
    u_tiled_np = np.tile(u_np[np.newaxis, ...], (B, 1, 1, 1))
    body_u_batch = wp.array(u_tiled_np, dtype=wp.spatial_vector, device=engine.device)

    lam_f_np = engine.data.body_lambda.f.numpy()
    lam_f_tiled_np = np.tile(lam_f_np[np.newaxis, ...], (B, 1, 1))
    body_lambda_f_batch = wp.array(lam_f_tiled_np, dtype=wp.float32, device=engine.device)

    # -- B. Create Batched Outputs --
    h_d_batch = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_b), dtype=wp.spatial_vector, device=engine.device
    )
    h_f_batch = wp.zeros(
        (B, engine.dims.N_w, engine.data.body_lambda.f.shape[1]),
        dtype=wp.float32,
        device=engine.device,
    )

    # -- C. Launch Standard Batched Kernel --
    wp.launch(
        kernel=batch_friction_residual_kernel,
        dim=(B, engine.dims.N_w, engine.dims.N_n),
        inputs=[
            body_u_batch,
            body_lambda_f_batch,
            engine.data.body_lambda_prev.f,  # Shared
            engine.data.body_lambda_prev.n,  # Shared
            engine.data.s_n_prev,  # Shared
            engine.data.contact_interaction,  # Shared
            engine.data.world_M_inv,  # Shared
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[h_d_batch, h_f_batch],
    )

    print_stats("Batch Output h_d (All Batches)", h_d_batch)

    # Check Slice 0 against Reference
    batch_diff_d = h_d_batch.numpy()[0] - ref_h_d.numpy()
    batch_diff_f = h_f_batch.numpy()[0] - ref_h_f.numpy()

    # Check Slice 3 (Last Batch) against Reference (Should be identical if inputs are tiled)
    batch_last_diff_d = h_d_batch.numpy()[B - 1] - ref_h_d.numpy()

    if np.max(np.abs(batch_diff_d)) < 1e-5 and np.max(np.abs(batch_last_diff_d)) < 1e-5:
        print("✅ Standard Batched Kernel Match (All Slices)")
    else:
        print(f"❌ Standard Batched Mismatch: {np.max(np.abs(batch_diff_d)):.2e}")

    # -- D. Launch Fused Batched Kernel --
    print("\n[Step 5] Checking Fused Batched Kernel...")
    h_d_fused = wp.zeros_like(h_d_batch)
    h_f_fused = wp.zeros_like(h_f_batch)

    wp.launch(
        kernel=fused_batch_friction_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),  # Note: Dim is 2D!
        inputs=[
            body_u_batch,
            body_lambda_f_batch,
            engine.data.body_lambda_prev.f,
            engine.data.body_lambda_prev.n,
            engine.data.s_n_prev,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
            B,  # num_batches
        ],
        outputs=[h_d_fused, h_f_fused],
    )

    print_stats("Fused Output h_d (All Batches)", h_d_fused)

    fused_diff_d = h_d_fused.numpy()[0] - ref_h_d.numpy()
    fused_diff_f = h_f_fused.numpy()[0] - ref_h_f.numpy()
    fused_last_diff_d = h_d_fused.numpy()[B - 1] - ref_h_d.numpy()

    if np.max(np.abs(fused_diff_d)) < 1e-5 and np.max(np.abs(fused_last_diff_d)) < 1e-5:
        print("✅ Fused Batched Kernel Match (All Slices)")
    else:
        print(f"❌ Fused Batched Mismatch: {np.max(np.abs(fused_diff_d)):.2e}")

    print("\n========================================================")
    print("      TEST COMPLETE                                     ")
    print("========================================================")


if __name__ == "__main__":
    test_friction_residual_consistency()
