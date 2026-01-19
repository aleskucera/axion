import newton
import numpy as np
import warp as wp
from axion.constraints.positional_contact_constraint import batch_positional_contact_residual_kernel
from axion.constraints.positional_contact_constraint import fused_batch_positional_contact_residual_kernel
from axion.constraints.positional_contact_constraint import positional_contact_constraint_kernel
from axion.constraints.positional_contact_constraint import positional_contact_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import SceneGenerator
from axion.types import ContactInteraction
# Adjust imports to match your folder structure

wp.init()


@wp.kernel
def setup_dummy_positional_interactions(
    interactions: wp.array(dtype=ContactInteraction, ndim=2), num_bodies: int
):
    w, c = wp.tid()
    inter = interactions[w, c]

    # 1. Activate
    inter.is_active = True

    # 2. Assign Bodies
    inter.body_a_idx = c % num_bodies
    inter.body_b_idx = (c + 1) % num_bodies

    # 3. Set Geometry for Signed Distance
    # Distance = dot(n, (p_a - th_a*n) - (p_b + th_b*n))
    # We set non-zero normal and points
    normal = wp.normalize(wp.vec3(0.0, 1.0, 0.0))
    inter.basis_a.normal = wp.spatial_vector(normal, wp.vec3(0.0))  # Top 3 are linear normal

    # Points in local space (will be transformed by random body_q)
    inter.contact_point_a = wp.vec3(0.0, -0.5, 0.0)
    inter.contact_point_b = wp.vec3(0.0, 0.5, 0.0)
    inter.contact_thickness_a = 0.01
    inter.contact_thickness_b = 0.01

    interactions[w, c] = inter


def print_stats(name, array):
    np_arr = array.numpy()
    flat = np_arr.flatten()
    print(
        f"  > [{name}] Range: [{np.min(flat):.2e}, {np.max(flat):.2e}], Non-Zero: {np.count_nonzero(flat)}"
    )


def test_positional_residual_consistency():
    print("========================================================")
    print("    TESTING POSITIONAL CONTACT RESIDUAL CONSISTENCY     ")
    print("========================================================")

    # --- 1. Setup ---
    builder = AxionModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    gen = SceneGenerator(builder, seed=42)
    for _ in range(3):
        gen.generate_random_ground_touching()
    model = builder.finalize_replicated(num_worlds=1)

    config = AxionEngineConfig(contact_compliance=0.0)
    engine = AxionEngine(
        model=model,
        config=config,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
    )

    # --- 2. Random Inputs ---
    rng = np.random.default_rng(42)

    # Random Lambda N (Normal Impulse)
    lam_n = rng.uniform(-100.0, 100.0, size=engine.data.body_lambda.n.shape).astype(np.float32)
    wp.copy(engine.data.body_lambda.n, wp.array(lam_n, device=engine.device))

    # Random Body Transforms (Position + Rotation)
    # We create random positions and identity rotations for simplicity
    pos = rng.uniform(-2.0, 2.0, size=(engine.dims.N_w, engine.dims.N_b, 3))
    rot = np.zeros((engine.dims.N_w, engine.dims.N_b, 4))
    rot[:, :, 3] = 1.0  # w=1 (Identity quaternion)

    # Pack into wp.transform (7 floats: p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    q_np = np.concatenate([pos, rot], axis=2).astype(np.float32)
    wp.copy(engine.data.body_q, wp.array(q_np, dtype=wp.transform, device=engine.device))

    engine.data.set_dt(0.01)
    engine._update_mass_matrix()

    # Populate Dummy Interactions
    wp.launch(
        kernel=setup_dummy_positional_interactions,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[engine.data.contact_interaction, engine.dims.N_b],
    )

    print_stats("Lambda N", engine.data.body_lambda.n)

    # --- 3. Reference Kernel ---
    ref_h_d = wp.zeros_like(engine.data.h.d_spatial)
    ref_h_n = wp.zeros_like(engine.data.h.c.n)  # Normal part

    dummy_J = wp.zeros_like(engine.data.J_values.n)  # Note: .n for normal
    dummy_C = wp.zeros_like(engine.data.C_values.n)
    dummy_mask = wp.zeros_like(engine.data.constraint_active_mask.n)
    dummy_s = wp.zeros_like(engine.data.s_n)

    wp.launch(
        kernel=positional_contact_constraint_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            engine.data.body_q,
            engine.data.body_u,
            engine.data.body_u,  # prev (unused)
            engine.data.body_lambda.n,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            0.0,  # stabilization
            engine.config.contact_compliance,
        ],
        outputs=[dummy_mask, ref_h_d, ref_h_n, dummy_J, dummy_C, dummy_s],
    )

    print_stats("Ref h_d", ref_h_d)
    print_stats("Ref h_n", ref_h_n)

    # --- 4. Test Kernel (Residual) ---
    test_h_d = wp.zeros_like(ref_h_d)
    test_h_n = wp.zeros_like(ref_h_n)

    wp.launch(
        kernel=positional_contact_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            engine.data.body_q,
            engine.data.body_u,
            engine.data.body_u,
            engine.data.body_lambda.n,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[test_h_d, test_h_n],
    )

    # --- 5. Compare ---
    diff_d = np.max(np.abs(ref_h_d.numpy() - test_h_d.numpy()))
    diff_n = np.max(np.abs(ref_h_n.numpy() - test_h_n.numpy()))

    if diff_d < 1e-5 and diff_n < 1e-5:
        print(f"✅ Single World Match (Err: {diff_d:.1e}, {diff_n:.1e})")
    else:
        print(f"❌ Single World Fail (Err: {diff_d:.1e}, {diff_n:.1e})")

    # --- 6. Batched Check ---
    B = 4
    print(f"\n[Batched Check B={B}]")

    # Tile Inputs
    q_tiled = np.tile(q_np[np.newaxis, ...], (B, 1, 1, 1))
    body_q_batch = wp.array(q_tiled, dtype=wp.transform, device=engine.device)

    lam_n_tiled = np.tile(engine.data.body_lambda.n.numpy()[np.newaxis, ...], (B, 1, 1))
    body_lam_n_batch = wp.array(lam_n_tiled, dtype=wp.float32, device=engine.device)

    # Dummy U (unused but needed)
    u_zeros = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_b), dtype=wp.spatial_vector, device=engine.device
    )

    h_d_batch = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_b), dtype=wp.spatial_vector, device=engine.device
    )
    h_n_batch = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_n), dtype=wp.float32, device=engine.device
    )

    # Launch Standard Batch
    wp.launch(
        kernel=batch_positional_contact_residual_kernel,
        dim=(B, engine.dims.N_w, engine.dims.N_n),
        inputs=[
            body_q_batch,
            u_zeros,  # body_u (3D)
            engine.data.body_u,  # body_u_prev (2D shared) <--- FIXED
            body_lam_n_batch,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
        ],
        outputs=[h_d_batch, h_n_batch],
    )

    batch_diff = np.max(np.abs(h_d_batch.numpy()[0] - ref_h_d.numpy()))
    if batch_diff < 1e-5:
        print(f"✅ Standard Batch Match")
    else:
        print(f"❌ Standard Batch Fail: {batch_diff:.1e}")

    # Launch Fused Batch
    h_d_fused = wp.zeros_like(h_d_batch)
    h_n_fused = wp.zeros_like(h_n_batch)

    wp.launch(
        kernel=fused_batch_positional_contact_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_n),
        inputs=[
            body_q_batch,
            u_zeros,  # body_u (3D)
            engine.data.body_u,  # body_u_prev (2D shared) <--- FIXED
            body_lam_n_batch,
            engine.data.contact_interaction,
            engine.data.world_M_inv,
            engine.data.dt,
            engine.config.contact_compliance,
            B,
        ],
        outputs=[h_d_fused, h_n_fused],
    )

    fused_diff = np.max(np.abs(h_d_fused.numpy()[0] - ref_h_d.numpy()))
    if fused_diff < 1e-5:
        print(f"✅ Fused Batch Match")
    else:
        print(f"❌ Fused Batch Fail: {fused_diff:.1e}")


if __name__ == "__main__":
    test_positional_residual_consistency()
