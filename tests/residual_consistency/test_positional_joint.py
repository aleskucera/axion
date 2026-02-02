import sys

import newton
import numpy as np
import warp as wp
from axion.constraints.positional_joint_constraint import batch_positional_joint_residual_kernel
from axion.constraints.positional_joint_constraint import (
    fused_batch_positional_joint_residual_kernel,
)
from axion.constraints.positional_joint_constraint import positional_joint_constraint_kernel
from axion.constraints.positional_joint_constraint import positional_joint_residual_kernel
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import SceneGenerator

# Adjust imports to match your folder structure

wp.init()


def get_arg(engine, name):
    """
    Helper to find an array in engine.data, engine.axion_model, or engine.model.
    """
    # 1. Try Data (Dynamic state)
    if hasattr(engine.data, name):
        val = getattr(engine.data, name)
        if val is not None:
            return val

    # 2. Try Model (Static def) - User mentioned 'axion_model'
    if hasattr(engine, "axion_model") and hasattr(engine.axion_model, name):
        val = getattr(engine.axion_model, name)
        if val is not None:
            return val

    # 3. Try generic 'model' attribute
    if hasattr(engine, "model") and hasattr(engine.model, name):
        val = getattr(engine.model, name)
        if val is not None:
            return val

    raise ValueError(f"Could not find array '{name}' in engine.data or engine.axion_model")


def print_stats(name, arr):
    if hasattr(arr, "numpy"):
        data = arr.numpy().flatten()
        print(
            f"  [{name}] Range: [{data.min():.2e}, {data.max():.2e}], Non-Zero: {np.count_nonzero(data!=0)}/{data.size}"
        )
    else:
        print(f"  [{name}] Scalar: {arr}")


def test_joint_residual_consistency():
    print("========================================================")
    print("      TESTING JOINT RESIDUAL CONSISTENCY                ")
    print("========================================================")

    # --- 1. Setup Scene with Diverse Joints ---
    builder = AxionModelBuilder()
    gen = SceneGenerator(builder, seed=42)

    # Create chains to ensure we have valid joints with non-zero transforms
    gen.generate_chain(length=2, start_pos=(0, 2, 0), joint_type="revolute")
    gen.generate_chain(length=2, start_pos=(2, 2, 0), joint_type="prismatic")

    model = builder.finalize_replicated(num_worlds=1)

    num_joints = model.joint_count
    print(f"  > Generated {num_joints} joints.")
    if num_joints == 0:
        print("  ❌ No joints generated!")
        return

    # --- 2. Engine & Inputs ---
    config = AxionEngineConfig()
    engine = AxionEngine(
        model=model,
        config=config,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
    )

    rng = np.random.default_rng(42)

    # Random Body States (Normalized Quaternions)
    q_np = rng.uniform(-1.0, 1.0, size=(engine.dims.N_w, engine.dims.N_b, 7)).astype(np.float32)
    q_np[:, :, 3:] /= np.linalg.norm(q_np[:, :, 3:], axis=2, keepdims=True)
    wp.copy(engine.data.body_q, wp.array(q_np, dtype=wp.transform, device=engine.device))

    # Random Lambda (Constraint Impulses)
    lam_np = rng.uniform(-10.0, 10.0, size=engine.data.body_lambda.j.shape).astype(np.float32)
    wp.copy(engine.data.body_lambda.j, wp.array(lam_np, device=engine.device))

    engine.data.set_dt(0.01)

    # IMPORTANT: Ensure model params are not zero
    # Some importers might leave compliance/offsets as 0 if not specified
    print("\n[Input Verification]")
    print_stats("Body Q", engine.data.body_q)
    print_stats("Body Lambda", engine.data.body_lambda.j)

    # Resolve arrays dynamically to ensure we get the right ones
    args = [
        engine.data.body_q,
        engine.data.body_lambda.j,
        engine.axion_model.body_com,
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
    ]

    # Verify critical model inputs
    print_stats("Joint X_p", args[6])
    print_stats("Joint Offsets", args[11])

    # --- 3. Reference Kernel ---
    print("\n[Running Reference Kernel]")
    ref_h_d = wp.zeros_like(engine.data.h.d_spatial)
    ref_h_j = wp.zeros_like(engine.data.h.c.j)

    # Dummies
    d_mask = wp.zeros_like(engine.data.constraint_active_mask.j)
    d_J = wp.zeros_like(engine.data.J_values.j)
    d_C = wp.zeros_like(engine.data.C_values.j)

    wp.launch(
        kernel=positional_joint_constraint_kernel,
        dim=(engine.dims.N_w, engine.dims.N_j),
        inputs=args,
        outputs=[d_mask, ref_h_d, ref_h_j, d_J, d_C],
    )

    print_stats("Ref h_d", ref_h_d)
    print_stats("Ref h_j", ref_h_j)

    # --- 4. Test Kernel ---
    print("\n[Running Test Kernel]")
    test_h_d = wp.zeros_like(ref_h_d)
    test_h_j = wp.zeros_like(ref_h_j)

    wp.launch(
        kernel=positional_joint_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_j),
        inputs=args,  # Use EXACT same args list
        outputs=[test_h_d, test_h_j],
    )

    # --- 5. Compare ---
    diff_d_arr = np.abs(ref_h_d.numpy() - test_h_d.numpy())
    diff_j_arr = np.abs(ref_h_j.numpy() - test_h_j.numpy())

    diff_d = np.max(diff_d_arr)
    diff_j = np.max(diff_j_arr)

    if diff_d < 1e-5 and diff_j < 1e-5:
        print(f"\n✅ Single World Match (Err: {diff_d:.1e}, {diff_j:.1e})")
    else:
        print(f"\n❌ Single World Fail (Err: {diff_d:.1e}, {diff_j:.1e})")
        # Find first mismatch
        idx = np.unravel_index(np.argmax(diff_j_arr), diff_j_arr.shape)
        print(f"   Mismatch at indices {idx}:")
        print(f"   Ref:  {ref_h_j.numpy()[idx]}")
        print(f"   Test: {test_h_j.numpy()[idx]}")
        print(f"   Diff: {diff_j_arr[idx]}")

    # --- 6. Batched Checks ---
    B = 4
    print(f"\n[Batched Check B={B}]")

    # Tile Inputs
    q_tiled = np.tile(q_np[np.newaxis, ...], (B, 1, 1, 1))
    body_q_batch = wp.array(q_tiled, dtype=wp.transform, device=engine.device)

    lam_tiled = np.tile(lam_np[np.newaxis, ...], (B, 1, 1))
    body_lam_batch = wp.array(lam_tiled, dtype=wp.float32, device=engine.device)

    h_d_batch = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_b), dtype=wp.spatial_vector, device=engine.device
    )
    h_j_batch = wp.zeros(
        (B, engine.dims.N_w, engine.dims.N_j), dtype=wp.float32, device=engine.device
    )

    # Prepare Batched Args (Removing single-world arrays, keeping shared arrays)
    # Args mapping:
    # 0: body_q (Batched)
    # 1: body_lambda (Batched)
    # 2..14: Shared arrays (com, type, parent, etc...)

    shared_args = args[2:]  # Everything from body_com onwards

    # Standard Batch
    wp.launch(
        kernel=batch_positional_joint_residual_kernel,
        dim=(B, engine.dims.N_w, engine.dims.N_j),
        inputs=[
            body_q_batch,
            body_lam_batch,
            engine.axion_model.body_com,
            engine.axion_model.joint_type,
            engine.axion_model.joint_parent,
            engine.axion_model.joint_child,
            engine.axion_model.joint_X_p,
            engine.axion_model.joint_X_c,
            engine.axion_model.joint_axis,
            engine.axion_model.joint_qd_start,
            engine.axion_model.joint_enabled,
            engine.data.joint_constraint_offsets,
            engine.data.dt,
            engine.config.joint_compliance,
        ],
        outputs=[h_d_batch, h_j_batch],
    )

    # Verify outputs
    b_diff_d = np.max(np.abs(h_d_batch.numpy()[0] - ref_h_d.numpy()))
    b_diff_j = np.max(np.abs(h_j_batch.numpy()[0] - ref_h_j.numpy()))

    if b_diff_d < 1e-5 and b_diff_j < 1e-5:
        print("✅ Standard Batch Match")
    else:
        print(f"❌ Standard Batch Fail: F={b_diff_d:.1e}, J={b_diff_j:.1e}")

    # Fused Batch
    h_d_fused = wp.zeros_like(h_d_batch)
    h_j_fused = wp.zeros_like(h_j_batch)

    wp.launch(
        kernel=fused_batch_positional_joint_residual_kernel,
        dim=(engine.dims.N_w, engine.dims.N_j),
        inputs=[
            body_q_batch,
            body_lam_batch,
            engine.axion_model.body_com,
            engine.axion_model.joint_type,
            engine.axion_model.joint_parent,
            engine.axion_model.joint_child,
            engine.axion_model.joint_X_p,
            engine.axion_model.joint_X_c,
            engine.axion_model.joint_axis,
            engine.axion_model.joint_qd_start,
            engine.axion_model.joint_enabled,
            engine.data.joint_constraint_offsets,
            engine.data.dt,
            engine.config.joint_compliance,
            B,
        ],
        outputs=[h_d_fused, h_j_fused],
    )

    f_diff_d = np.max(np.abs(h_d_fused.numpy()[0] - ref_h_d.numpy()))
    f_diff_j = np.max(np.abs(h_j_fused.numpy()[0] - ref_h_j.numpy()))

    if f_diff_d < 1e-5 and f_diff_j < 1e-5:
        print("✅ Fused Batch Match")
    else:
        print(f"❌ Fused Batch Fail: F={f_diff_d:.1e}, J={f_diff_j:.1e}")


if __name__ == "__main__":
    test_joint_residual_consistency()
