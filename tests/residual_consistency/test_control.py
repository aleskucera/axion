import numpy as np
import warp as wp
from axion.constraints.control_constraint import (
    batch_control_constraint_residual_kernel,
    control_constraint_kernel,
    control_constraint_residual_kernel,
    fused_batch_control_constraint_residual_kernel,
)
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import SceneGenerator
from axion import JointMode

wp.init()


def print_stats(name, array):
    """Helper to print min/max/mean/zeros of a Warp array."""
    np_arr = array.numpy()
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


def test_control_residual_consistency():
    print("========================================================")
    print("      TESTING CONTROL RESIDUAL CONSISTENCY              ")
    print("========================================================")

    # --- 1. Setup Scene with Controlled Joints ---
    builder = AxionModelBuilder()
    gen = SceneGenerator(builder, seed=42)

    # Create joints
    gen.generate_chain(length=2, start_pos=(0, 2, 0), joint_type="revolute")
    gen.generate_chain(length=2, start_pos=(2, 2, 0), joint_type="prismatic")

    model = builder.finalize_replicated(num_worlds=1)

    # IMPORTANT: Enable control modes BEFORE creating the engine
    # because engine calculates N_ctrl based on these modes at init time.
    num_dofs = model.joint_dof_count
    num_worlds = model.num_worlds
    
    # Enable control on all DOFs
    dof_modes = np.random.choice([JointMode.TARGET_POSITION, JointMode.TARGET_VELOCITY], size=num_dofs)
    # model.joint_dof_mode is 1D in newton.Model, AxionModel will later reshape it.
    # Actually builder.finalize_replicated returns a newton.Model where joint_dof_mode is 1D (all worlds concatenated)
    wp.copy(model.joint_dof_mode, wp.array(np.tile(dof_modes.astype(np.int32), num_worlds), device=model.device))

    # Set some gains
    ke = np.random.uniform(100.0, 1000.0, size=num_dofs).astype(np.float32)
    kd = np.random.uniform(10.0, 100.0, size=num_dofs).astype(np.float32)
    wp.copy(model.joint_target_ke, wp.array(np.tile(ke, num_worlds), device=model.device))
    wp.copy(model.joint_target_kd, wp.array(np.tile(kd, num_worlds), device=model.device))

    # --- 2. Engine & Inputs ---
    config = AxionEngineConfig()
    engine = AxionEngine(
        model=model,
        config=config,
        init_state_fn=lambda si, so, c, dt: engine.integrate_bodies(model, si, so, dt),
    )

    if engine.dims.N_ctrl == 0:
        print("❌ No control constraints found!")
        # Print joint modes to debug
        print(f"Joint modes: {engine.axion_model.joint_dof_mode.numpy()}")
        return

    rng = np.random.default_rng(42)
    N_w = engine.dims.N_w
    N_b = engine.dims.N_b
    N_ctrl = engine.dims.N_ctrl
    total_dofs_per_world = engine.axion_model.joint_dof_count

    # Random State
    q_np = rng.uniform(-1.0, 1.0, size=(N_w, N_b, 7)).astype(np.float32)
    q_np[:, :, 3:] /= np.linalg.norm(q_np[:, :, 3:], axis=2, keepdims=True)
    wp.copy(engine.data.body_q, wp.array(q_np, dtype=wp.transform, device=engine.device))

    u_np = rng.uniform(-5.0, 5.0, size=(N_w, N_b, 6)).astype(np.float32)
    wp.copy(engine.data.body_u, wp.array(u_np, dtype=wp.spatial_vector, device=engine.device))

    lam_ctrl_np = rng.uniform(-100.0, 100.0, size=(N_w, N_ctrl)).astype(np.float32)
    wp.copy(engine.data.body_lambda.ctrl, wp.array(lam_ctrl_np, device=engine.device))

    # Random Targets
    target_pos_np = rng.uniform(-1.0, 1.0, size=(N_w, total_dofs_per_world)).astype(np.float32)
    target_vel_np = rng.uniform(-1.0, 1.0, size=(N_w, total_dofs_per_world)).astype(np.float32)
    wp.copy(engine.data.joint_target_pos, wp.array(target_pos_np, device=engine.device))
    wp.copy(engine.data.joint_target_vel, wp.array(target_vel_np, device=engine.device))

    engine.data.set_dt(0.01)

    print(f"  > Generated {engine.dims.joint_count} joints, {N_ctrl} control constraints.")

    # Shared Args
    args = [
        engine.data.body_q,
        engine.data.body_u,
        engine.data.body_lambda.ctrl,
        engine.axion_model.body_com,
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
        engine.data.joint_target_pos,
        engine.data.joint_target_vel,
        engine.axion_model.joint_target_ke,
        engine.axion_model.joint_target_kd,
        engine.data.dt,
    ]

    # --- 3. Reference Kernel ---
    print("\n[Running Reference Kernel]")
    ref_h_d = wp.zeros_like(engine.data.h.d_spatial)
    ref_h_ctrl = wp.zeros_like(engine.data.h.c.ctrl)
    
    dummy_mask = wp.zeros_like(engine.data.constraint_active_mask.ctrl)
    dummy_J = wp.zeros_like(engine.data.J_values.ctrl)
    dummy_C = wp.zeros_like(engine.data.C_values.ctrl)

    wp.launch(
        kernel=control_constraint_kernel,
        dim=(N_w, engine.dims.joint_count),
        inputs=args,
        outputs=[dummy_mask, ref_h_d, ref_h_ctrl, dummy_J, dummy_C],
    )

    print_stats("Ref h_d", ref_h_d)
    print_stats("Ref h_ctrl", ref_h_ctrl)

    # --- 4. Residual Kernel ---
    print("\n[Running Residual Kernel]")
    test_h_d = wp.zeros_like(ref_h_d)
    test_h_ctrl = wp.zeros_like(ref_h_ctrl)

    wp.launch(
        kernel=control_constraint_residual_kernel,
        dim=(N_w, engine.dims.joint_count),
        inputs=args,
        outputs=[test_h_d, test_h_ctrl],
    )

    diff_d = np.max(np.abs(ref_h_d.numpy() - test_h_d.numpy()))
    diff_ctrl = np.max(np.abs(ref_h_ctrl.numpy() - test_h_ctrl.numpy()))

    if diff_d < 1e-5 and diff_ctrl < 1e-5:
        print(f"✅ Single World Match (Err: {diff_d:.1e}, {diff_ctrl:.1e})")
    else:
        print(f"❌ Single World Fail (Err: {diff_d:.1e}, {diff_ctrl:.1e})")

    # --- 5. Batched Check ---
    B = 4
    print(f"\n[Batched Check B={B}]")

    q_batch = wp.array(np.tile(q_np[np.newaxis, ...], (B, 1, 1, 1)), dtype=wp.transform, device=engine.device)
    u_batch = wp.array(np.tile(u_np[np.newaxis, ...], (B, 1, 1, 1)), dtype=wp.spatial_vector, device=engine.device)
    lam_batch = wp.array(np.tile(lam_ctrl_np[np.newaxis, ...], (B, 1, 1)), device=engine.device)

    h_d_batch = wp.zeros((B, N_w, N_b), dtype=wp.spatial_vector, device=engine.device)
    h_ctrl_batch = wp.zeros((B, N_w, N_ctrl), dtype=wp.float32, device=engine.device)

    wp.launch(
        kernel=batch_control_constraint_residual_kernel,
        dim=(B, N_w, engine.dims.joint_count),
        inputs=[
            q_batch, u_batch, lam_batch,
            engine.axion_model.body_com,
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
            engine.data.joint_target_pos,
            engine.data.joint_target_vel,
            engine.axion_model.joint_target_ke,
            engine.axion_model.joint_target_kd,
            engine.data.dt,
        ],
        outputs=[h_d_batch, h_ctrl_batch],
    )

    b_diff_d = np.max(np.abs(h_d_batch.numpy()[0] - ref_h_d.numpy()))
    b_diff_ctrl = np.max(np.abs(h_ctrl_batch.numpy()[0] - ref_h_ctrl.numpy()))

    if b_diff_d < 1e-5 and b_diff_ctrl < 1e-5:
        print("✅ Standard Batch Match")
    else:
        print(f"❌ Standard Batch Fail: D={b_diff_d:.1e}, C={b_diff_ctrl:.1e}")

    # --- 6. Fused Batch Check ---
    print(f"\n[Fused Batch Check B={B}]")
    h_d_fused = wp.zeros_like(h_d_batch)
    h_ctrl_fused = wp.zeros_like(h_ctrl_batch)

    wp.launch(
        kernel=fused_batch_control_constraint_residual_kernel,
        dim=(N_w, engine.dims.joint_count),
        inputs=[
            q_batch, u_batch, lam_batch,
            engine.axion_model.body_com,
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
            engine.data.joint_target_pos,
            engine.data.joint_target_vel,
            engine.axion_model.joint_target_ke,
            engine.axion_model.joint_target_kd,
            engine.data.dt,
            B,
        ],
        outputs=[h_d_fused, h_ctrl_fused],
    )

    f_diff_d = np.max(np.abs(h_d_fused.numpy()[0] - ref_h_d.numpy()))
    f_diff_ctrl = np.max(np.abs(h_ctrl_fused.numpy()[0] - ref_h_ctrl.numpy()))

    if f_diff_d < 1e-5 and f_diff_ctrl < 1e-5:
        print("✅ Fused Batch Match")
    else:
        print(f"❌ Fused Batch Fail: D={f_diff_d:.1e}, C={f_diff_ctrl:.1e}")


if __name__ == "__main__":
    test_control_residual_consistency()
