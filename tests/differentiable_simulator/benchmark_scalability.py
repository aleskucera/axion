"""Scalability benchmark: Helhest forward+backward time and GPU memory vs batch size."""

import sys
import time
from pathlib import Path

import warp as wp
wp.init()

import numpy as np
import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from helhest.common import HelhestConfig, create_helhest_model

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3
DT = 0.05
SIM_STEPS = 60  # 3s at 50ms


def benchmark(num_worlds, sim_steps=SIM_STEPS, warmup=3, repeats=5):
    """Run forward+backward and return median time (ms) and peak GPU memory (MB)."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane()
    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()),
        is_visible=False,
        control_mode="velocity",
        k_p=HelhestConfig.TARGET_KE,
        k_d=HelhestConfig.TARGET_KD,
        friction_left_right=0.7,
        friction_rear=0.35,
    )
    model = builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)

    config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
    )
    engine = AxionEngine(
        model=model,
        sim_steps=sim_steps,
        config=config,
        logging_config=LoggingConfig(),
        differentiable_simulation=True,
    )

    dims = engine.dims

    # Setup controls
    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = 5.0
    control = model.control()
    wp.copy(
        control.joint_target_vel,
        wp.array(
            np.tile(target_vel.reshape(1, -1), (num_worlds, 1)),
            dtype=wp.float32, device=model.device
        ),
    )

    # Random loss direction — match exact shape of body_vel_grad
    np.random.seed(42)
    grad_shape = engine.data.body_vel_grad.numpy().shape
    w = np.random.randn(*grad_shape).astype(np.float32)

    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=sim_steps, device=model.device,
    )

    def run_episode():
        state_in = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_out = model.state()

        # Forward
        for step in range(sim_steps):
            contacts = model.collide(state_in)
            engine.step(state_in, state_out, control, contacts, DT)
            buffer.save_step(step, engine.data, engine.axion_contacts)
            state_in, state_out = state_out, state_in

        # Set loss gradient
        buffer.zero_grad()
        wp.copy(
            buffer.body_vel.grad[sim_steps],
            wp.array(w, dtype=wp.spatial_vector, device=model.device),
        )

        # Backward
        for step in range(sim_steps - 1, -1, -1):
            buffer.load_step(step, engine.data, engine.axion_contacts)
            engine.data.zero_gradients()
            engine.step_backward()
            if step > 0:
                buffer.save_gradients(step, engine.data)

    # Warmup
    for _ in range(warmup):
        run_episode()
    wp.synchronize()

    # Measure peak GPU memory via Warp mempool
    device = wp.get_device("cuda:0")
    # Reset high-water mark by reading current
    _ = wp.get_mempool_used_mem_high(device)

    # Timed runs
    times = []
    for _ in range(repeats):
        wp.synchronize()
        t0 = time.perf_counter()
        run_episode()
        wp.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = np.median(times)
    peak_mb = wp.get_mempool_used_mem_high(device) / (1024 * 1024)

    return median_ms, peak_mb


print("=" * 70)
print(f"Helhest scalability: {SIM_STEPS} steps at dt={DT}s ({SIM_STEPS*DT:.1f}s horizon)")
print("=" * 70)
print(f"{'batch':>8} {'time_ms':>12} {'mem_MB':>12}")
print("-" * 35)

for batch in [1, 2, 4, 8, 16, 32, 64]:
    try:
        t_ms, mem_mb = benchmark(batch)
        print(f"{batch:8d} {t_ms:12.1f} {mem_mb:12.1f}")
    except Exception as e:
        print(f"{batch:8d}  FAILED: {e}")
        break
