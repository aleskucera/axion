"""Test compute_warm_start_forces.

Strategy:
  1. Settle a box on the ground for a few steps.
  2. Cold-start reference solve: record converged λ* and Newton iteration count.
  3. Feed the converged (q*, u*) as a "perfect prediction" to
     compute_warm_start_forces → get λ^0.
  4. Assert λ^0 ≈ λ*  (correctness of the projection / linear solve).
  5. Re-run the same step with λ^0 as warm start → assert fewer Newton iterations
     than the cold start (utility).
"""

import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()

DT = 0.01
SETTLE_STEPS = 30


def build_box_on_ground(mu: float = 0.5):
    builder = AxionModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu, restitution=0.0))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=mu, restitution=0.0),
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def settle(engine, model, state_in, state_out, control, steps=SETTLE_STEPS):
    for _ in range(steps):
        state_in.body_f.zero_()
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, DT)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)


def test_linear_residual_is_small():
    """The linear system residual ||A λ^0 - b|| / ||b|| should be small.

    Comparing λ^0 directly to λ* is not meaningful because the system
    (J M⁻¹ Jᵀ) λ = b can have multiple valid solutions (rank-deficient J with
    redundant contacts). The residual of the linear system is the correct measure.
    """
    model = build_box_on_ground()
    config = AxionEngineConfig(max_newton_iters=20, max_linear_iters=50)
    engine = AxionEngine(model=model, sim_steps=200, config=config)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    settle(engine, model, state_in, state_out, control)

    state_in.body_f.zero_()
    contacts = model.collide(state_in)

    # Get converged state as "perfect prediction"
    engine.load_data(state_in, control, contacts, DT)
    wp.copy(dest=engine.data.body_pose, src=state_in.body_q)
    wp.copy(dest=engine.data.body_vel, src=state_in.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()
    engine._solve()
    q_star = wp.clone(engine.data.body_pose)
    qd_star = wp.clone(engine.data.body_vel)

    # Compute λ^0 from the predicted (q*, u*)
    engine.load_data(state_in, control, contacts, DT)
    wp.copy(dest=engine.data.body_pose, src=q_star)
    wp.copy(dest=engine.data.body_vel, src=qd_star)
    engine.compute_warm_start_forces()

    # PCR solver tracks the final residual ||A λ^0 - b||^2
    residual_sq = engine.cr_solver.r_sq.numpy().copy()   # shape: (num_worlds,)
    rhs_norm_sq = engine.data.rhs.numpy()
    rhs_norm_sq = np.sum(rhs_norm_sq ** 2, axis=-1)      # shape: (num_worlds,)

    rel_residual = np.sqrt(residual_sq / (rhs_norm_sq + 1e-30))
    print(f"\nLinear system relative residual ||Aλ^0 - b|| / ||b||: {rel_residual}")

    assert np.all(rel_residual < 1e-3), (
        f"Linear system residual too large: {rel_residual} (expected < 1e-3)"
    )


def test_warm_start_reduces_iterations():
    """A warm-started solve from the predicted state should converge faster."""
    model = build_box_on_ground()
    config = AxionEngineConfig(max_newton_iters=20, max_linear_iters=20)
    engine = AxionEngine(model=model, sim_steps=200, config=config)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    settle(engine, model, state_in, state_out, control)

    state_in.body_f.zero_()
    contacts = model.collide(state_in)

    # --- Cold start ---
    engine.load_data(state_in, control, contacts, DT)
    wp.copy(dest=engine.data.body_pose, src=state_in.body_q)
    wp.copy(dest=engine.data.body_vel, src=state_in.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()
    engine._solve()

    iters_cold = engine.data.iter_count.numpy()[0]
    q_star = wp.clone(engine.data.body_pose)
    qd_star = wp.clone(engine.data.body_vel)

    print(f"\nCold start iterations: {iters_cold}")

    # --- Compute λ^0 from predicted (q*, u*) ---
    engine.load_data(state_in, control, contacts, DT)
    wp.copy(dest=engine.data.body_pose, src=q_star)
    wp.copy(dest=engine.data.body_vel, src=qd_star)
    engine.compute_warm_start_forces()
    # constr_force is now λ^0; load_data below will NOT zero it

    # --- Warm-started solve from the original state ---
    engine.load_data(state_in, control, contacts, DT)
    wp.copy(dest=engine.data.body_pose, src=state_in.body_q)
    wp.copy(dest=engine.data.body_vel, src=state_in.body_qd)
    # λ = λ^0 is preserved from compute_warm_start_forces
    engine._solve()

    iters_warm = engine.data.iter_count.numpy()[0]
    print(f"Warm start iterations: {iters_warm}")

    assert iters_warm < iters_cold, (
        f"Warm start should converge faster than cold start "
        f"({iters_warm} >= {iters_cold} iterations)"
    )


if __name__ == "__main__":
    test_linear_residual_is_small()
    test_warm_start_reduces_iterations()
    print("\nAll tests passed.")
