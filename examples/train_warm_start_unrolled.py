"""Preconditioned gradient descent for raw tensor overfitting.

Uses the physics solver's own preconditioner (Jacobi = diag(A)⁻¹ for constraints,
M⁻¹ for velocities) to scale the gradient before the optimizer step.
This accounts for the vel/cf scale mismatch naturally.

Usage:
    python examples/train_warm_start_unrolled.py rendering=headless execution.use_cuda_graph=false
"""
import pathlib
import time

import hydra
import newton
import numpy as np
import torch
import warp as wp
from axion import AxionEngine
from axion import EngineConfig
from axion import LoggingConfig
from axion import SimulationConfig
from axion.core.linear_utils import compute_linear_system
from axion.core.model_builder import AxionModelBuilder
from axion.generation.scene_generator_new import SceneGenerator
from axion.learning.torch_residual_ad import AxionResidualAD
from axion.math import integrate_body_pose_kernel
from omegaconf import DictConfig

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")

WARMUP_STEPS = 10
NUM_EPOCHS = 500
SEED = 42


def build_random_model(num_worlds: int = 1, seed: int = SEED) -> newton.Model:
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.2
    builder.add_ground_plane()

    gen = SceneGenerator(builder, seed=seed)
    np.random.seed(seed)
    num_objects = np.random.randint(3, 8)
    for _ in range(num_objects):
        gen.generate_random_object(
            pos_bounds=((-1, -1, 0.3), (1, 1, 2.0)),
            density_bounds=(10.0, 100.0),
            size_bounds=(0.1, 0.3),
        )

    return builder.finalize_replicated(num_worlds=num_worlds)


def count_solver_iters(engine, dims, init_vel, init_cf):
    vel_wp = wp.from_torch(
        init_vel.detach().reshape(dims.num_worlds, dims.body_count, 6).contiguous(),
        dtype=wp.spatial_vector,
    )
    wp.copy(dest=engine.data.body_vel, src=vel_wp)

    cf_wp = wp.from_torch(init_cf.detach().contiguous())
    wp.copy(dest=engine.data._constr_force, src=cf_wp)
    wp.copy(dest=engine.data._constr_force_prev_iter, src=cf_wp)

    engine._solve()

    iters = engine.data.iter_count.numpy()[0]
    res_sq = wp.to_torch(engine.data.res_norm_sq).sum().item()
    return iters, res_sq


def compute_loss(engine, body_vel_param, constr_force_param):
    residual = AxionResidualAD.apply(
        engine.axion_model,
        engine.axion_contacts,
        engine.data,
        engine.config,
        engine.dims,
        body_vel_param,
        constr_force_param,
    )
    return torch.sum(residual**2)


def get_preconditioner_diag(engine):
    """Extract the Jacobi preconditioner diagonal (diag(A)⁻¹) as a torch tensor.

    This must be called after compute_linear_system + preconditioner.update().
    """
    return wp.to_torch(engine.preconditioner._P_inv_diag).clone()


def get_mass_inv_diag(engine):
    """Build a per-DOF inverse mass diagonal for body velocities.

    Each body has 6 DOFs (3 angular + 3 linear). We use:
    - Angular DOFs: scale by ||I⁻¹|| (Frobenius norm of inverse inertia)
    - Linear DOFs: scale by 1/m (scalar inverse mass)

    Returns tensor of shape (num_worlds, N_u).
    """
    dims = engine.dims
    # Scalar inverse mass per body: (num_worlds, num_bodies)
    inv_mass = wp.to_torch(engine.axion_model.body_inv_mass).clone()
    # Inverse inertia: (num_worlds, num_bodies, 3, 3)
    inv_inertia = wp.to_torch(engine.data.world_inv_inertia).clone()

    # Build per-DOF diagonal: (num_worlds, num_bodies, 6)
    diag = torch.zeros(dims.num_worlds, dims.body_count, 6, device=inv_mass.device)

    # Angular DOFs (0:3): use Frobenius norm of inv_inertia as scale
    inv_inertia_scale = torch.norm(inv_inertia, dim=(-2, -1))  # (num_worlds, num_bodies)
    diag[:, :, 0:3] = inv_inertia_scale.unsqueeze(-1)

    # Linear DOFs (3:6): use scalar inverse mass
    diag[:, :, 3:6] = inv_mass.unsqueeze(-1)

    return diag.reshape(dims.num_worlds, dims.N_u)


def setup_linearization(engine, body_vel, constr_force):
    """Write state into engine and run linearization to update preconditioner."""
    dims = engine.dims
    v_reshaped = body_vel.detach().reshape(dims.num_worlds, dims.body_count, 6).contiguous()
    engine.data.body_vel = wp.from_torch(v_reshaped, dtype=wp.spatial_vector, requires_grad=False)

    cf_wp = wp.from_torch(constr_force.detach().contiguous(), requires_grad=False)
    wp.copy(engine.data._constr_force, cf_wp)
    wp.copy(engine.data._constr_force_prev_iter, cf_wp)

    wp.launch(
        kernel=integrate_body_pose_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            engine.data.body_vel,
            engine.data.body_pose_prev,
            engine.axion_model.body_com,
            engine.data.dt,
        ],
        outputs=[engine.data.body_pose],
        device=engine.data.device,
    )
    compute_linear_system(
        engine.axion_model,
        engine.axion_contacts,
        engine.data,
        engine.config,
        engine.dims,
    )
    engine.preconditioner.update()


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def main(cfg: DictConfig):
    wp.init()

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    model = build_random_model(num_worlds=sim_config.num_worlds, seed=SEED)
    dt = sim_config.target_timestep_seconds

    engine = engine_config.create_engine(
        model=model,
        sim_steps=WARMUP_STEPS + 10,
        logging_config=LoggingConfig(),
    )
    dims = engine.dims

    state_cur = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_cur)

    print(f"Scene: {dims.body_count} bodies, {dims.num_constraints} constraints")

    # === Warm up physics ===
    print(f"Warming up for {WARMUP_STEPS} steps...")
    for step in range(WARMUP_STEPS):
        contacts = model.collide(state_cur)
        engine.load_data(state_cur, control, contacts, dt)
        wp.copy(dest=engine.data.body_pose, src=state_cur.body_q)
        wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
        engine.data._constr_force.zero_()
        engine.data._constr_force_prev_iter.zero_()
        engine._solve()
        wp.copy(dest=state_next.body_q, src=engine.data.body_pose)
        wp.copy(dest=state_next.body_qd, src=engine.data.body_vel)
        state_cur, state_next = state_next, state_cur

    # === Snapshot one timestep ===
    contacts = model.collide(state_cur)
    engine.load_data(state_cur, control, contacts, dt)
    saved_q = state_cur.body_q.numpy().copy()
    saved_qd = state_cur.body_qd.numpy().copy()

    # Default solver baseline
    wp.copy(dest=engine.data.body_pose, src=state_cur.body_q)
    wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()
    engine._solve()
    default_iters = engine.data.iter_count.numpy()[0]
    default_res = wp.to_torch(engine.data.res_norm_sq).sum().item()

    sol_vel = wp.to_torch(engine.data.body_vel).reshape(dims.num_worlds, dims.N_u).clone()
    sol_cf = wp.to_torch(engine.data._constr_force).clone()
    print(f"Default solver: {default_iters} iters, ||r||^2 = {default_res:.4e}")
    print(f"  sol_vel norm: {torch.norm(sol_vel).item():.4f}")
    print(f"  sol_cf norm:  {torch.norm(sol_cf).item():.4f}")

    torch_device = wp.device_to_torch(engine.data.device)

    # === Define optimizer configs ===
    configs = {
        "Adam (no precond)": {
            "precond": False,
            "lr": 1e-2,
            "cf_lr_mult": 100.0,
        },
        "Adam + physics precond": {
            "precond": True,
            "lr": 1e-2,
            "cf_lr_mult": 1.0,  # preconditioner handles scaling
        },
        "SGD + physics precond": {
            "precond": True,
            "lr": 0.1,
            "cf_lr_mult": 1.0,
            "optimizer": "sgd",
        },
        "RMSprop + physics precond": {
            "precond": True,
            "lr": 1e-2,
            "cf_lr_mult": 1.0,
            "optimizer": "rmsprop",
        },
    }

    results = {}

    for name, cfg_opt in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Fresh parameters from same init
        torch.manual_seed(SEED)
        init_vel = sol_vel + 0.5 * torch.randn_like(sol_vel)
        init_cf = sol_cf + 50.0 * torch.randn_like(sol_cf)

        body_vel_param = torch.nn.Parameter(init_vel.clone())
        constr_force_param = torch.nn.Parameter(init_cf.clone())

        lr = cfg_opt["lr"]
        cf_mult = cfg_opt["cf_lr_mult"]
        opt_type = cfg_opt.get("optimizer", "adam")

        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {"params": [body_vel_param], "lr": lr},
                    {"params": [constr_force_param], "lr": lr * cf_mult},
                ],
                momentum=0.9,
            )
        elif opt_type == "rmsprop":
            optimizer = torch.optim.RMSprop(
                [
                    {"params": [body_vel_param], "lr": lr},
                    {"params": [constr_force_param], "lr": lr * cf_mult},
                ],
            )
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": [body_vel_param], "lr": lr},
                    {"params": [constr_force_param], "lr": lr * cf_mult},
                ],
            )

        use_precond = cfg_opt["precond"]
        losses = []
        t0 = time.time()

        for epoch in range(NUM_EPOCHS):
            wp.copy(dest=state_cur.body_q, src=wp.from_numpy(saved_q, dtype=wp.transform))
            wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(saved_qd, dtype=wp.spatial_vector))
            engine.load_data(state_cur, control, contacts, dt)

            optimizer.zero_grad()
            loss = compute_loss(engine, body_vel_param, constr_force_param)
            loss.backward()

            # Apply physics preconditioner to gradients
            if use_precond and body_vel_param.grad is not None:
                # Linearize at current point to get fresh preconditioner
                setup_linearization(engine, body_vel_param, constr_force_param)

                # Precondition cf gradient: multiply by diag(A)⁻¹
                P_inv = get_preconditioner_diag(engine)
                constr_force_param.grad.data.mul_(P_inv.clamp(min=1e-10))

                # Precondition vel gradient: multiply by M⁻¹ diagonal
                M_inv = get_mass_inv_diag(engine)
                body_vel_param.grad.data.mul_(M_inv.clamp(min=1e-10))

            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
                dist_vel = torch.norm(body_vel_param.data - sol_vel).item()
                dist_cf = torch.norm(constr_force_param.data - sol_cf).item()
                grad_vel = (
                    body_vel_param.grad.norm().item() if body_vel_param.grad is not None else 0
                )
                grad_cf = (
                    constr_force_param.grad.norm().item()
                    if constr_force_param.grad is not None
                    else 0
                )
                print(
                    f"    epoch {epoch:4d} | loss: {loss_val:.4e} | "
                    f"d_vel: {dist_vel:.4f} | d_cf: {dist_cf:.4f} | "
                    f"g_vel: {grad_vel:.2e} | g_cf: {grad_cf:.2e}"
                )

        elapsed = time.time() - t0

        # Evaluate
        wp.copy(dest=state_cur.body_q, src=wp.from_numpy(saved_q, dtype=wp.transform))
        wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(saved_qd, dtype=wp.spatial_vector))
        engine.load_data(state_cur, control, contacts, dt)

        nn_iters, nn_res = count_solver_iters(engine, dims, body_vel_param, constr_force_param)
        dist_vel = torch.norm(body_vel_param.data - sol_vel).item()
        dist_cf = torch.norm(constr_force_param.data - sol_cf).item()

        results[name] = {
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "iters": nn_iters,
            "res_sq": nn_res,
            "dist_vel": dist_vel,
            "dist_cf": dist_cf,
            "time": elapsed,
        }

    # === Summary ===
    print(f"\n{'='*90}")
    print(f"  SUMMARY  (default: {default_iters} iters, ||r||^2 = {default_res:.4e})")
    print(f"{'='*90}")
    print(
        f"  {'Optimizer':<30s} {'Final loss':>12s} {'Best loss':>12s} "
        f"{'Iters':>6s} {'||r||^2':>10s} {'dist_vel':>10s} {'dist_cf':>10s} {'Time':>8s}"
    )
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for opt_name, r in results.items():
        print(
            f"  {opt_name:<30s} {r['final_loss']:12.4e} {r['best_loss']:12.4e} "
            f"{r['iters']:6d} {r['res_sq']:10.4e} {r['dist_vel']:10.4f} {r['dist_cf']:10.4f} "
            f"{r['time']:7.1f}s"
        )


if __name__ == "__main__":
    main()
