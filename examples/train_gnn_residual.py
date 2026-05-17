"""GNN online training with residual loss across multiple random scenes.

Each epoch is an independent episode:
  1. A fresh random scene is built with SceneGenerator (different objects every epoch).
  2. The Axion engine is created for that scene.
  3. The GNN is trained for `steps_per_episode` physics steps using the residual loss.
  4. The engine is discarded; the GNN weights and optimizer state carry over.

This gives the GNN exposure to diverse topologies (different object counts, shapes,
sizes, joint types) while the graph-based architecture handles the varying structure
naturally.

Usage
-----
    # Default: random multi-object scenes, 8 parallel worlds
    python train_gnn_residual.py

    # More worlds for larger batches, more objects per scene, warm-started forces
    python train_gnn_residual.py --num_worlds 32 --num_objects 3 --warm_start_constr_forces

    # Resume from a checkpoint
    python train_gnn_residual.py --checkpoint data/gnn_data/residual/gnn_epoch_0100.pt

Saved models are compatible with GNNEngineConfig (loadable with torch.load for
direct inference inside GNNEngine).
"""

import argparse
import pathlib
import random

import newton
import torch
import warp as wp
from tqdm import tqdm

from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation.scene_generator_new import SceneGenerator
from axion.gnn import AxionGNN
from axion.gnn.residual_trainer import GNNResidualTrainer
from axion.simulation.dataset_simulator import random_velocities_kernel

# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def build_random_scene(
    num_worlds: int,
    num_objects: int,
    seed: int,
    pos_bounds: tuple = ((-1.0, -1.0, 0.2), (1.0, 1.0, 2.0)),
    size_bounds: tuple = (0.1, 0.3),
    density_bounds: tuple = (10.0, 100.0),
) -> newton.Model:
    """Random objects above a ground plane, replicated across worlds."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.5
    builder.add_ground_plane()

    gen = SceneGenerator(builder, seed=seed)
    for _ in range(num_objects):
        gen.generate_random_object(
            pos_bounds=pos_bounds,
            density_bounds=density_bounds,
            size_bounds=size_bounds,
        )

    return builder.finalize_replicated(num_worlds=num_worlds)


def build_ball_scene(num_worlds: int, **_) -> newton.Model:
    """Single ball bouncing above a ground plane (fixed topology)."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.5
    ball = builder.add_body(xform=wp.transform((0.0, 0.0, 1.5), wp.quat_identity()), label="ball")
    builder.add_shape_sphere(
        body=ball,
        radius=1.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=10.0, ke=6000.0, kd=1000.0, kf=200.0, mu=0.4, restitution=0.0
        ),
    )
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(ke=6000.0, kd=1000.0, kf=200.0, mu=0.4, restitution=0.0)
    )
    return builder.finalize_replicated(num_worlds=num_worlds)


SCENE_BUILDERS = {
    "random": build_random_scene,
    "ball": build_ball_scene,
}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def run_training_step(
    engine: AxionEngine,
    trainer: GNNResidualTrainer,
    current_state,
    control,
    dt: float,
) -> float:
    """One physics timestep with an embedded GNN training step.

    Sequence: load_data → train_step (residual loss) → reset initial guess → _solve
    """
    contacts = engine.model.collide(current_state)
    engine.load_data(current_state, control, contacts, dt)

    loss = trainer.train_step()

    # Reset initial guess for the Newton solver (same as AxionEngine.step)
    wp.copy(dest=engine.data.body_pose, src=current_state.body_q)
    wp.copy(dest=engine.data.body_vel, src=current_state.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()

    engine._solve()

    wp.copy(dest=current_state.body_q, src=engine.data.body_pose)
    wp.copy(dest=current_state.body_qd, src=engine.data.body_vel)

    return loss


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def run_episode(
    gnn: AxionGNN,
    trainer: GNNResidualTrainer,
    engine: AxionEngine,
    steps: int,
    dt: float,
    vel_bound: float,
) -> float:
    """Run one training episode on the given engine.

    Randomises initial body velocities so that each episode starts from a
    different dynamic state even when the scene topology is the same.

    Returns:
        Average residual loss over all steps.
    """
    model = engine.model
    current_state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, current_state)

    # Randomise initial velocities (angular and linear)
    wp.launch(
        kernel=random_velocities_kernel,
        dim=current_state.body_qd.shape[0],
        inputs=[
            current_state.body_qd,
            -vel_bound,
            vel_bound,  # linear bounds
            -vel_bound,
            vel_bound,  # angular bounds
            random.randint(0, 2**31 - 1),
        ],
        device=model.device,
    )

    total_loss = 0.0
    for _ in range(steps):
        total_loss += run_training_step(engine, trainer, current_state, control, dt)

    return total_loss / steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running on {device_str}  scene={args.scene}  "
        f"num_worlds={args.num_worlds}  num_objects={args.num_objects}"
    )

    wp.init()

    engine_config = AxionEngineConfig(max_newton_iters=8, max_linear_iters=16)
    build_scene = SCENE_BUILDERS[args.scene]

    # --- GNN (weights persist across all episodes) ---
    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else None
    if checkpoint and checkpoint.exists():
        print(f"Resuming from checkpoint: {checkpoint}")
        gnn = torch.load(checkpoint, map_location=device_str, weights_only=False)
    else:
        gnn = AxionGNN(
            message_passes=args.message_passes,
            hidden_dims=args.hidden_dims,
            hidden_layers=args.hidden_layers,
            normalize=True,
        )
    gnn.to(device_str)

    # Build a placeholder engine just to construct the trainer (and its optimizer).
    # set_engine() will replace the references at the start of each episode.
    init_model = build_scene(args.num_worlds, num_objects=args.num_objects, seed=0)
    init_engine = AxionEngine(init_model, sim_steps=args.steps_per_episode, config=engine_config)

    trainer = GNNResidualTrainer(
        net=gnn,
        axion_model=init_engine.axion_model,
        axion_contacts=init_engine.axion_contacts,
        data=init_engine.data,
        config=init_engine.config,
        dims=init_engine.dims,
        lr=args.lr,
        engine=init_engine,
        warm_start_constr_forces=args.warm_start_constr_forces,
    )

    # --- Optional wandb ---
    use_wandb = args.wandb is not None
    if use_wandb:
        import wandb

        wandb.init(project="axion-gnn-residual", name=args.wandb, config=vars(args))

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Episode loop ---
    for epoch in tqdm(range(args.epochs), desc="Episodes"):
        # Fresh scene every epoch (different seed → different topology/size/density)
        scene_seed = args.base_seed + epoch
        model = build_scene(
            args.num_worlds,
            num_objects=args.num_objects,
            seed=scene_seed,
        )
        engine = AxionEngine(model, sim_steps=args.steps_per_episode, config=engine_config)

        # Point the trainer at the new engine (optimizer state is preserved)
        trainer.set_engine(engine)

        avg_loss = run_episode(
            gnn,
            trainer,
            engine,
            steps=args.steps_per_episode,
            dt=args.dt,
            vel_bound=args.vel_bound,
        )

        tqdm.write(f"Epoch {epoch + 1:4d}/{args.epochs}  avg_loss={avg_loss:.4e}")

        if use_wandb:
            wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = save_dir / f"gnn_epoch_{epoch + 1:04d}.pt"
            torch.save(gnn, ckpt_path)
            tqdm.write(f"  → checkpoint saved to {ckpt_path}")

    final_path = save_dir / "gnn_final.pt"
    torch.save(gnn, final_path)
    print(f"Training complete. Model saved to {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AxionGNN online with residual loss across multiple random scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scene
    parser.add_argument(
        "--scene",
        choices=list(SCENE_BUILDERS),
        default="random",
        help="Scene type. 'random' rebuilds a new random scene every epoch; "
        "'ball' reuses the same fixed ball-bounce scene.",
    )
    parser.add_argument(
        "--num_worlds",
        type=int,
        default=8,
        help="Parallel worlds per scene (batch size ≈ num_worlds × num_bodies)",
    )
    parser.add_argument(
        "--num_objects", type=int, default=2, help="Objects per world in 'random' scenes"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Epoch 0 uses base_seed, epoch 1 uses base_seed+1, etc.",
    )

    # GNN architecture (ignored when loading from a checkpoint)
    parser.add_argument("--message_passes", type=int, default=10)
    parser.add_argument("--hidden_dims", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=2)

    # Training
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of episodes (one scene per episode)"
    )
    parser.add_argument(
        "--steps_per_episode", type=int, default=50, help="Physics steps per episode"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="Physics timestep in seconds")
    parser.add_argument(
        "--vel_bound",
        type=float,
        default=2.0,
        help="Initial velocity randomisation range [-v, v] m/s and rad/s",
    )
    parser.add_argument(
        "--warm_start_constr_forces",
        action="store_true",
        default=False,
        help="Compute physically consistent constraint forces before "
        "evaluating the residual (2 extra linear solves per step).",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a saved AxionGNN to resume from"
    )
    parser.add_argument("--save_dir", type=str, default="data/gnn_data/residual_training")
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save a checkpoint every N epochs"
    )

    # Misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--wandb", type=str, default=None, help="W&B run name (omit to disable wandb)"
    )

    args = parser.parse_args()
    main(args)
