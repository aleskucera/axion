import argparse
from copy import deepcopy
from pathlib import Path

import wandb

from axion.neural_solver.algorithms.mse_trainer import MSETrainer
from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.train.config_builder import (
    build_cli_cfg,
    load_default_cfg,
    validate_cfg,
)
from axion.neural_solver.utils.python_utils import set_random_seed


sweep_config_mse_transformer = {
    "method": "random",
    "name": "mse-transformer-architecture-sweep",
    "metric": {"goal": "minimize", "name": "train/regression_loss"},
    "parameters": {
        "block_size": {"values": [16, 24, 32]},
        "n_layer": {"values": [6, 8, 10, 12]},
        "n_head": {"values": [4, 6, 8, 12, 16]},
        "n_embd": {"values": [128, 192, 256, 320, 384]},
        "head_size": {"values": [[], [32], [32, 32], [128], [128, 128]]},
    },
}


def _parse_args():
    p = argparse.ArgumentParser(description="Run MSE transformer architecture W&B sweep.")
    p.add_argument("--cfg", required=True, help="Path to base YAML config (defaults)")
    p.add_argument("--logdir", required=True, help="Base directory for checkpoints/logs")
    p.add_argument("--checkpoint", default=None, help="Checkpoint to restore")
    p.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0)")
    p.add_argument("--sweep_id", type=str, default=None, help="Existing sweep id to join")
    p.add_argument("--project", type=str, default="neural-solver-mse-transformer", help="W&B project")
    p.add_argument("--count", type=int, default=None, help="Max runs for this agent")
    return p.parse_args()


def _apply_mse_sweep_overrides(cfg: dict, sweep: dict) -> dict:
    out = deepcopy(cfg)

    transformer_cfg = out.setdefault("network", {}).setdefault("transformer", {})
    model_mlp_cfg = (
        out.setdefault("network", {})
        .setdefault("model", {})
        .setdefault("mlp", {})
    )

    if "block_size" in sweep and sweep["block_size"] is not None:
        transformer_cfg["block_size"] = int(sweep["block_size"])
    if "n_layer" in sweep and sweep["n_layer"] is not None:
        transformer_cfg["n_layer"] = int(sweep["n_layer"])
    if "n_head" in sweep and sweep["n_head"] is not None:
        transformer_cfg["n_head"] = int(sweep["n_head"])
    if "n_embd" in sweep and sweep["n_embd"] is not None:
        transformer_cfg["n_embd"] = int(sweep["n_embd"])
    if "head_size" in sweep and sweep["head_size"] is not None:
        model_mlp_cfg["layer_sizes"] = list(sweep["head_size"])

    return out


def _validate_transformer_constraints(cfg: dict) -> None:
    transformer_cfg = cfg["network"]["transformer"]
    n_head = int(transformer_cfg["n_head"])
    n_embd = int(transformer_cfg["n_embd"])
    if n_embd % n_head != 0:
        raise ValueError(
            f"Invalid sweep sample: n_embd={n_embd} must be divisible by n_head={n_head}."
        )


def _make_sweep_train_fn(args):
    def _run_one_sweep_iteration():
        run = wandb.init()
        should_skip_run = False
        try:
            cfg = load_default_cfg(args.cfg)
            cfg = _apply_mse_sweep_overrides(cfg, dict(wandb.config))
            cfg.setdefault("network", {})
            cfg["network"]["model_impl"] = "mse"
            _validate_transformer_constraints(cfg)
            validate_cfg(cfg)

            seed = cfg["algorithm"].get("seed", 0)
            set_random_seed(seed)
            cfg["algorithm"]["seed"] = seed

            run_id = run.id if run is not None else "unknown-run"
            run_logdir = Path(args.logdir) / run_id
            cfg["cli"] = build_cli_cfg(
                logdir=run_logdir,
                train=True,
                cfg=cfg,
                skip_check_log_override=True,
            )

            print(f"Device = {args.device}")
            print(f"Sweep run logdir = {run_logdir}")
            neural_env = NnTrainingInterface(**cfg["env"], device=args.device)
            algo = MSETrainer(
                neural_env=neural_env,
                model_checkpoint_path=args.checkpoint,
                cfg=cfg,
                device=args.device,
            )
            algo.train()
        except Exception as exc:
            if isinstance(exc, ValueError) and "must be divisible by n_head" in str(exc):
                should_skip_run = True
                print(f"Skipping invalid sweep sample: {exc}")
                return
            print(f"Sweep run failed: {exc}")
            raise
        finally:
            if run is not None:
                if should_skip_run:
                    run.summary["skipped_invalid_sample"] = True
                run.finish()

    return _run_one_sweep_iteration


if __name__ == "__main__":
    args = _parse_args()
    wandb.login()

    project_name = args.project
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_config_mse_transformer, project=project_name)
        print(f"\n=== Created NEW Sweep: {sweep_id} ===")
    else:
        sweep_id = args.sweep_id
        print(f"\n=== Joining EXISTING Sweep: {sweep_id} ===")

    wandb.agent(
        sweep_id,
        function=_make_sweep_train_fn(args),
        project=project_name,
        count=args.count,
    )
