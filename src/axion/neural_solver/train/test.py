import sys
import os
import csv

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

import yaml
import torch
import numpy as np
import warp as wp
wp.config.verify_cuda = True

from torch.utils.data import DataLoader

from axion.neural_solver.utils.python_utils import set_random_seed
from axion.neural_solver.algorithms.sequence_model_trainer import SequenceModelTrainer
from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.utils.evaluator import NeuralSimEvaluator
from axion.neural_solver.utils.time_report import TimeReport

# ========================= EDITABLE FIELDS =========================

MODEL_DIR = "src/axion/neural_solver/train/trained_models/03-19-2026-01-16-45"

VALIDATION_DATASETS = [
    "src/axion/neural_solver/datasets/Pendulum/pendulumContactsValid250klen500envs250seed1.hdf5",
    "src/axion/neural_solver/datasets/Pendulum/pendulumContactsValid1Mlen2000envs250seed1.hdf5"
]

DEVICE = "cuda:0"

# ====================================================================


def discover_checkpoints(nn_dir):
    """Return list of .pt files found in the nn/ directory."""
    if not os.path.isdir(nn_dir):
        raise FileNotFoundError(f"nn/ directory not found at {nn_dir}")
    return sorted(
        f for f in os.listdir(nn_dir)
        if f.endswith(".pt")
    )


def read_last_epoch(filepath):
    """Read the last line from an epoch-log text file, return int or None."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def resolve_epoch(checkpoint_name, nn_dir, num_epochs):
    """Best-effort epoch resolution for a given checkpoint filename."""
    stem = os.path.splitext(checkpoint_name)[0]

    if stem == "final_model":
        return num_epochs - 1

    if stem.startswith("model_epoch"):
        try:
            return int(stem.replace("model_epoch", ""))
        except ValueError:
            pass

    epoch_log = os.path.join(nn_dir, f"saved_{stem}_epochs.txt")
    epoch = read_last_epoch(epoch_log)
    if epoch is not None:
        return epoch

    return None


def dataset_display_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def print_header(text, width=80):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(text, width=80):
    print("-" * width)
    print(f"  {text}")
    print("-" * width)


def print_metric(name, value, indent=4):
    print(f"{' ' * indent}{name:<35s} {to_scalar(value):.8f}")


def to_scalar(v):
    return v.item() if hasattr(v, "item") else v


def export_csv(filepath, checkpoint_files, checkpoint_epochs, validation_datasets,
               all_results, model_dir, eval_horizon, num_eval_rollouts, num_epochs):
    """Write all results to a CSV file with one row per (checkpoint, dataset) pair."""
    ds_names = [dataset_display_name(p) for p in validation_datasets]

    # Collect the union of all metric keys across checkpoints/datasets
    valid_keys = set()
    eval_keys = set()
    for ckpt in checkpoint_files:
        for ds in ds_names:
            res = all_results[ckpt][ds]
            valid_keys.update(res["valid_loss_itemized"].keys())
            eval_keys.update(res["eval_error_stats"]["overall"].keys())
    valid_keys = sorted(valid_keys)
    eval_keys = sorted(eval_keys)

    header = (
        ["model_dir", "checkpoint", "epoch", "dataset",
         "num_epochs", "eval_horizon", "num_rollouts",
         "valid_total_loss"]
        + [f"valid_{k}" for k in valid_keys]
        + [f"eval_{k}" for k in eval_keys]
    )

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ckpt in checkpoint_files:
            ep = checkpoint_epochs[ckpt]
            stem = os.path.splitext(ckpt)[0]
            for ds in ds_names:
                res = all_results[ckpt][ds]
                row = [
                    model_dir, stem,
                    ep if ep is not None else "",
                    ds, num_epochs, eval_horizon, num_eval_rollouts,
                    f"{to_scalar(res['valid_loss']):.8f}",
                ]
                for k in valid_keys:
                    row.append(f"{to_scalar(res['valid_loss_itemized'].get(k, '')):.8f}"
                               if k in res["valid_loss_itemized"] else "")
                for k in eval_keys:
                    v = res["eval_error_stats"]["overall"].get(k, "")
                    row.append(f"{to_scalar(v):.8f}" if v != "" else "")
                writer.writerow(row)

    print(f"\n  CSV saved to: {filepath}")


def load_model_into_trainer(algo, checkpoint_path, device):
    """Swap the neural model inside an existing SequenceModelTrainer."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algo.neural_model = checkpoint[0]
    algo.neural_model.to(device)
    algo.neural_model.eval()
    algo.utils_provider.set_neural_model(algo.neural_model)


def evaluate_checkpoint(algo, neural_env, validation_datasets, eval_horizon,
                        num_eval_rollouts, eval_passive, device):
    """Run validation loss + rollout eval for each dataset. Returns results dict."""
    results = {}

    for dataset_path in validation_datasets:
        name = dataset_display_name(dataset_path)
        results[name] = {}

        # --- Validation loss ---
        print(f"    Computing validation loss ...")
        ds = algo.valid_datasets[name]
        loader = DataLoader(
            dataset=ds,
            batch_size=algo.batch_size,
            collate_fn=algo.collate_fn,
            shuffle=False,
            num_workers=algo.num_data_workers,
            drop_last=True,
        )
        loader_iter = iter(loader)
        num_batches = len(loader)

        avg_loss, avg_loss_itemized, _ = algo.one_epoch(
            train=False,
            dataloader=loader,
            dataloader_iter=loader_iter,
            num_batches=num_batches,
            shuffle=False,
        )
        results[name]["valid_loss"] = avg_loss
        results[name]["valid_loss_itemized"] = avg_loss_itemized

        print(f"    Validation loss = {avg_loss:.8f}")

        # --- Rollout evaluation ---
        print(f"    Running rollout evaluation (horizon={eval_horizon}, "
              f"rollouts={num_eval_rollouts}) ...")
        evaluator = NeuralSimEvaluator(
            neural_env,
            hdf5_dataset_path=dataset_path,
            eval_horizon=eval_horizon,
            device=device,
        )
        eval_error, _, error_stats = evaluator.evaluate_action_mode(
            num_traj=num_eval_rollouts,
            eval_mode="rollout",
            env_mode="neural",
            trajectory_source="dataset",
            passive=eval_passive,
        )
        results[name]["eval_error_stats"] = error_stats
        results[name]["eval_mse"] = (eval_error ** 2).mean()

        print(f"    Rollout MSE    = {error_stats['overall']['error(MSE)'].item():.8f}")

    return results


if __name__ == "__main__":
    model_dir = os.path.abspath(MODEL_DIR)
    nn_dir = os.path.join(model_dir, "nn")
    cfg_path = os.path.join(model_dir, "cfg.yaml")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Could not find cfg.yaml at {cfg_path}. "
            f"Expected directory structure: <model_dir>/cfg.yaml"
        )

    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    num_epochs = int(cfg["algorithm"].get("num_epochs", 0))

    # Discover checkpoints
    checkpoint_files = discover_checkpoints(nn_dir)
    if not checkpoint_files:
        raise RuntimeError(f"No .pt checkpoint files found in {nn_dir}")

    checkpoint_epochs = {}
    for ckpt in checkpoint_files:
        checkpoint_epochs[ckpt] = resolve_epoch(ckpt, nn_dir, num_epochs)

    # Build the valid_datasets mapping from user-supplied list
    valid_datasets_cfg = {}
    for path in VALIDATION_DATASETS:
        valid_datasets_cfg[dataset_display_name(path)] = path
    cfg["algorithm"]["dataset"]["valid_datasets"] = valid_datasets_cfg

    # Point the training dataset to the first validation dataset so
    # get_datasets() doesn't crash if the original training data is absent.
    cfg["algorithm"]["dataset"]["train_dataset_path"] = VALIDATION_DATASETS[0]

    # Eval config
    cfg["algorithm"]["eval"]["mode"] = "dataset"
    cfg["algorithm"]["eval"]["dataset_path"] = VALIDATION_DATASETS[0]

    # CLI config for test mode
    cfg["cli"] = {
        "logdir": "./test_eval_logs",
        "train": False,
        "render": False,
        "save_interval": 50,
        "log_interval": 1,
        "eval_interval": 1,
        "skip_check_log_override": False,
    }

    seed = cfg["algorithm"].get("seed", 0)
    set_random_seed(seed)

    eval_cfg = cfg["algorithm"]["eval"]
    eval_horizon = eval_cfg.get("rollout_horizon", 10)
    num_eval_rollouts = eval_cfg.get("num_rollouts", 0)
    eval_passive = eval_cfg.get("passive", True)

    # --- Print run info ---

    print_header("MODEL TESTING")
    print(f"  Model dir  : {MODEL_DIR}")
    print(f"  Config     : {cfg_path}")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoints:")
    for ckpt in checkpoint_files:
        ep = checkpoint_epochs[ckpt]
        ep_str = f"epoch {ep}" if ep is not None else "epoch n/a"
        print(f"    - {ckpt}  ({ep_str})")
    print(f"  Datasets ({len(VALIDATION_DATASETS)}):")
    for i, p in enumerate(VALIDATION_DATASETS):
        print(f"    [{i}] {p}")

    # --- Create environment and trainer once (with first checkpoint) ---

    first_checkpoint_path = os.path.join(nn_dir, checkpoint_files[0])
    neural_env = NnTrainingInterface(**cfg["env"], device=DEVICE)

    if num_eval_rollouts == 0:
        num_eval_rollouts = neural_env.num_envs

    algo = SequenceModelTrainer(
        neural_env=neural_env,
        model_checkpoint_path=first_checkpoint_path,
        cfg=cfg,
        device=DEVICE,
    )

    algo.time_report = TimeReport(cuda_synchronize=False)
    algo.time_report.add_timers(
        ["epoch", "other", "dataloader", "compute_loss", "backward", "eval"]
    )

    # --- Evaluate each checkpoint ---

    all_results = {}

    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(nn_dir, ckpt)
        ep = checkpoint_epochs[ckpt]
        ep_str = f"epoch {ep}" if ep is not None else "epoch n/a"

        print_header(f"Checkpoint: {ckpt}  ({ep_str})")

        load_model_into_trainer(algo, ckpt_path, DEVICE)

        all_results[ckpt] = evaluate_checkpoint(
            algo, neural_env, VALIDATION_DATASETS,
            eval_horizon, num_eval_rollouts, eval_passive, DEVICE,
        )

    # ========================= FINAL SUMMARY =========================

    print_header("TEST RESULTS SUMMARY")
    print(f"  Model dir      : {MODEL_DIR}")
    print(f"  Config         : {cfg_path}")
    print(f"  Num epochs     : {num_epochs}")
    print(f"  Eval horizon   : {eval_horizon}")
    print(f"  Num rollouts   : {num_eval_rollouts}")

    for ckpt in checkpoint_files:
        ep = checkpoint_epochs[ckpt]
        ep_str = f"epoch {ep}" if ep is not None else "epoch n/a"
        print_header(f"{ckpt}  ({ep_str})")

        for dataset_path in VALIDATION_DATASETS:
            name = dataset_display_name(dataset_path)
            res = all_results[ckpt][name]

            print_section(f"Dataset: {name}")

            print(f"\n  [Validation Loss]")
            print_metric("total_loss", res["valid_loss"])
            for key in sorted(res["valid_loss_itemized"]):
                print_metric(key, res["valid_loss_itemized"][key])

            stats = res["eval_error_stats"]
            print(f"\n  [Rollout Evaluation]")
            for key in sorted(stats["overall"]):
                print_metric(key, stats["overall"][key])

    # Compact comparison table across checkpoints
    print_header("COMPARISON (validation loss / rollout MSE)")
    header_cols = ["Checkpoint", "Epoch"]
    ds_names = [dataset_display_name(p) for p in VALIDATION_DATASETS]
    for ds in ds_names:
        short = ds[:40]
        header_cols += [f"{short} valid", f"{short} eval"]

    col_w = [max(30, len(c) + 2) for c in header_cols]
    col_w[0] = 30
    col_w[1] = 8

    header_line = "".join(c.ljust(w) for c, w in zip(header_cols, col_w))
    print(header_line)
    print("-" * len(header_line))

    for ckpt in checkpoint_files:
        ep = checkpoint_epochs[ckpt]
        row = [os.path.splitext(ckpt)[0], str(ep) if ep is not None else "n/a"]
        for ds in ds_names:
            res = all_results[ckpt][ds]
            v_loss = res["valid_loss"]
            e_mse = res["eval_error_stats"]["overall"]["error(MSE)"]
            row += [f"{v_loss:.8f}", f"{to_scalar(e_mse):.8f}"]
        print("".join(v.ljust(w) for v, w in zip(row, col_w)))

    # Export CSV
    model_dir_name = os.path.basename(model_dir)
    csv_path = os.path.join(model_dir, f"test_results_{model_dir_name}.csv")
    export_csv(
        csv_path, checkpoint_files, checkpoint_epochs, VALIDATION_DATASETS,
        all_results, MODEL_DIR, eval_horizon, num_eval_rollouts, num_epochs,
    )

    print("\n" + "=" * 80)
    print("  Done.")
    print("=" * 80)
