import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import wandb

from axion.gnn import LossGNN
from axion.gnn import AxionGNN
from axion.gnn import AxionDatasetGNN


def train_epoch(
    model: AxionGNN,
    loss_fn: LossGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
):
    model.train()
    total_losses = defaultdict(float)
    loss_counts = defaultdict(int)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for i, data in enumerate(pbar, 1):
        data: HeteroData = data.to(device)
        optimizer.zero_grad()
        states, lambdas = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        loss_dict = loss_fn(data, states, lambdas)
        loss_dict["total_loss"][0].backward()
        optimizer.step()
        for k, v in loss_dict.items():
            total_losses[k] += v[0].item() * v[1]
            loss_counts[k] += v[1]
        pbar.set_postfix({"loss": f"{total_losses['total_loss']/loss_counts['total_loss']:.4f}"})
    return {k: v / loss_counts[k] if loss_counts[k] else 0.0 for k, v in total_losses.items()}


def validate_epoch(
    model: AxionGNN,
    loss_fn: LossGNN,
    loader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int,
):
    model.eval()
    total_losses = defaultdict(float)
    loss_counts = defaultdict(int)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    with torch.no_grad():
        for i, data in enumerate(pbar, 1):
            data: HeteroData = data.to(device)
            states, lambdas = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss_dict = loss_fn(data, states, lambdas)
            for k, v in loss_dict.items():
                total_losses[k] += v[0].item() * v[1]
                loss_counts[k] += v[1]
            pbar.set_postfix(
                {"loss": f"{total_losses['total_loss']/loss_counts['total_loss']:.4f}"}
            )
    return {k: v / loss_counts[k] if loss_counts[k] else 0.0 for k, v in total_losses.items()}


def train(
    model: AxionGNN,
    loss_fn: LossGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
):
    min_val_loss = torch.inf
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            config["device"],
            epoch,
            config["epochs"],
        )
        val_loss = validate_epoch(
            model,
            loss_fn,
            val_loader,
            config["device"],
            epoch,
            config["epochs"],
        )
        print(
            f"Epoch {epoch+1} Complete. "
            f"Train Loss: {train_loss['total_loss']:.6f}, "
            f"Val Loss: {val_loss['total_loss']:.6f}"
        )
        wandb.log(
            {
                **{f"train/{k}": v for k, v in train_loss.items()},
                **{f"val/{k}": v for k, v in val_loss.items()},
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss["total_loss"])
        else:
            scheduler.step()
        if val_loss["total_loss"] < min_val_loss:
            min_val_loss = val_loss["total_loss"]
            save_dir = config["dataset_root"] / "models"
            save_dir.mkdir(parents=True, exist_ok=True)
            if not config["model_name"] is None:
                torch.save(model, config["dataset_root"] / "models" / config["model_name"])


def main(config):
    loss_fn = LossGNN(config["loss_type"], config["device"])
    train_dataset = AxionDatasetGNN(root=config["dataset_root"] / "train_dataset")
    val_dataset = AxionDatasetGNN(root=config["dataset_root"] / "val_dataset")
    model = AxionGNN(
        config["message_passes"],
        config["hidden_dims"],
        config["hidden_layers"],
        config["normalize"],
        stats=train_dataset.stats if config["normalize_input"] else None,
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_init"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    model.to(config["device"])
    wandb_group_name = config["wandb"].split("/") if not config["wandb"] is None else [""]
    wandb.init(
        project="axion-gnn",
        config=config,
        group=wandb_group_name[0] if len(wandb_group_name) > 1 else None,
        name=wandb_group_name[-1],
        mode="online" if not config["wandb"] is None else "disabled",
    )
    wandb.config.update(
        {
            "optimizer_type": optimizer.__class__.__name__,
            "scheduler_type": scheduler.__class__.__name__,
        }
    )
    try:
        train(model, loss_fn, train_loader, val_loader, optimizer, scheduler, config)
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN Sim2D")

    parser.add_argument("--message_passes", type=int, default=5)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_dims", type=int, default=128)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize_input", action="store_true", default=False)

    parser.add_argument("--loss_type", type=str, default="l1_loss")
    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/gnn_data/dataset"))
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--wandb", type=str, default=None)

    args = parser.parse_args()
    config = vars(args)

    if config["device"] is None:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config["device"] = torch.device(config["device"])
    print(f"Running on {config["device"]}.")

    main(config)
