"""
Export a trained MSEModel checkpoint to ONNX.

Usage
-----
python export_to_onnx.py \\
    --checkpoint path/to/best_model.pt \\
    --cfg       path/to/mseNetwork.yaml \\
    --output    fast_mse.onnx \\
    --batch-size 1 \\
    --device    cuda:0

All flags except --checkpoint have sensible defaults:
  --cfg         not required; enables yaml cross-validation when supplied
  --output      defaults to <checkpoint_stem>.onnx beside the checkpoint
  --batch-size  1
  --device      cuda:0 (use cpu if no GPU available)
"""

import argparse
import sys
import warnings
from pathlib import Path

import onnx
import torch
import yaml

try:
    from axion.neural_solver.models.fast_mse_model import FastMSEModel
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from axion.neural_solver.models.fast_mse_model import FastMSEModel


def _load_yaml(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_checkpoint(checkpoint_path: Path, device: str):
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(payload, (list, tuple)):
        model = payload[0]
        robot_name = payload[1] if len(payload) > 1 else None
    else:
        model = payload
        robot_name = None
    return model, robot_name


def _cross_validate(fast_model: FastMSEModel, cfg: dict, yaml_path: Path) -> None:
    network = cfg.get("network", {})

    yaml_block_size = network.get("transformer", {}).get("block_size")
    model_block_size = fast_model.transformer_model.config.block_size
    if yaml_block_size is not None and yaml_block_size != model_block_size:
        warnings.warn(
            f"[cross-validate] block_size mismatch: yaml={yaml_block_size}, "
            f"checkpoint={model_block_size}. Using checkpoint value.",
            stacklevel=2,
        )

    yaml_lambda = network.get("enable_lambda_head", False)
    model_has_lambda = fast_model.lambda_output_dim > 0
    if yaml_lambda != model_has_lambda:
        warnings.warn(
            f"[cross-validate] enable_lambda_head mismatch: "
            f"yaml={yaml_lambda}, checkpoint implies lambda_output_dim="
            f"{fast_model.lambda_output_dim}. Checkpoint is authoritative.",
            stacklevel=2,
        )


def _print_summary(fast_model: FastMSEModel, low_dim_names: list[str], output_path: Path) -> None:
    T = fast_model.transformer_model.config.block_size
    D = fast_model.total_low_dim
    print("\n--- Export summary ---")
    print(f"  input shape      : (B, T={T}, D={D})")
    print(f"  state_output_dim : {fast_model.state_output_dim}")
    print(f"  lambda_output_dim: {fast_model.lambda_output_dim}")
    if low_dim_names:
        print(f"  low_dim keys (concatenation order):")
        for name in low_dim_names:
            print(f"    {name}")
    else:
        print("  (no yaml cfg supplied — key order unknown)")
    print(f"  output file      : {output_path}")
    print("----------------------\n")


def export(
    checkpoint_path: str,
    cfg_path: str | None = None,
    output_path: str | None = None,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> Path:
    """
    Convert a trained MSEModel checkpoint to ONNX.

    Returns the path to the written .onnx file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load yaml (optional)
    cfg = {}
    low_dim_names: list[str] = []
    if cfg_path is not None:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        low_dim_names = cfg.get("inputs", {}).get("low_dim", [])

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    mse_model, robot_name = _load_checkpoint(checkpoint_path, device)
    if robot_name:
        print(f"  robot: {robot_name}")
    mse_model.eval()

    # Convert to export-friendly model
    print("Converting to FastMSEModel...")
    fast_model = FastMSEModel.from_mse_model(mse_model, device=device)
    fast_model.eval()

    # Cross-validate yaml vs checkpoint
    if cfg:
        _cross_validate(fast_model, cfg, cfg_path)

    _print_summary(fast_model, low_dim_names, output_path)

    # Build static dummy input (zeros; shape is all that matters for tracing)
    B = batch_size
    T = fast_model.transformer_model.config.block_size
    D = fast_model.total_low_dim
    dummy = torch.zeros(B, T, D, device=device)

    # Export
    print(f"Running torch.onnx.export  (opset 17, static shapes B={B} T={T} D={D}) ...")
    torch.onnx.export(
        fast_model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,       # static shapes required for TRT + CUDA graph capture
        do_constant_folding=True,
    )

    # Validate
    print("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"\nONNX export successful: {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained MSEModel checkpoint to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the trained .pt checkpoint (saved as [model, robot_name]).",
    )
    parser.add_argument(
        "--cfg", default=None,
        help="Path to mseNetwork.yaml. Optional; enables cross-validation when supplied.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx file path. Defaults to <checkpoint>.onnx beside the checkpoint.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Static batch size baked into the ONNX graph.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device to run the export on.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(
        checkpoint_path=args.checkpoint,
        cfg_path=args.cfg,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
