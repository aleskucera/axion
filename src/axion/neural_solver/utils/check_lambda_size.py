#!/usr/bin/env python3
"""Check whether multiple trajectory datasets are compatible for joint training.

This script validates the same dataset-level assumptions used by
`TrajectoryDataset` in `utils/datasets.py`:
- every file has `data` group with `mode == "trajectory"`
- same effective sample keys across datasets
- same per-key flattened feature dimensions (last-dim after flatten)

Usage examples:
  python src/axion/neural_solver/utils/check_lambda_size.py \
    src/axion/neural_solver/datasets/Pendulum/a.hdf5 \
    src/axion/neural_solver/datasets/Pendulum/b.hdf5

  python src/axion/neural_solver/utils/check_lambda_size.py \
    --config src/axion/neural_solver/train/cfg/Pendulum/mseNetwork.yaml \
    --config-field train_dataset_path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import yaml


def _product(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _extract_signature(dataset_path: Path) -> Dict:
    with h5py.File(dataset_path, "r", swmr=True, libver="latest") as f:
        if "data" not in f:
            raise ValueError(f"{dataset_path}: missing `data` group")
        data = f["data"]

        mode = data.attrs.get("mode", None)
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode != "trajectory":
            raise ValueError(
                f"{dataset_path}: expected data.attrs['mode']=='trajectory', got {mode!r}"
            )

        key_to_dim: Dict[str, int] = {}
        key_to_shape: Dict[str, Tuple[int, ...]] = {}

        for key in data.keys():
            if key in ("traj_lengths", "axion_contacts"):
                continue
            arr = data[key]
            # Expected shape in generator: (T, B, ...)
            if arr.ndim < 3:
                raise ValueError(
                    f"{dataset_path}: data/{key} has ndim={arr.ndim}, expected >= 3 (T, B, ...)"
                )
            flat_dim = _product(arr.shape[2:])
            key_to_dim[key] = flat_dim
            key_to_shape[key] = tuple(int(x) for x in arr.shape)

        if "axion_contacts" in data:
            contacts = data["axion_contacts"]
            for sub_key in contacts.keys():
                arr = contacts[sub_key]
                # Contact channels can be scalar per (T, B) sample (ndim == 2),
                # e.g. contact_count. TrajectoryDataset handles these as dim=1.
                if arr.ndim < 2:
                    raise ValueError(
                        f"{dataset_path}: data/axion_contacts/{sub_key} has ndim={arr.ndim}, expected >= 2"
                    )
                merged_key = f"axion_{sub_key}"
                key_to_dim[merged_key] = (
                    _product(arr.shape[2:]) if arr.ndim > 2 else 1
                )
                key_to_shape[merged_key] = tuple(int(x) for x in arr.shape)

        traj_lengths = "traj_lengths" in data
        num_steps = int(data["states"].shape[0]) if "states" in data else None
        num_trajs = int(data["states"].shape[1]) if "states" in data else None

        def _attr_str(name: str):
            raw = data.attrs.get(name, None)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)

        lambda_layout = _attr_str("lambda_layout")
        lambda_active_blocks = _attr_str("lambda_active_blocks")
        lambda_dim_attr = data.attrs.get("lambda_dim", None)
        if lambda_dim_attr is not None:
            lambda_dim_attr = int(lambda_dim_attr)

        return {
            "path": str(dataset_path),
            "mode": mode,
            "keys": set(key_to_dim.keys()),
            "key_to_dim": key_to_dim,
            "key_to_shape": key_to_shape,
            "has_traj_lengths": traj_lengths,
            "num_steps": num_steps,
            "num_trajs": num_trajs,
            "lambda_layout": lambda_layout,
            "lambda_active_blocks": lambda_active_blocks,
            "lambda_dim_attr": lambda_dim_attr,
        }


def _compare_signatures(signatures: List[Dict]) -> List[str]:
    errors: List[str] = []
    ref = signatures[0]
    ref_keys = ref["keys"]
    ref_dims = ref["key_to_dim"]

    for sig in signatures[1:]:
        path = sig["path"]
        keys = sig["keys"]

        missing = sorted(ref_keys - keys)
        extra = sorted(keys - ref_keys)
        if missing:
            errors.append(f"{path}: missing keys vs first dataset: {missing}")
        if extra:
            errors.append(f"{path}: extra keys vs first dataset: {extra}")

        shared = sorted(ref_keys & keys)
        for key in shared:
            if int(sig["key_to_dim"][key]) != int(ref_dims[key]):
                errors.append(
                    f"{path}: key `{key}` feature dim mismatch "
                    f"({sig['key_to_dim'][key]} vs {ref_dims[key]})"
                )

    return errors


def _read_paths_from_config(config_path: Path, config_field: str) -> List[str]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        value = cfg["algorithm"]["dataset"][config_field]
    except Exception as exc:
        raise KeyError(
            f"Could not find algorithm.dataset.{config_field} in {config_path}"
        ) from exc

    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    raise TypeError(
        f"algorithm.dataset.{config_field} must be a string or list, got {type(value).__name__}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check compatibility of multiple trajectory HDF5 datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset paths (same style as train_dataset_path/valid_dataset_path lists).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config to load dataset list from.",
    )
    parser.add_argument(
        "--config-field",
        type=str,
        default="train_dataset_path",
        choices=("train_dataset_path", "valid_dataset_path"),
        help="Which algorithm.dataset field to read from --config.",
    )
    args = parser.parse_args()

    raw_paths: List[str] = list(args.datasets)
    if args.config:
        raw_paths.extend(_read_paths_from_config(Path(args.config), args.config_field))

    # Preserve order while deduplicating.
    dataset_paths: List[Path] = []
    seen = set()
    for p in raw_paths:
        path = Path(p)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dataset_paths.append(path)

    if len(dataset_paths) < 2:
        print("Need at least 2 dataset paths to compare.")
        return 2

    print("Checking datasets:")
    for p in dataset_paths:
        print(f"  - {p}")
        if not p.exists():
            print(f"ERROR: file does not exist: {p}")
            return 2

    signatures: List[Dict] = []
    for p in dataset_paths:
        signatures.append(_extract_signature(p))

    errors = _compare_signatures(signatures)

    print("\nDataset quick summary:")
    for sig in signatures:
        lambda_extra = ""
        if sig.get("lambda_dim_attr") is not None:
            lambda_extra += f", lambda_dim(attr)={sig['lambda_dim_attr']}"
        if sig.get("lambda_layout"):
            lambda_extra += f", lambda_layout={sig['lambda_layout']!r}"
        if sig.get("lambda_active_blocks"):
            lambda_extra += f", lambda_active_blocks={sig['lambda_active_blocks']!r}"
        print(
            f"- {sig['path']}: keys={len(sig['keys'])}, "
            f"states(T,B)=({sig['num_steps']},{sig['num_trajs']}), "
            f"traj_lengths={sig['has_traj_lengths']}{lambda_extra}"
        )

    if errors:
        print("\nINCOMPATIBLE:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nCOMPATIBLE: all compared datasets can be concatenated for training.")
    print("Checked: key set equality and per-key flattened feature dimensions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
