import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split trajectory-based HDF5 dataset into train/validation files."
    )
    parser.add_argument(
        "--input",
        default=(
            "src/axion/neural_solver/datasets/Pendulum/"
            "pendulumLambdasValid500klen400envs250seed1WithActivityLabels_th1000.hdf5"
        ),
        help="Path to source HDF5 dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of trajectories used for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for shuffled trajectory split (default: 1).",
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="Optional path for train output HDF5 file.",
    )
    parser.add_argument(
        "--val-output",
        default=None,
        help="Optional path for validation output HDF5 file.",
    )
    return parser.parse_args()


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    suffix = input_path.suffix or ".hdf5"
    parent = input_path.parent
    train_path = parent / f"{stem}_train{suffix}"
    val_path = parent / f"{stem}_val{suffix}"
    return train_path, val_path


def copy_attrs(src_obj, dst_obj):
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def split_hdf5(input_path: Path, train_output: Path, val_output: Path, train_ratio: float, seed: int):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be strictly between 0 and 1.")

    with h5py.File(input_path, "r") as src:
        if "data" not in src:
            raise KeyError("Expected top-level group 'data' in input HDF5.")

        data_group = src["data"]
        dataset_names = list(data_group.keys())
        if not dataset_names:
            raise ValueError("Input HDF5 group 'data' is empty.")

        n_trajectories = data_group[dataset_names[0]].shape[1]
        train_count = int(n_trajectories * train_ratio)
        val_count = n_trajectories - train_count
        if train_count == 0 or val_count == 0:
            raise ValueError("Split produced empty train or validation set.")

        rng = np.random.default_rng(seed)
        indices = np.arange(n_trajectories)
        rng.shuffle(indices)
        train_idx = np.sort(indices[:train_count])
        val_idx = np.sort(indices[train_count:])

        with h5py.File(train_output, "w") as train_f, h5py.File(val_output, "w") as val_f:
            copy_attrs(src, train_f)
            copy_attrs(src, val_f)
            train_group = train_f.create_group("data")
            val_group = val_f.create_group("data")
            copy_attrs(data_group, train_group)
            copy_attrs(data_group, val_group)

            for name in dataset_names:
                ds = data_group[name]
                if ds.ndim < 2:
                    raise ValueError(
                        f"Dataset 'data/{name}' has ndim={ds.ndim}; expected trajectories on axis 1."
                    )
                if ds.shape[1] != n_trajectories:
                    raise ValueError(
                        f"Dataset 'data/{name}' has inconsistent trajectory axis shape {ds.shape[1]}."
                    )

                train_data = ds[:, train_idx, ...]
                val_data = ds[:, val_idx, ...]

                train_ds = train_group.create_dataset(name, data=train_data, dtype=ds.dtype)
                val_ds = val_group.create_dataset(name, data=val_data, dtype=ds.dtype)
                copy_attrs(ds, train_ds)
                copy_attrs(ds, val_ds)

    print(f"Input trajectories: {n_trajectories}")
    print(f"Train trajectories: {train_count} -> {train_output}")
    print(f"Validation trajectories: {val_count} -> {val_output}")


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    default_train, default_val = default_output_paths(input_path)
    train_output = Path(args.train_output).expanduser().resolve() if args.train_output else default_train
    val_output = Path(args.val_output).expanduser().resolve() if args.val_output else default_val

    os.makedirs(train_output.parent, exist_ok=True)
    os.makedirs(val_output.parent, exist_ok=True)

    split_hdf5(
        input_path=input_path,
        train_output=train_output,
        val_output=val_output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
