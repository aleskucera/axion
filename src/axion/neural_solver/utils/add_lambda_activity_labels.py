"""
Add binary lambda-activity labels to an HDF5 dataset.

Example:
python src/axion/neural_solver/utils/add_lambda_activity_labels.py \
    --input src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1.hdf5 \
    --output src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1WithActivityLabels.hdf5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

THRESHOLD_DEFAULT = 1000


def _resolve_data_group(h5_file: h5py.File) -> h5py.Group:
    if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
        return h5_file["data"]
    return h5_file


def _fill_lonely_zeros(
    lambda_activity: np.ndarray,
    *,
    min_ones_each_side: int = 3,
) -> np.ndarray:
    """
    Fill isolated zero labels in time when surrounded by activity.

    For each time step t, replace lambda_activity[t] = 0 with 1 only if
    there are at least `min_ones_each_side` consecutive 1s immediately
    before and after t (checked along axis 0 / time).
    """
    if min_ones_each_side <= 0:
        return lambda_activity

    if lambda_activity.shape[0] < (2 * min_ones_each_side + 1):
        return lambda_activity

    k = min_ones_each_side
    smoothed = lambda_activity.copy()
    activity = smoothed.reshape(smoothed.shape[0], -1)
    center = activity[k : activity.shape[0] - k]
    lonely_zero_mask = center == 0

    for offset in range(1, k + 1):
        lonely_zero_mask &= activity[k - offset : activity.shape[0] - k - offset] == 1
        lonely_zero_mask &= activity[k + offset : activity.shape[0] - k + offset] == 1

    center[lonely_zero_mask] = 1
    return smoothed


def add_lambda_activity_labels(
    dataset_path: Path,
    *,
    threshold: float = THRESHOLD_DEFAULT,
    dataset_name: str = "lambda_activity",
    lonely_zero_context: int = 3,
    overwrite: bool = False,
) -> None:
    with h5py.File(dataset_path, "r+") as h5_file:
        data_group = _resolve_data_group(h5_file)

        if "lambdas" not in data_group or "next_lambdas" not in data_group:
            missing = [
                name for name in ("lambdas", "next_lambdas") if name not in data_group
            ]
            raise KeyError(
                f"Cannot compute lambda activity, missing dataset(s): {missing}"
            )

        lambdas_ds = data_group["lambdas"]
        next_lambdas_ds = data_group["next_lambdas"]

        if lambdas_ds.shape != next_lambdas_ds.shape:
            raise ValueError(
                "Cannot compute lambda activity because shapes differ: "
                f"lambdas={lambdas_ds.shape}, next_lambdas={next_lambdas_ds.shape}"
            )

        if dataset_name in data_group:
            if not overwrite:
                raise ValueError(
                    f"Dataset '{dataset_name}' already exists. "
                    "Use --overwrite to replace it."
                )
            del data_group[dataset_name]

        lambda_activity = (
            np.abs(next_lambdas_ds[...] - lambdas_ds[...]) > float(threshold)
        ).astype(np.uint8)
        lambda_activity = _fill_lonely_zeros(
            lambda_activity,
            min_ones_each_side=lonely_zero_context,
        )

        data_group.create_dataset(
            dataset_name,
            data=lambda_activity,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
        data_group[dataset_name].attrs["threshold"] = float(threshold)
        data_group[dataset_name].attrs["description"] = (
            "Binary lambda activity label where abs(next_lambdas - lambdas) > threshold."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add binary lambda-activity labels to an HDF5 dataset using "
            "abs(next_lambdas - lambdas) > threshold."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source HDF5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output HDF5 path. If provided, input file is copied first and "
            "labels are added to the copy."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default= THRESHOLD_DEFAULT,
        help="Activity threshold used in abs(next_lambdas - lambdas) > threshold.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lambda_activity",
        help="Name of the output activity dataset.",
    )
    parser.add_argument(
        "--lonely-zero-context",
        type=int,
        default=3,
        help=(
            "If a single 0 is surrounded by this many consecutive 1s on both sides "
            "in time, convert that 0 to 1."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace output dataset if it already exists. "
            "When --output is provided, replacement is already enabled by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path
    overwrite_labels = args.overwrite or args.output is not None

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError(f"Input file must be an HDF5 file, got: {input_path}")

    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

    add_lambda_activity_labels(
        output_path,
        threshold=args.threshold,
        dataset_name=args.dataset_name,
        lonely_zero_context=args.lonely_zero_context,
        overwrite=overwrite_labels,
    )
    print(f"Added '{args.dataset_name}' to: {output_path}")


if __name__ == "__main__":
    main()

