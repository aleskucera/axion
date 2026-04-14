"""
Add lambda-activity labels to an HDF5 dataset.

Example:
python src/axion/neural_solver/utils/add_lambda_activity_labels.py \
    --input src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1.hdf5 \
    --output src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1WithActivityLabels.hdf5

# Multiclass 0/1/2
--multiclass

"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

THRESHOLD_DEFAULT = 1000
LOW_THRESHOLD_DEFAULT = 0.1


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


def _fill_lonely_zeros_multiclass(
    labels: np.ndarray,
    *,
    min_nonzero_each_side: int = 3,
) -> np.ndarray:
    """
    Fill isolated 0 labels in time when surrounded by non-zero activity.

    For each time step t, replace labels[t] = 0 with the maximum label (1 or 2)
    if there are at least `min_nonzero_each_side` consecutive steps immediately
    before and after t where the label is non-zero.
    """
    if min_nonzero_each_side <= 0:
        return labels

    if labels.shape[0] < (2 * min_nonzero_each_side + 1):
        return labels

    k = min_nonzero_each_side
    smoothed = labels.copy()
    activity = smoothed.reshape(smoothed.shape[0], -1)
    center = activity[k : activity.shape[0] - k]
    lonely_zero_mask = center == 0

    neighbor_max = np.zeros_like(center)
    for offset in range(1, k + 1):
        left = activity[k - offset : activity.shape[0] - k - offset]
        right = activity[k + offset : activity.shape[0] - k + offset]

        lonely_zero_mask &= (left > 0) & (right > 0)
        neighbor_max = np.maximum(neighbor_max, np.maximum(left, right))

    center[lonely_zero_mask] = neighbor_max[lonely_zero_mask]
    return smoothed


def _diff_to_binary_labels(
    diff: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    # Binary activity: 1 iff abs(next_lambdas - lambdas) > threshold.
    return (diff > float(threshold)).astype(np.uint8)


def _diff_to_multiclass_labels(
    diff: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """
    Multiclass activity based on abs(next_lambdas - lambdas):
      0: diff < low_threshold
      1: low_threshold <= diff <= high_threshold
      2: diff > high_threshold
    """
    low = float(low_threshold)
    high = float(high_threshold)
    labels = np.zeros(diff.shape, dtype=np.uint8)
    labels[diff >= low] = 1
    labels[diff > high] = 2
    return labels


def add_lambda_activity_labels(
    dataset_path: Path,
    *,
    threshold: float = THRESHOLD_DEFAULT,
    low_threshold: float = LOW_THRESHOLD_DEFAULT,
    multiclass: bool = False,
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

        diff = np.abs(next_lambdas_ds[...] - lambdas_ds[...])

        if multiclass:
            lambda_activity = _diff_to_multiclass_labels(
                diff,
                low_threshold=low_threshold,
                high_threshold=threshold,
            )
            lambda_activity = _fill_lonely_zeros_multiclass(
                lambda_activity,
                min_nonzero_each_side=lonely_zero_context,
            )
        else:
            lambda_activity = _diff_to_binary_labels(diff, threshold=threshold)
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
        data_group[dataset_name].attrs["multiclass"] = bool(multiclass)
        data_group[dataset_name].attrs["threshold"] = float(threshold)
        data_group[dataset_name].attrs["low_threshold"] = float(low_threshold)

        if multiclass:
            data_group[dataset_name].attrs["description"] = (
                "Multiclass lambda activity label based on abs(next_lambdas - lambdas): "
                "0 if diff < low_threshold; 1 if low_threshold <= diff <= threshold; "
                "2 if diff > threshold."
            )
        else:
            data_group[dataset_name].attrs["description"] = (
                "Binary lambda activity label where abs(next_lambdas - lambdas) > threshold."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add lambda-activity labels to an HDF5 dataset. "
            "Binary mode uses abs(next_lambdas - lambdas) > threshold; "
            "multiclass mode additionally supports 3 classes (0/1/2)."
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
        help=(
            "Binary mode: activity threshold in abs(next_lambdas - lambdas) > threshold. "
            "Multiclass mode: upper cut (diff > threshold => class 2)."
        ),
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help=(
            "Enable multiclass activity labels. "
            "Classes: 0 (diff < low-threshold), 1 (low-threshold <= diff <= threshold), "
            "2 (diff > threshold)."
        ),
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=LOW_THRESHOLD_DEFAULT,
        help="Multiclass lower cut (diff < low-threshold => class 0).",
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
            "If a single 0 label is surrounded by this many consecutive non-zero labels "
            "on both sides in time, convert that 0 to the surrounding activity "
            "(binary: 0->1; multiclass: 0->max(1,2) in the neighborhood)."
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
        low_threshold=args.low_threshold,
        multiclass=args.multiclass,
        dataset_name=args.dataset_name,
        lonely_zero_context=args.lonely_zero_context,
        overwrite=overwrite_labels,
    )
    print(f"Added '{args.dataset_name}' (multiclass={args.multiclass}) to: {output_path}")


if __name__ == "__main__":
    main()

