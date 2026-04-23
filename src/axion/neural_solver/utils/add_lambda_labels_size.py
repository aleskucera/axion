"""
Add lambda-size binary labels to an HDF5 dataset.

Example:
python src/axion/neural_solver/utils/add_lambda_labels_size.py --input src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1.hdf5 --output src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1WithSizeLabels.hdf5 --threshold 500
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

THRESHOLD_DEFAULT = 500.0

def _resolve_data_group(h5_file: h5py.File) -> h5py.Group:
    if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
        return h5_file["data"]
    return h5_file


def _lambda_to_binary_labels(lambdas: np.ndarray, *, threshold: float) -> np.ndarray:
    # Size label: 1 iff current lambda is strictly above threshold.
    return (lambdas > float(threshold)).astype(np.uint8)


def add_lambda_labels_size(
    dataset_path: Path,
    *,
    threshold: float = THRESHOLD_DEFAULT,
    dataset_name: str = "lambda_size_labels",
    overwrite: bool = False,
) -> None:
    with h5py.File(dataset_path, "r+") as h5_file:
        data_group = _resolve_data_group(h5_file)

        if "lambdas" not in data_group:
            raise KeyError("Cannot compute lambda size labels, missing dataset: 'lambdas'")

        if dataset_name in data_group:
            if not overwrite:
                raise ValueError(
                    f"Dataset '{dataset_name}' already exists. "
                    "Use --overwrite to replace it."
                )
            del data_group[dataset_name]

        lambdas = data_group["lambdas"][...]
        labels = _lambda_to_binary_labels(lambdas, threshold=threshold)

        data_group.create_dataset(
            dataset_name,
            data=labels,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
        data_group[dataset_name].attrs["threshold"] = float(threshold)
        data_group[dataset_name].attrs["description"] = (
            "Binary lambda size label where label=1 if lambdas > threshold, else 0."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add lambda-size labels to an HDF5 dataset. "
            "Label is 1 when current lambda > threshold, otherwise 0."
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
        default=THRESHOLD_DEFAULT,
        help="Label threshold. Label is 1 when lambdas > threshold, else 0.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lambda_size_labels",
        help="Name of the output labels dataset.",
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

    add_lambda_labels_size(
        output_path,
        threshold=args.threshold,
        dataset_name=args.dataset_name,
        overwrite=overwrite_labels,
    )
    print(f"Added '{args.dataset_name}' to: {output_path}")


if __name__ == "__main__":
    main()
