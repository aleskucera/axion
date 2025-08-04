import os
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import h5py
import numpy as np


class HDF5Logger:
    """
    A standalone, hierarchical logger for saving simulation data to an HDF5 file.
    """

    def __init__(self, filepath: str, mode: str = "w"):
        self.filepath = filepath
        self.mode = mode
        self._file = None
        self._current_scope = ""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if self._file is None:
            directory = os.path.dirname(self.filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._file = h5py.File(self.filepath, self.mode)
            print(f"HDF5Logger: Opened {self.filepath}")

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            print(f"HDF5Logger: Closed {self.filepath}")

    def scope(self, name: str):
        return HDF5Scope(self, name)

    def _get_current_group(self) -> h5py.Group:
        """Helper to get the h5py.Group object for the current scope."""
        if self._file is None:
            raise IOError("Logger is not open. Cannot get group.")
        if self._current_scope:
            return self._file.require_group(self._current_scope)
        return self._file  # Return the root group

    def log_dataset(
        self, name: str, data: np.ndarray, attributes: Dict[str, Any] = None
    ):
        """Logs a NumPy array as a dataset within the current scope."""
        group = self._get_current_group()
        dset = group.create_dataset(name, data=data, compression="gzip")
        if attributes:
            for key, val in attributes.items():
                dset.attrs[key] = val

    def log_scalar(self, name: str, data: Union[int, float, str]):
        """
        Logs a single value (int, float, or string) as a scalar dataset.

        In HDF5, strings are typically stored with UTF-8 encoding.
        For details, see the h5py documentation on strings. [docs.h5py.org](https://docs.h5py.org/en/latest/strings.html)

        Args:
            name (str): The name of the dataset.
            data (Union[int, float, str]): The single value to save.
        """
        # h5py handles the conversion of basic Python types automatically.
        # So we can just pass them to create_dataset.
        group = self._get_current_group()
        group.create_dataset(name, data=data)

    def log_attribute(self, name: str, value: Any, target_path: str = "."):
        """
        Logs a piece of metadata as an attribute to a group or dataset.

        Args:
            name (str): The name of the attribute.
            value: The value of the attribute (e.g., a string, int, float).
            target_path (str): Relative path to a child group/dataset to attach to.
                               Defaults to '.', which means the current scope/group.
        """
        group = self._get_current_group()
        target_obj = group[target_path]
        target_obj.attrs[name] = value


class HDF5Scope:
    def __init__(self, logger: HDF5Logger, name: str):
        self._logger = logger
        self._name = name
        self._previous_scope = None

    def __enter__(self):
        self._previous_scope = self._logger._current_scope
        self._logger._current_scope = os.path.join(
            self._logger._current_scope, self._name
        ).replace("\\", "/")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logger._current_scope = self._previous_scope


class HDF5Reader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Log file not found: {self.filepath}")
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

    def close(self):
        if self._file is not None:
            self._file.close()

    def get_dataset(self, path: str) -> np.ndarray:
        try:
            return self._file[path][()]
        except KeyError:
            raise KeyError(f"Dataset not found at path: {path}")

    def get_scalar(self, path: str) -> Union[int, float, str]:
        """Retrieves a scalar dataset and returns it as a native Python type."""
        scalar = self.get_dataset(path)
        if isinstance(scalar, np.ndarray):
            if scalar.size != 1:
                raise ValueError(
                    f"Dataset at '{path}' is not a scalar (size={scalar.size})."
                )
            return scalar.item()
        return scalar

    def get_attribute(self, path: str, attr_name: str) -> Any:
        """Retrieves an attribute from a group or dataset."""
        try:
            return self._file[path].attrs[attr_name]
        except KeyError:
            raise KeyError(f"Object '{path}' or attribute '{attr_name}' not found.")

    def list_groups(self, path: str = "/") -> List[str]:
        # ... (this method is the same) ...
        return [
            key for key, val in self._file[path].items() if isinstance(val, h5py.Group)
        ]

    def list_datasets(self, path: str = "/") -> List[str]:
        return [
            key
            for key, val in self._file[path].items()
            if isinstance(val, h5py.Dataset)
        ]

    def list_attributes(self, path: str = "/") -> List[str]:
        """Lists all attribute names on a given group or dataset."""
        return list(self._file[path].attrs.keys())

    def print_tree(self):
        print(f"File Tree for {self.filepath}:")
        self._file.visititems(self._print_node)

    def _print_node(self, name: str, obj: h5py.HLObject):
        depth = name.count("/")
        indent = "    " * depth
        base_name = os.path.basename(name) if name else "/"

        if isinstance(obj, h5py.Dataset):
            print(
                f"{indent}├── {base_name} (Dataset: shape={obj.shape}, dtype={obj.dtype})"
            )
        else:  # It's a group
            print(f"{indent}└── {base_name} (Group)")

        # Print attributes for this object
        if obj.attrs:
            attr_indent = indent + "    " + "|"
            for key, val in obj.attrs.items():
                print(f"{attr_indent}  @{key}: {val}")


if __name__ == "__main__":
    """
    This demo shows how to use the enhanced logger and reader.
    """
    DEMO_FILENAME = "demo_log_with_scalars.h5"

    def run_logging_demo():
        print("--- Running Logging Demo ---")
        with HDF5Logger(DEMO_FILENAME) as logger:
            # Set top-level attributes for the whole file
            logger.log_attribute("simulation_name", "Helhest Demo")
            logger.log_attribute("version", "1.1")

            for t in range(2):
                with logger.scope(f"timestep_{t:02d}"):
                    # Log an attribute specific to this timestep
                    logger.log_attribute("dt", 0.016)

                    # Log a single float and a single string as SCALAR DATASETS
                    logger.log_scalar("residual_norm", 1.23e-4 * (t + 1))
                    logger.log_scalar("status_message", f"Timestep {t} completed.")

                    fake_J = np.random.rand(20, 60)
                    logger.log_dataset("J", fake_J)

                    # Attach an attribute DIRECTLY to the 'J' dataset
                    logger.log_attribute(
                        "description", "This is the Jacobian matrix.", target_path="J"
                    )

        print("--- Logging Demo Finished ---")

    def run_reading_demo():
        print("\n--- Running Reading Demo ---")
        with HDF5Reader(DEMO_FILENAME) as reader:
            # The enhanced tree view will now show attributes
            reader.print_tree()

            # Retrieve different types of data
            print("\n--- Retrieving Data ---")

            # 1. Get a top-level attribute
            sim_name = reader.get_attribute("/", "simulation_name")
            print(f"Retrieved root attribute 'simulation_name': {sim_name}")

            # 2. Get an attribute from a group
            dt_val = reader.get_attribute("timestep_01", "dt")
            print(f"Retrieved group attribute 'dt' from timestep_01: {dt_val}")

            # 3. Get an attribute from a dataset
            desc = reader.get_attribute("timestep_01/J", "description")
            print(f"Retrieved dataset attribute 'description' from J: {desc}")

            # 4. Get a scalar dataset
            status = reader.get_scalar("timestep_01/status_message")
            res_norm = reader.get_scalar("timestep_01/residual_norm")
            print(f"Retrieved scalar 'status_message': {status}")
            print(f"Retrieved scalar 'residual_norm': {res_norm:.6f}")

            # 5. List attributes on an object
            attrs_on_j = reader.list_attributes("timestep_01/J")
            print(f"Attributes on timestep_01/J: {attrs_on_j}")

        print("--- Reading Demo Finished ---")

    run_logging_demo()
    run_reading_demo()
