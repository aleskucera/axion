import os
from pathlib import PurePosixPath
from typing import Any
from typing import Dict
from typing import Union

import h5py
import numpy as np
import warp as wp
from warp.codegen import Struct as WpStruct


class HDF5Logger:
    def __init__(self, filepath: str, mode: str = "w"):
        self.filepath = filepath
        self.mode = mode
        self._file: h5py.File | None = None
        self._scope_stack: list[str] = [""]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def _current_scope(self) -> str:
        """The current path inside the HDF5 file, dynamically joined."""
        return str(PurePosixPath(*self._scope_stack))

    def _push_scope(self, name: str):
        """Adds a new level to the scope stack."""
        self._scope_stack.append(name)

    def _pop_scope(self):
        """Removes the most recent level from the scope stack."""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()

    def open(self):
        # Return if the _file is initialized
        if self._file is not None:
            return

        # Make sure that the directory exists
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Open the file
        self._file = h5py.File(self.filepath, self.mode)

        print(f"HDF5Logger: Opened {self.filepath}")

    def close(self):
        # Make sure that the _file is not already closed
        if self._file is None:
            return

        # Close the file
        self._file.close()
        self._file = None

        print(f"HDF5Logger: Closed {self.filepath}")

    def scope(self, name: str):
        """Creates a new hierarchical group within the HDF5 file."""
        return HDF5Scope(self, name)

    def _get_current_group(self) -> h5py.Group:
        """Helper to get the h5py.Group object for the current scope."""
        if self._file is None:
            raise IOError("Logger is not open. Cannot get group.")

        # Use the property to get the path string
        path = self._current_scope
        if path and path != ".":
            return self._file.require_group(path)
        return self._file  # Return the root group

    def log_np_dataset(self, name: str, data: np.ndarray, attributes: Dict[str, Any] = None):
        """Logs a NumPy array as a dataset within the current scope."""
        group = self._get_current_group()
        dset = group.create_dataset(name, data=data, compression="gzip")
        if not attributes:
            return

        for key, val in attributes.items():
            dset.attrs[key] = val

    def log_wp_dataset(self, name: str, data: "wp.array", attributes: Dict[str, Any] = None):
        """Logs a Warp array as a dataset within the current scope."""
        if isinstance(data.dtype, WpStruct):
            print(
                f"WARNING: 'log_wp_dataset' called on struct array '{name}'. "
                f"Use 'log_struct_array' for better organization."
            )
        self.log_np_dataset(name, data.numpy(), attributes)

    def _log_struct_recursive(self, struct_np: np.ndarray):
        """
        Recursively traverses a NumPy structured array and logs its fields.
        If a field is itself a struct, it creates a new group and recurses.
        """
        # A NumPy array is structured if its dtype has a .names attribute.
        if struct_np.dtype.names is None:
            return

        for field_name in struct_np.dtype.names:
            component_data = struct_np[field_name]

            # Check if this component is ALSO a structured array (i.e., a nested struct)
            if component_data.dtype.names:
                with self.scope(field_name):
                    # Recursive call to handle the nested structure
                    self._log_struct_recursive(component_data)
            else:
                # This is a primitive field, log it directly as a dataset
                self.log_np_dataset(name=field_name, data=component_data)

    def log_struct_array(self, name: str, data: wp.array, attributes: Dict[str, Any] = None):
        """
        Logs a wp.array of structs, correctly handling nested structs by
        recursively creating groups for each level of nesting.
        """
        if not isinstance(data.dtype, WpStruct):
            print(
                f"Warning: 'log_struct_array' called on a non-struct array '{name}'."
                f"Redirecting to 'log_wp_dataset'."
            )
            self.log_wp_dataset(name, data, attributes)
            return

        # The main entry point for logging a struct array.
        # It creates the top-level group and kicks off the recursive process.
        with self.scope(name):
            if attributes:
                group = self._get_current_group()
                for key, val in attributes.items():
                    group.attrs[key] = val

            # Convert to NumPy and start the recursive logging
            struct_np = data.numpy()
            self._log_struct_recursive(struct_np)

    def log_scalar(self, name: str, data: Union[int, float, str]):
        group = self._get_current_group()
        group.create_dataset(name, data=data)

    def log_attribute(self, name: str, value: Any, target_path: str = "."):
        group = self._get_current_group()
        target_obj = group[target_path]
        target_obj.attrs[name] = value


class HDF5Scope:
    """A context manager to handle nested groups in the HDF5Logger."""

    def __init__(self, logger: HDF5Logger, name: str):
        self._logger = logger
        self._name = name

    def __enter__(self):
        self._logger._push_scope(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logger._pop_scope()
