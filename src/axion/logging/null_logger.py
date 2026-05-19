from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import warp as wp


class _NullScope:
    """A dummy context manager that does nothing, for use by NullLogger."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class NullLogger:
    """
    A logger that conforms to the HDF5Logger interface but performs no actions.
    This is an implementation of the Null Object pattern, allowing you to
    unconditionally call logging methods without checking for None.
    """

    def __init__(self, *args, **kwargs):
        """Accepts any arguments but ignores them."""
        self._null_scope = _NullScope()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def scope(self, name: str):
        """Returns a context manager that does nothing."""
        return self._null_scope

    def log_np_dataset(self, name: str, data: np.ndarray, attributes: Dict[str, Any] = None):
        """Does nothing."""
        pass

    def log_wp_dataset(self, name: str, data: wp.array, attributes: Dict[str, Any] = None):
        """Does nothing."""
        pass

    def log_struct_array(self, name: str, data: wp.array, attributes: Dict[str, Any] = None):
        """Does nothing."""
        pass

    def log_scalar(self, name: str, data: Union[int, float, str]):
        """Does nothing."""
        pass

    def log_attribute(self, name: str, value: Any, target_path: str = "."):
        """Does nothing."""
        pass
