from dataclasses import dataclass
from typing import Generic
from typing import Optional
from typing import TypeVar

import warp as wp

from .engine_dims import EngineDimensions

# Generic type var to indicate this works for float arrays, vector arrays, etc.
T = TypeVar("T")


@dataclass
class ConstraintView(Generic[T]):
    """
    A lightweight wrapper around a Warp array that provides automatic
    slicing for Joint (j), Normal (n), and Friction (f) constraints.
    """

    data: wp.array
    dims: EngineDimensions

    @property
    def full(self) -> wp.array:
        """Returns the underlying raw array."""
        return self.data

    @property
    def j(self) -> Optional[wp.array]:
        """Slice for Joint constraints."""
        return self.data[self.dims.slice_j] if self.dims.N_j > 0 else None

    @property
    def n(self) -> Optional[wp.array]:
        """Slice for Normal contact constraints."""
        return self.data[self.dims.slice_n] if self.dims.N_n > 0 else None

    @property
    def f(self) -> Optional[wp.array]:
        """Slice for Friction constraints."""
        return self.data[self.dims.slice_f] if self.dims.N_f > 0 else None

    def zero_(self):
        """Helper to clear the underlying data."""
        self.data.zero_()


@dataclass
class SystemView(Generic[T]):
    data: wp.array
    dims: EngineDimensions

    @property
    def full(self) -> wp.array:
        return self.data

    @property
    def d(self) -> wp.array:
        """Dynamics part (first N_u elements)."""
        return self.data[: self.dims.N_u]

    @property
    def d_spatial(self) -> wp.array:
        """
        Dynamics part viewed as spatial vectors (N_b, 6).
        Useful for h_d_v.
        """
        # Matches your original h_d_v logic
        return wp.array(self.d, shape=self.dims.N_b, dtype=wp.spatial_vector)

    @property
    def c(self) -> ConstraintView:
        """
        Constraint part (remaining N_c elements).
        Returns a ConstraintView so you can do .c.j, .c.n, etc.
        """
        return ConstraintView(self.data[self.dims.N_u :], self.dims)

    def zero_(self):
        self.data.zero_()
