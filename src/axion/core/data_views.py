from dataclasses import dataclass
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import TypeVar

import warp as wp

from .engine_dims import EngineDimensions

# Generic type var to indicate this works for float arrays, vector arrays, etc.
T = TypeVar("T")


@dataclass(frozen=True)
class ConstraintView(Generic[T]):
    """
    Wrapper for constraint data that handles slicing for j/n/f parts.
    Supports specifying which axis contains the stacked constraints.
    """

    data: wp.array
    dims: EngineDimensions
    axis: int = -1  # Default to last dimension (works for flat 1D and stacked Batches)

    def _slice_at_axis(self, s: slice) -> Optional[wp.array]:
        """
        Slices self.data at self.axis using slice s.
        Replaces Ellipsis usage with explicit slice tuples for compatibility.
        """
        # 1. Return None if the slice is empty (e.g. no joints)
        if s.start == s.stop:
            return None

        # 2. Normalise bounds (handle negative axis)
        ndim = self.data.ndim
        target_axis = self.axis % ndim

        # 3. Construct explicit indexer tuple: (:, :, ..., s, ..., :)
        #    Pre-axis: slice(None) for 0 to target_axis
        #    At-axis:  s
        #    Post-axis: slice(None) for target_axis+1 to ndim

        pre_slice = (slice(None),) * target_axis
        post_slice = (slice(None),) * (ndim - target_axis - 1)

        indexer = pre_slice + (s,) + post_slice

        return self.data[indexer]

    @property
    def full(self) -> wp.array:
        return self.data

    @property
    def j(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_j)

    @property
    def ctrl(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_ctrl)

    @property
    def eq(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_eq)

    @property
    def n(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_n)

    @property
    def f(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_f)

    def zero_(self):
        self.data.zero_()


@dataclass(frozen=True)
class SystemView(Generic[T]):
    """
    Wraps (..., N_u + N_c) arrays.
    Always assumes the split dimension is the last one.
    """

    data: wp.array
    dims: EngineDimensions
    _d_spatial: Optional[wp.array] = None

    def _get_split_slice(self, start, stop) -> Tuple[slice]:
        """Helper to generate tuple slice for last dimension."""
        ndim = self.data.ndim
        # (:, :, ..., start:stop)
        return (slice(None),) * (ndim - 1) + (slice(start, stop),)

    @property
    def full(self) -> wp.array:
        return self.data

    @property
    def d(self) -> wp.array:
        """Dynamics part (..., :N_u)."""
        indexer = self._get_split_slice(None, self.dims.N_u)
        return self.data[indexer]

    @property
    def d_spatial(self) -> wp.array:
        """
        Dynamics part reinterpreted as spatial vectors.
        Shape change: (..., N_u) -> (..., N_b) [dtype=spatial]
        """
        if self._d_spatial is not None:
            return self._d_spatial

        d_data = self.d

        # Calculate new shape: preserve batch dims, set last dim to N_b
        base_shape = d_data.shape[:-1]
        new_shape = base_shape + (self.dims.N_b,)

        return wp.array(d_data, shape=new_shape, dtype=wp.spatial_vector)

    @property
    def c(self) -> ConstraintView[T]:
        """
        Constraint part (..., N_u:).
        Returns a View configured for axis=-1 (the natural ends of the system vector).
        """
        indexer = self._get_split_slice(self.dims.N_u, None)
        return ConstraintView(self.data[indexer], self.dims, axis=-1)

    def zero_(self):
        self.data.zero_()
