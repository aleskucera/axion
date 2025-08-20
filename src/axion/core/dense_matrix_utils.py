from typing import Tuple

import numpy as np
import warp as wp
from axion.types import GeneralizedMass


@wp.kernel
def update_J_dense(
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Output array
    J_dense: wp.array(dtype=wp.float32, ndim=2),
):
    constraint_idx = wp.tid()

    body_a = constraint_body_idx[constraint_idx, 0]
    body_b = constraint_body_idx[constraint_idx, 1]

    J_ia = J_values[constraint_idx, 0]
    J_ib = J_values[constraint_idx, 1]

    if body_a >= 0:
        body_idx = body_a * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, constraint_idx, body_idx + st_i, J_ia[st_i])

    if body_b >= 0:
        body_idx = body_b * 6
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            wp.atomic_add(J_dense, constraint_idx, body_idx + st_i, J_ib[st_i])


@wp.kernel
def update_Hinv_dense_kernel(
    gen_inv_mass: wp.array(dtype=GeneralizedMass),
    H_dense: wp.array(dtype=wp.float32, ndim=2),
):
    body_idx = wp.tid()

    if body_idx >= gen_inv_mass.shape[0]:
        return

    # Angular part, write the tensor of inertia inverse
    for i in range(wp.static(3)):
        for j in range(wp.static(3)):
            st_i = wp.static(i)
            st_j = wp.static(j)
            h_row = body_idx * 6 + st_i
            h_col = body_idx * 6 + st_j
            body_I_inv = gen_inv_mass.inertia[body_idx]
            H_dense[h_row, h_col] = body_I_inv[st_i, st_j]

    # Linear part, write the mass inverse
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        h_row = body_idx * 6 + 3 + st_i
        h_col = body_idx * 6 + 3 + st_i
        H_dense[h_row, h_col] = gen_inv_mass.m[body_idx]


@wp.kernel
def update_C_dense_kernel(
    C_values: wp.array(dtype=wp.float32),
    C_dense: wp.array(dtype=wp.float32, ndim=2),
):
    constraint_idx = wp.tid()
    if constraint_idx >= C_values.shape[0]:
        return

    # Fill the diagonal of the constraint matrix C_dense
    C_value = C_values[constraint_idx]
    C_dense[constraint_idx, constraint_idx] = C_value


class DenseMatrixMixin:
    """Mixin providing dense matrix operations for NSN engine components."""

    def _ensure_dense_matrices_exist(self):
        """Lazy initialization of dense matrices."""
        if not hasattr(self, "Hinv_dense"):
            self.Hinv_dense = wp.zeros(
                (self.dyn_dim, self.dyn_dim), dtype=wp.float32, device=self.device
            )
        if not hasattr(self, "J_dense"):
            self.J_dense = wp.zeros(
                (self.con_dim, self.dyn_dim), dtype=wp.float32, device=self.device
            )
        if not hasattr(self, "C_dense"):
            self.C_dense = wp.zeros(
                (self.con_dim, self.con_dim), dtype=wp.float32, device=self.device
            )

    def update_dense_matrices(self, synchronize: bool = True) -> None:
        """
        Update all dense matrices from sparse representations.

        Args:
            synchronize: Whether to synchronize GPU after updates
        """
        self._ensure_dense_matrices_exist()

        # Clear matrices
        self.Hinv_dense.zero_()
        self.J_dense.zero_()
        self.C_dense.zero_()

        # Update H^-1 (inverse mass matrix)
        wp.launch(
            kernel=update_Hinv_dense_kernel,
            dim=self.N_b,
            inputs=[self.gen_inv_mass],
            outputs=[self.Hinv_dense],
            device=self.device,
        )

        # Update J (constraint Jacobian)
        wp.launch(
            kernel=update_J_dense,
            dim=self.con_dim,
            inputs=[
                self._constraint_body_idx,
                self._J_values,
            ],
            outputs=[self.J_dense],
            device=self.device,
        )

        # Update C (compliance matrix)
        wp.launch(
            kernel=update_C_dense_kernel,
            dim=self.con_dim,
            inputs=[self._C_values],
            outputs=[self.C_dense],
            device=self.device,
        )

        if synchronize:
            wp.synchronize()

    def get_dense_matrices_numpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all dense matrices as numpy arrays.

        Returns:
            Tuple of (Hinv, J, C, g, h) as numpy arrays
        """
        return (
            self.Hinv_dense.numpy(),
            self.J_dense.numpy(),
            self.C_dense.numpy(),
            self._g.numpy(),
            self._h.numpy(),
        )

    def compute_system_matrix_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full system matrix A and RHS vector b in numpy.

        Returns:
            Tuple of (A, b) where A = J*H^-1*J^T + C, b = J*H^-1*g - h
        """
        Hinv_np, J_np, C_np, g_np, h_np = self.get_dense_matrices_numpy()

        A = J_np @ Hinv_np @ J_np.T + C_np
        b = J_np @ Hinv_np @ g_np - h_np

        return A, b
