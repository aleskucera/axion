import numpy as np
import scipy.optimize
import warp as wp

from .dense_matrix_utils import DenseMatrixMixin


class ScipySolverMixin(DenseMatrixMixin):

    def solve_scipy(
        self, method: str = "hybr", tolerance: float = 1e-8, max_iterations: int = 1000
    ) -> scipy.optimize.OptimizeResult:
        """
        Solve the full nonlinear system with SciPy.

        Args:
            method: SciPy optimization method
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations

        Returns:
            SciPy optimization result
        """

        def residual_function(x: np.ndarray) -> np.ndarray:
            return self._evaluate_residual(x)

        # Initial guess from current state
        x0 = np.concatenate([self._lambda.numpy(), self._body_qd.numpy().flatten()])

        # Solve with scipy.optimize.root
        result = scipy.optimize.root(
            residual_function,
            x0,
            method=method,
            options={"xtol": tolerance, "maxfev": max_iterations},
        )

        n_lambda = self.dims.con_dim
        self._lambda.assign(wp.from_numpy(result.x[:n_lambda].astype(np.float32)))
        body_qd_solution = result.x[n_lambda:][np.newaxis, :]
        self._body_qd.assign(wp.from_numpy(body_qd_solution.astype(np.float32)))

        return result

    def _evaluate_residual_old(self, x: np.ndarray) -> np.ndarray:
        """Evaluate residual function for SciPy solver."""
        n_lambda = self.dims.con_dim
        lambda_vals = x[:n_lambda]
        body_qd_vals = x[n_lambda:]

        # Store current state
        lambda_backup = wp.clone(self._lambda)
        body_qd_backup = wp.clone(self._body_qd)

        try:
            # Set state from input vector
            self._lambda.assign(lambda_vals)
            self._body_qd.assign(body_qd_vals)

            # Compute residuals
            self.update_system_values()

            # Extract system and compute residual
            self.update_dense_matrices()
            A, b = self.compute_system_matrix_numpy()

            # Combined residual: [constraint_residual; momentum_residual]
            constraint_residual = A @ lambda_vals - b
            momentum_residual = self._g.numpy() - self.J_dense.numpy().T @ lambda_vals

            return np.concatenate([constraint_residual, momentum_residual])

        finally:
            # Restore original state
            wp.copy(dest=self._lambda, src=lambda_backup)
            wp.copy(dest=self._body_qd, src=body_qd_backup)

    def _evaluate_residual(self, x: np.ndarray) -> np.ndarray:
        # x contains both lambda and body_qd
        n_lambda = self.dims.con_dim
        lambda_vals = x[:n_lambda]
        body_qd_vals = x[n_lambda:]

        # Store current state
        lambda_backup = wp.clone(self._lambda)
        body_qd_backup = wp.clone(self._body_qd)

        try:
            # Set state from input vector
            self._lambda.assign(lambda_vals)
            self._body_qd.assign(body_qd_vals)

            # Compute residuals
            self.update_system_values()

            # Residual is concatenation of g and h vector
            return np.concatenate([self._g.numpy(), self._h.numpy()])

        finally:
            # Restore original state
            wp.copy(dest=self._lambda, src=lambda_backup)
            wp.copy(dest=self._body_qd, src=body_qd_backup)
