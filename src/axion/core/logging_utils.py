"""
Enhanced logging and debugging utilities for the NSN engine.
"""
from typing import Dict

import numpy as np

from .dense_matrix_utils import DenseMatrixMixin


class LoggingMixin(DenseMatrixMixin):
    """Enhanced mixin providing comprehensive logging functionality."""

    def setup_logging_matrices(self) -> None:
        """Initialize dense matrices for logging if logger is present."""
        if self.logger:
            self._ensure_dense_matrices_exist()

    def log_newton_state(self) -> None:
        """
        Log the current state of the Newton iteration.

        Args:
            iteration: Current Newton iteration number
        """
        if not self.logger:
            return

        # Update dense matrices
        self.update_dense_matrices()

        # Compute system matrices and residuals
        Hinv_np, J_np, C_np, g_np, h_np = self.get_dense_matrices_numpy()
        A, b = self.compute_system_matrix_numpy()

        # Compute various error metrics
        metrics = self._compute_convergence_metrics(A, b, g_np, h_np, J_np)

        # Log all data
        self._log_matrices(Hinv_np, J_np, C_np, g_np, h_np, A, b)
        self._log_metrics(metrics)

    def _compute_convergence_metrics(
        self,
        A: np.ndarray,
        b: np.ndarray,
        g_np: np.ndarray,
        h_np: np.ndarray,
        J_np: np.ndarray,
    ) -> Dict[str, float]:
        """Compute various convergence and error metrics."""
        lambda_np = self._lambda.numpy()

        # System residuals
        lambda_residual = A @ lambda_np - b
        momentum_residual = g_np - J_np.T @ lambda_np

        # Norms
        lambda_norm = np.linalg.norm(lambda_np)
        lambda_residual_norm = np.linalg.norm(lambda_residual)
        momentum_residual_norm = np.linalg.norm(momentum_residual)
        constraint_violation_norm = np.linalg.norm(h_np)

        # Condition numbers (with safeguards)
        try:
            A_condition = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            A_condition = float("inf")

        return {
            "lambda_norm": lambda_norm,
            "lambda_residual_norm": lambda_residual_norm,
            "momentum_residual_norm": momentum_residual_norm,
            "constraint_violation_norm": constraint_violation_norm,
            "total_residual_norm": lambda_residual_norm + momentum_residual_norm,
            "system_condition_number": A_condition,
            "max_lambda": np.max(np.abs(lambda_np)) if lambda_np.size > 0 else 0.0,
            "max_constraint_violation": np.max(np.abs(h_np)) if h_np.size > 0 else 0.0,
        }

    def _log_matrices(
        self,
        Hinv_np: np.ndarray,
        J_np: np.ndarray,
        C_np: np.ndarray,
        g_np: np.ndarray,
        h_np: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> None:
        """Log all system matrices."""
        self.logger.log_dataset("Hinv", Hinv_np)
        self.logger.log_dataset("J", J_np)
        self.logger.log_dataset("C", C_np)
        self.logger.log_dataset("g", g_np)
        self.logger.log_dataset("h", h_np)
        self.logger.log_dataset("A", A)
        self.logger.log_dataset("b", b)
        self.logger.log_dataset("lambda", self._lambda.numpy())

        # Log step information if available
        if hasattr(self, "_delta_lambda"):
            self.logger.log_dataset("delta_lambda", self._delta_lambda.numpy())

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log convergence metrics."""
        for key, value in metrics.items():
            self.logger.log_scalar(key, value)

    def log_simulation_summary(self, total_time: float, newton_stats: list) -> None:
        """Log summary statistics for the entire simulation step."""
        if not self.logger:
            return

        # Aggregate Newton iteration statistics
        total_newton_time = sum(stats.get("solve_time", 0) for stats in newton_stats)
        avg_linear_iters = np.mean(
            [stats.get("linear_iters", 0) for stats in newton_stats]
        )

        self.logger.log_scalar("total_simulation_time", total_time)
        self.logger.log_scalar("total_newton_time", total_newton_time)
        self.logger.log_scalar("avg_linear_iterations", avg_linear_iters)
        self.logger.log_scalar("newton_iterations_completed", len(newton_stats))
