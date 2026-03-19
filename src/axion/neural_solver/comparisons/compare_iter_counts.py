#!/usr/bin/env python3
"""Plot Newton iter_count per step for Axion vs HybridGPT logs."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
AXION_H5_PATH = REPO_ROOT / "data" / "logs" / "pendulum_AxionEngine.h5"
HYBRID_H5_PATH = REPO_ROOT / "data" / "logs" / "pendulum_HybridGPTEngine.h5"


def load_iter_count(h5_path: Path) -> np.ndarray:
    """Load iter_count dataset as a flat numpy array."""
    with h5py.File(h5_path, "r") as f:
        if "iter_count" not in f:
            available = ", ".join(sorted(f.keys()))
            raise KeyError(
                f"'iter_count' dataset not found in {h5_path}. Top-level keys: {available}"
            )
        iter_count = np.asarray(f["iter_count"][:]).reshape(-1)
    return iter_count


def main() -> None:
    if not AXION_H5_PATH.exists():
        raise FileNotFoundError(f"Missing file: {AXION_H5_PATH}")
    if not HYBRID_H5_PATH.exists():
        raise FileNotFoundError(f"Missing file: {HYBRID_H5_PATH}")

    axion_iter_count = load_iter_count(AXION_H5_PATH)
    hybrid_iter_count = load_iter_count(HYBRID_H5_PATH)
    axion_steps = np.arange(axion_iter_count.shape[0])
    hybrid_steps = np.arange(hybrid_iter_count.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(axion_steps, axion_iter_count, linewidth=1.5, label="AxionEngine")
    plt.plot(hybrid_steps, hybrid_iter_count, linewidth=1.5, label="HybridGPTEngine")
    plt.xlabel("Simulation step")
    plt.ylabel("Newton iter_count")
    plt.ylim(bottom=0)
    plt.title("Newton Iterations per Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
