# Installation

This guide will help you install Axion and its dependencies on your system.

## Prerequisites

Before installing Axion, ensure you have the following:

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: Version 3.12 or higher
- **CUDA** (optional but recommended): Version 11.8 or higher for GPU acceleration
- **RAM**: Minimum 8GB, 16GB+ recommended for large simulations
- **GPU** (optional): NVIDIA GPU with compute capability 7.0+ for CUDA acceleration

### Check Your Environment

Verify your Python version:

```bash
python --version
# Should output: Python 3.12.x or higher
```

Check CUDA installation (if using GPU):

```bash
nvidia-smi
nvcc --version
```

## Installation with uv

We highly recommend using [uv](https://github.com/astral-sh/uv) for installing and managing Axion. It's a blazing-fast Python package installer and resolver written in Rust that makes dependency management simple and reliable.

### Step 1: Install uv

=== "Linux/macOS"

    ```bash
    # Install using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Or using pip
    pip install uv
    ```

=== "Windows"

    ```powershell
    # Using PowerShell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Or using pip
    pip install uv
    ```

=== "Using Homebrew"

    ```bash
    brew install uv
    ```

### Step 2: Clone the Repository

```bash
git clone https://github.com/aleskucera/axion.git
cd axion
```

### Step 3: Install Axion

```bash
# Install all dependencies and set up the environment
uv sync

# This creates a virtual environment and installs all dependencies
# in one command - no manual venv creation needed!
```

### Step 4: Verify Installation

Run a test example:

```bash
# Run the ball bounce example
uv run ball_bounce_example

# Or run any Python script with Axion
uv run python -m axion.examples.ball_bounce
```

## Why uv?

Using uv provides several advantages:

- **⚡ Speed**: 10-100x faster than pip for dependency resolution
- **🔒 Reliability**: Guaranteed reproducible installs with lock file
- **🎯 Simplicity**: Single command to set up everything
- **📦 Isolation**: Automatic virtual environment management
- **🔄 Consistency**: Same environment across all machines

## Running Examples with uv

Once installed, you can run any of the Axion examples:

```bash
# Ball bounce simulation
uv run ball_bounce_example

# Robot simulation
uv run helhest_example

# Simple robot example
uv run helhest_simple_example

# Collision primitives demo
uv run collision_primitives_example
```

## Dependency Overview

Axion's dependencies are managed through `pyproject.toml` and include:

| Package | Purpose |
|---------|---------|
| `warp-lang` | NVIDIA's differentiable graphics and physics library |
| `torch` | PyTorch for machine learning integration |
| `hydra-core` | Configuration management system |
| `h5py` | HDF5 file format for data logging |
| `trimesh` | 3D mesh processing and visualization |
| `scipy` | Scientific computing utilities |
| `nvtx` | NVIDIA profiling tools |

All dependencies are automatically installed when you run `uv sync`.

## Next Steps

Now that you have Axion installed:

1. Continue to the [Quick Start Guide](quickstart.md) to understand the basics
2. Follow the [First Simulation Tutorial](first-simulation.md) to create your own simulation
3. Explore the [User Guide](../user-guide/concepts.md) for advanced features

## Getting Help

If you encounter issues:

- Check [GitHub Issues](https://github.com/aleskucera/axion/issues)
- Contact: [kuceral4@fel.cvut.cz](mailto:kuceral4@fel.cvut.cz)

---

!!! success "Installation Complete!"
    You're now ready to start using Axion for differentiable physics simulations!

