<div align="center">
    <h1>Axion: A Differentiable Physics Simulator</h1>
</div>

**Axion** is a high-performance, differentiable physics simulator designed for robotics research, machine learning, and applications requiring accurate gradients through complex dynamics. Built on top of NVIDIA Warp, it leverages GPU acceleration and automatic differentiation to enable gradient-based optimization for physics-driven systems.

## 🚀 Overview

Differentiable simulators are crucial tools for bridging physics-based modeling with gradient-based optimization techniques common in machine learning and robotics. Axion provides a powerful and flexible simulation environment that addresses key challenges in the field:

### Core Capabilities

* **⚡ GPU Acceleration:** Built on NVIDIA Warp for parallel execution on modern GPUs, maximizing computational throughput
* **🔧 Flexible Configuration:** Hydra-based configuration system for easy experimentation and parameter tuning
* **📊 Comprehensive Logging:** HDF5-based logging system for detailed simulation data capture and analysis
* **🎮 Simulation Visualization:** Rendering with configurable frame rates

### Physics Features

* **Rigid Body Dynamics:** Robust simulation of articulated and unarticulated rigid body systems
* **Contact & Collision:** Advanced contact constraint handling with configurable stiffness and damping
* **Friction Modeling:** Coulomb friction implementation with customizable coefficients
* **Joint Constraints:** Support for various joint types in articulated systems
* **Adaptive Time Stepping:** Dynamic integration methods for optimal performance and stability

## 📦 Installation

### Prerequisites

* Python 3.12 or higher
* CUDA-capable GPU (recommended)
* NVIDIA Warp (automatically installed as dependency)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/aleskucera/axion.git
cd axion

# Install with pip
pip install -e .

# Or install with uv (if you have it)
uv pip install -e .
```

### Dependencies

Axion depends on the following key packages:

* `warp-lang`: NVIDIA's differentiable graphics and physics library
* `torch`: PyTorch for machine learning integration
* `hydra-core`: Configuration management
* `h5py`: HDF5 data logging
* `usd-core`: Universal Scene Description support
* `trimesh`: 3D mesh processing

## 🎯 Quick Start

### Running Examples

Axion comes with several pre-built examples that demonstrate its capabilities:

```bash
# Simple ball bouncing example
python -m axion.examples.ball_bounce

# Collision primitives demonstration
python -m axion.examples.collision_primitives

# Helhest robot simulation
python -m axion.examples.helhest

# Simplified Helhest example
python -m axion.examples.helhest_simple
```

## ⚙️ Configuration

Axion uses Hydra for flexible configuration management. Configuration files are organized in a modular structure:

```
conf/
├── config.yaml          # Main configuration
├── engine/             # Physics engine settings
│   ├── base.yaml
│   └── helhest.yaml
├── simulation/         # Simulation parameters
│   ├── base.yaml
│   └── helhest.yaml
├── rendering/          # Visualization settings
│   ├── 30_fps.yaml
│   └── headless.yaml
├── execution/          # Execution modes
│   ├── cuda_graph.yaml
│   └── no_graph.yaml
└── profiling/          # Performance profiling
    ├── disabled.yaml
    ├── timing.yaml
    └── hdf5_logging.yaml
```

### Configuration Override

You can override configuration parameters from the command line:

```bash
# Run with different frame rate
python -m axion.examples.ball_bounce rendering=headless

# Change engine parameters
python -m axion.examples.helhest engine=helhest simulation=helhest

# Enable profiling
python -m axion.examples.ball_bounce profiling=timing
```

## 🏗️ Architecture

### Project Structure

```
axion/
├── core/               # Core simulation engine
│   ├── engine.py      # Main physics engine
│   ├── engine_config.py
│   ├── engine_data.py
│   └── abstract_simulator.py
├── constraints/        # Physics constraints
│   ├── contact_constraint.py
│   ├── friction_constraint.py
│   ├── joint_constraint.py
│   └── dynamics_constraint.py
├── optim/             # Optimization algorithms
│   ├── cr.py          # Conjugate residual solver
│   ├── preconditioner.py
│   └── matrix_operator.py
├── logging/           # Data logging utilities
│   ├── hdf5_logger.py
│   └── hdf5_reader.py
├── types/             # Data structures
│   ├── contact_interaction.py
│   ├── joint_interaction.py
│   └── generalized_mass.py
└── examples/          # Example simulations
```

### Key Components

1. **Engine Core**: The `AxionEngine` class handles the main simulation loop, integrating Newton's method for solving nonlinear systems arising from implicit time integration.

2. **Constraint System**: Modular constraint handlers for different physics phenomena:
   * Contact constraints with configurable stiffness/damping
   * Friction constraints using Coulomb model
   * Joint constraints for articulated bodies
   * Dynamics constraints for Newton-Euler equations

3. **Optimization Layer**: Efficient linear solvers and preconditioners for handling large-scale systems:
   * Conjugate Residual (CR) solver
   * Matrix-free operators for memory efficiency
   * Jacobi preconditioning

4. **Logging Infrastructure**: Comprehensive data capture for analysis and debugging:
   * HDF5-based storage for efficient data management
   * Real-time performance metrics
   * Configurable logging levels and outputs

## 📊 Performance & Benchmarking

Axion includes benchmarking tools for performance analysis:

```bash
# Run constraint kernel benchmarks
python tests/contact_kernel_benchmark.py
python tests/dynamics_kernel_benchmark.py
python tests/friction_kernel_benchmark.py
python tests/joint_kernel_benchmark.py

# Combined kernel benchmark
python tests/combined_kernel_benchmark.py
```

### Performance Features

* **CUDA Graph Execution**: Optimized GPU kernel execution
* **Sparse Matrix Operations**: Efficient handling of large systems
* **Parallel Constraint Resolution**: GPU-accelerated constraint solving
* **Adaptive Timestep Control**: Dynamic adjustment for stability and speed

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_ncp.py -v
```

### Building Documentation

```bash
# Build with mkdocs
mkdocs build

# Serve locally
mkdocs serve
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

**Author:** Aleš Kučera  
**Email:** <kuceral4@fel.cvut.cz>  
**Institution:** Czech Technical University in Prague

---

<div align="center">
    <sub>Built with ❤️ for the robotics and machine learning community</sub>
</div>

