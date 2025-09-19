# Welcome to Axion

---

**Axion** is a cutting-edge differentiable physics simulator that bridges the gap between physical simulation and gradient-based optimization. Built on NVIDIA Warp, it provides researchers and engineers with a powerful tool for developing and testing algorithms in robotics, machine learning, and computational physics.

### Why Axion?

Modern robotics and machine learning applications require physics simulators that are not only fast and accurate but also differentiable. Axion addresses this need by providing:

- **🚀 GPU Acceleration** - Leverage CUDA for real-time or faster-than-real-time simulation
- **🔄 Differentiable Simulation** - Compute gradients through complex physics for optimization
- **⚙️ Flexible Architecture** - Modular design allows easy extension and customization
- **📊 Rich Data Logging** - Comprehensive HDF5-based logging for analysis and debugging

## Quick Navigation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started/installation.md)**
  
    New to Axion? Start here with installation and your first simulation

- :material-book-open-variant: **[User Guide](user-guide/concepts.md)**
  
    Learn core concepts and how to use Axion effectively

- :material-school: **[Tutorials](tutorials/ball-bounce.md)**
  
    Step-by-step guides for common simulation scenarios

- :material-api: **[API Reference](api/engine.md)**
  
    Detailed documentation of all classes and functions

</div>

## Key Features

### Physics Capabilities

Axion implements state-of-the-art physics simulation techniques:

- **Rigid Body Dynamics** - Fast and stable simulation of articulated and free-floating bodies
- **Contact & Collision** - Robust handling with configurable stiffness and damping parameters
- **Friction Models** - Accurate Coulomb friction for realistic contact interactions
- **Joint Constraints** - Support for revolute joints

### Technical Features

- **Differentiable Simulation** - Full gradient computation through the simulation pipeline
- **Constraint-Based Physics** - Modern constraint solver for accurate dynamics
- **Matrix-Free Solvers** - Memory-efficient algorithms for large-scale systems
- **Hydra Configuration** - Flexible experiment management and parameter sweeping
- **CUDA Graph Optimization** - Minimized kernel launch overhead for maximum GPU utilization

## Getting Help

- **📚 [User Guide](user-guide/concepts.md)** - Comprehensive guide to using Axion
- **🎓 [Tutorials](tutorials/ball-bounce.md)** - Learn by example
- **📖 [API Reference](api/engine.md)** - Detailed API documentation
- **🐛 [Issue Tracker](https://github.com/aleskucera/axion/issues)** - Report bugs or request features
- **📧 [Contact](mailto:kuceral4@fel.cvut.cz)** - Reach out directly

## Contributing

We welcome contributions from the community! Whether it's bug fixes, new features, or documentation improvements, please check our [Contributing Guide](developer/contributing.md) to get started.

## License

Axion is open-source software licensed under the MIT License. See the LICENSE file for details.

---

<div align="center">
    <p><strong>Ready to start simulating?</strong></p>
    <a href="getting-started/installation/" class="md-button md-button--primary">Get Started →</a>
</div>

