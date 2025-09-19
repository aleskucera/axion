# Quick Start Guide

Get up and running with Axion in minutes! This guide assumes you've already [installed Axion](installation.md) using uv.

## Running Your First Simulation

```bash
# Run the ball bounce example
uv run ball_bounce_example
```

This will:

1. Run the physics simulation
2. Generate a `.usd` file with the simulation results

The USD (Universal Scene Description) file can be played back in any USD-compatible viewer.

## Available Examples

### Ball Bounce

```bash
uv run ball_bounce_example
```

Simple ball bouncing on ground - demonstrates basic rigid body dynamics and collision.

### Collision Primitives

```bash
uv run collision_primitives_example
```

Multiple collision shapes interacting - demonstrates different geometry types.

### Helhest Robot

```bash
uv run helhest_example
```

Articulated robot simulation - demonstrates joints and complex dynamics.

## Configuration Basics

Override settings using Hydra syntax:

```bash
# Run without rendering (faster)
uv run ball_bounce_example rendering=headless

# Change simulation duration
uv run ball_bounce_example simulation.duration_seconds=10.0

# Adjust physics accuracy
uv run ball_bounce_example engine.newton_iters=20
```

To know about more available parameters and their default values, run

```bash
uv run ball_bounce_example -h
```

The hydra configuration is in `src/axion/examples/conf`.

## Next Steps

- [Create Your First Simulation](first-simulation.md) - Build a custom simulation
- [User Guide](../user-guide/concepts.md) - Understand core concepts
- [Configuration System](../user-guide/configuration.md) - Learn Hydra configuration

