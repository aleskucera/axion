# Introduction

**Axion Simulator** is differentiable simulator which leverages the GPU to make the simulation real-time and also physically accurate. We use mainly [NVIDIA Warp](https://github.com/NVIDIA/warp) framework to write a high-performance simulation that can run on GPU as well as CPU.

## Quickstart

To install Axion, we have to create python virtual environment. We highly recommend [uv](https://github.com/astral-sh/uv).

Once uv is installed, running Newton examples is straightforward:

```sh
# Clone the repository
git clone git@github.com:aleskucera/axion.git
cd axion

# Set up the uv environment for running examples
uv sync

# Run an example
uv run ball_bounce_example
```



