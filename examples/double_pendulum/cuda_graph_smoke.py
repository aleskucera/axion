"""
CUDA-graph smoke test for GPTEngine + FastNeuralPredictor + TensorRT.

Runs `engine.step(...)` eagerly a few times, captures one step with
`wp.ScopedCapture()`, then replays the captured graph 1000 times. This is the
fastest correctness gate when iterating on the capture-safe predictor /
engine plumbing — much faster than spinning up the full Hydra-based
simulator from `GPTEngine_example.py`.

Pre-requisites:
- `USE_TENSORRT_ENGINE = True` in `src/axion/core/gpt_engine.py`.
- A built `.plan` + `.engine_meta.pt` next to the checkpoint pointed to by
  `NN_PENDULUM_PT_PATH` (see `docs/torch_to_tensorrt_conversion.md`).

Run from the repo root:

    python examples/double_pendulum/cuda_graph_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
import newton

# Make sure `examples/double_pendulum/` is on sys.path so we can pull in the
# pendulum model definition without depending on Hydra-style packaging.
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from pendulum_articulation_definition import build_pendulum_model
from pendulum_utils import (
    generalized_to_maximal,
    set_tilted_plane_from_coefficients,
)

from axion.core.engine_config import GPTEngineConfig
from axion.core.gpt_engine import GPTEngine, USE_TENSORRT_ENGINE
from axion.logging import NullLogger


N_WARMUP = 12          # > T (=10) so the ring buffer has shifted past prewarm.
N_CAPTURE_REPLAYS = 1000


def main() -> int:
    if not USE_TENSORRT_ENGINE:
        print(
            "ERROR: USE_TENSORRT_ENGINE = False in src/axion/core/gpt_engine.py. "
            "CUDA-graph capture only works against the TensorRT path. "
            "Flip the toggle (and make sure .plan / .engine_meta.pt are built) "
            "before running this smoke test."
        )
        return 1

    device = "cuda:0"

    # 1. Build the model (with the same tilted plane the example uses, just so
    #    the engine sees realistic-ish contacts during warmup).
    print("Building pendulum model...")
    model = build_pendulum_model(num_worlds=1, device=device)
    set_tilted_plane_from_coefficients(
        model, 0.0, 0.0, 1.0, 0.0, world_idx=0
    )

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    generalized_to_maximal(model, state_in, q0=0.5, q1=-0.3, qd0=1.0, qd1=-2.0)

    # 2. Build the engine. Uses GPTEngine's configured TRT paths.
    print("Building GPTEngine (loads TensorRT plan)...")
    logger = NullLogger()
    engine = GPTEngine(model=model, logger=logger, config=GPTEngineConfig())

    dt = 1.0 / 60.0

    # Newton collide pass for a Contacts object the engine expects.
    contacts = model.collide(state_in)

    # 3. Prewarm: seed the predictor's history ring buffer eagerly.
    print("Pre-warming predictor history buffer...")
    engine.prewarm(state_in, contacts, dt)

    # 4. Eager warmup: run a few steps to verify the path works at all and to
    #    let the contact set + state evolve to something realistic.
    print(f"Running {N_WARMUP} eager warmup steps...")
    for i in range(N_WARMUP):
        engine.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

    # Snapshot the post-eager state on the host for the parity comparison.
    eager_q = state_in.joint_q.numpy().copy()
    eager_qd = state_in.joint_qd.numpy().copy()
    print(f"  eager joint_q  = {eager_q}")
    print(f"  eager joint_qd = {eager_qd}")

    # 5. Capture one step. This is the moment of truth for the
    #    capture-safety of the predictor + engine plumbing.
    print("Capturing CUDA graph for a single engine.step()...")
    with wp.ScopedCapture() as cap:
        engine.step(state_in, state_out, control, contacts, dt)
    print("Capture succeeded.")

    # 6. Replay the captured graph many times to confirm it doesn't blow up.
    print(f"Replaying captured graph {N_CAPTURE_REPLAYS} times...")
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_CAPTURE_REPLAYS):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    elapsed = time.perf_counter() - t0
    step_us = elapsed * 1e6 / N_CAPTURE_REPLAYS
    print(
        f"Replayed {N_CAPTURE_REPLAYS} steps in {elapsed:.3f}s "
        f"({step_us:.1f} us/step)."
    )

    # Show final post-replay state — should diverge from `eager_q` because we
    # ran another N_CAPTURE_REPLAYS + 1 steps after the eager snapshot.
    out_state = state_out  # last write target across replays
    print(f"  final joint_q  = {out_state.joint_q.numpy()}")
    print(f"  final joint_qd = {out_state.joint_qd.numpy()}")

    # Basic sanity: no NaNs/Infs in the output.
    qf = out_state.joint_q.numpy()
    qdf = out_state.joint_qd.numpy()
    if not (np.all(np.isfinite(qf)) and np.all(np.isfinite(qdf))):
        print("ERROR: NaN/Inf in final state after graph replay.")
        return 2

    print("OK: CUDA-graph capture + replay completed without errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
