# PyTorch → ONNX → TensorRT Conversion

This document covers:

1. What model changes were made to prepare the trained PyTorch model for ONNX export.
2. How the export script works.
3. The end-to-end reproducible workflow from training to a TensorRT engine.

---

## 1. Model Changes for ONNX Export Readiness

Two new "fast" inference modules mirror the training-time modules with the changes needed for clean ONNX tracing and TensorRT compatibility. The original files are left **untouched** and continue to be used for training.

| Training (unchanged) | Inference (new) |
|---|---|
| `models/model_transformer.py` | `models/fast_model_transformer.py` |
| `models/mse_model.py` | `models/fast_mse_model.py` |

All parameter names are identical between the original and fast versions, so weights from a trained checkpoint transfer cleanly via `load_state_dict`.

---

### 1.1 `fast_model_transformer.py` — GPT changes

#### Flash attention removed

The training model uses `torch.nn.functional.scaled_dot_product_attention` (Flash Attention) when available:

```python
# model_transformer.py (training)
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Flash attention decomposes inconsistently across ONNX exporters and TRT builds. The fast model always runs the explicit matmul path and the `flash` flag is removed entirely.

#### Causal mask buffer always registered

In the training model, the `attn.bias` causal mask buffer is only registered when flash is *unavailable*. Since all modern training runs use flash attention (PyTorch >= 2.0), this buffer is **never present in saved checkpoints**. The fast model always registers it:

```python
# fast_model_transformer.py
self.register_buffer(
    "bias",
    torch.tril(torch.ones(config.block_size, config.block_size)).view(
        1, 1, config.block_size, config.block_size
    ),
)
```

When loading weights from a checkpoint into `FastMSEModel`, the missing `attn.bias` keys are explicitly allowed — they are re-derived from `block_size` and require no stored values.

#### `masked_fill(-inf)` replaced with additive mask

The training model's fallback path uses:

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

`-inf` causes NaN propagation through softmax under FP16 precision and is not reliably handled by the TRT ONNX parser. The fast model uses a large-negative additive mask instead:

```python
mask = self.bias[:, :, :T, :T]
att = att + (mask == 0) * -1.0e9   # FP32 safe; use -1e4 for pure FP16
```

#### Position IDs precomputed as a buffer

The training model calls `torch.arange(0, t, device=device)` inside every forward pass. This traces as a dynamic `Range` op that some TRT engine builds reject. The fast model pre-registers the indices:

```python
self.register_buffer("pos_ids", torch.arange(config.block_size, dtype=torch.long), persistent=False)
# in forward:
pos = self.pos_ids[:t]
```

#### Training-only methods stripped

`from_pretrained`, `configure_optimizers`, `estimate_mfu`, `generate`, and `crop_block_size` are removed to keep the traceable graph minimal and dependency-free.

---

### 1.2 `fast_mse_model.py` — MSEModel changes

#### Dict input replaced with a single tensor

The training `MSEModel.forward` accepts a Python dict and iterates over named keys:

```python
# mse_model.py (training)
def forward(self, input_dict, ...):
    features = self._extract_input_features(input_dict)  # iterates dict
```

TensorRT cannot accept Python dict inputs. The fast model takes a single pre-concatenated tensor:

```python
# fast_mse_model.py
def forward(self, x):  # x: (B, T, total_low_dim)
```

The caller is responsible for concatenating all `inputs.low_dim` keys **in the order listed in `cfg.yaml`** before calling the model. For the Pendulum model this order is:

```
states_embedding | contact_normals | contact_points_1 | contact_depths | gravity_dir
```

(total: 4 + 12 + 12 + 4 + 3 = **35** features)

#### RMS normalization removed

The training model optionally normalizes inputs and outputs using running mean/std statistics (`normalize_input: true`). These depend on non-traceable Python state. The fast model strips all normalization logic — the caller must apply the same normalization externally before feeding tensors to the engine.

#### `input_sample` probing removed

The training model constructor probes tensor shapes from a sample dict at build time. The fast model constructor takes explicit integer dims (`total_low_dim`, `state_output_dim`, `lambda_output_dim`) instead.

#### `from_mse_model` classmethod

`FastMSEModel.from_mse_model(mse_model)` is the primary way to construct a fast model from a trained checkpoint:

1. Introspects encoder, transformer, and head configs from the source model.
2. Handles the identity-encoder case (`layer_sizes: []`) where `MLPBase.body` is an empty `nn.Sequential`.
3. Transfers all weights via `load_state_dict(strict=False)`.
4. Allows missing `*.attn.bias` keys (absent from flash-trained checkpoints) and regenerates them from `block_size`.
5. Raises on any other missing key (genuine weight mismatch).

---

## 2. How `export_to_onnx.py` Works

File: `fast_inference/export_to_onnx.py`

The script is designed to be run with **F5 in VS Code / Cursor** — no CLI arguments needed. Edit the four constants at the top of the file to point at a checkpoint, then run.

### 2.1 Configuration constants

```python
MODEL_PT     = "src/.../nn/best_valid_valid_model.pt"  # path to trained checkpoint
NN_MODEL_CFG = "src/.../cfg.yaml"                       # training config (optional but recommended)
BATCH_SIZE   = 1        # number of robots/worlds per forward pass, baked into the graph
DEVICE       = "cuda:0" # use "cpu" if no GPU is available
OUTPUT       = None     # None → writes <checkpoint_stem>.onnx beside the .pt file
```

### 2.2 Step-by-step execution

**Step 1 — Load checkpoint**

```python
payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
mse_model = payload[0]   # format: [MSEModel, robot_name]
```

The trainer saves checkpoints as `[model, robot_name]`. Both list and bare-object formats are handled.

**Step 2 — Convert to `FastMSEModel`**

```python
fast_model = FastMSEModel.from_mse_model(mse_model, device=device)
fast_model.eval()
```

All weights are copied; the export-incompatible ops are replaced with their fast-model equivalents.

**Step 3 — Cross-validate against yaml**

If `NN_MODEL_CFG` is set, the script checks:
- `network.transformer.block_size` matches the loaded model's `config.block_size`
- `network.enable_lambda_head` is consistent with `fast_model.lambda_output_dim > 0`

Mismatches produce warnings rather than errors (the checkpoint is authoritative). The yaml also provides the `inputs.low_dim` key list, which is printed in the export summary.

**Step 4 — Print export summary**

```
--- Export summary ---
  input shape      : (B, T=16, D=35)
  state_output_dim : 4
  lambda_output_dim: 22
  low_dim keys (concatenation order):
    states_embedding
    contact_normals
    contact_points_1
    contact_depths
    gravity_dir
  output file      : .../best_valid_valid_model.onnx
```

**Step 5 — Export**

```python
torch.onnx.export(
    fast_model, dummy_input, output_path,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,          # static shapes — required for TRT + CUDA Graph
    do_constant_folding=True,
)
```

**Why opset 18?** PyTorch's exporter internally uses the `Split` op with a `num_outputs` attribute that was introduced in opset 18. Requesting opset 17 triggers a downconversion attempt that fails.

**Why `dynamic_axes=None`?** TensorRT CUDA Graph capture requires all tensor shapes to be fixed. Changing `B`, `T`, or `D` requires rebuilding the engine.

**Step 6 — Validate**

```python
onnx.checker.check_model(onnx.load(output_path))
```

This performs a **structural/schema check** only:
- Every op is a known ONNX op with valid attributes for opset 18.
- Type and shape consistency throughout the graph.
- No dangling edges.

It does **not** run a forward pass or check numerical correctness. That requires a separate comparison against the PyTorch model (see Section 3, Step 6).

---

## 3. End-to-End Reproducible Workflow

```
Train PyTorch model
        │
        ▼
Export to ONNX                 (export_to_onnx.py)
        │
        ▼
Build TensorRT engine          (build_tensorrt_engine.py)
        │   produces .plan + .engine_meta.pt
        ▼
Flip USE_TENSORRT_ENGINE = True   (gpt_engine.py / axion_engine_with_neural_lambdas.py)
        │
        ▼
Fast inference via TensorRTMSEEngine (duck-types MSEModel)
```

### Step 1 — Train the MSE model

Run the MSE trainer with your config. The checkpoint is saved to:

```
src/axion/neural_solver/train/trained_models/mse/<timestamp>/nn/best_valid_valid_model.pt
src/axion/neural_solver/train/trained_models/mse/<timestamp>/cfg.yaml
```

The `cfg.yaml` in that directory is the frozen copy of the config used for that run (the authoritative source of truth).

### Step 2 — Edit export script constants

Open `fast_inference/export_to_onnx.py` and update:

```python
MODEL_PT     = "src/axion/neural_solver/train/trained_models/mse/<timestamp>/nn/best_valid_valid_model.pt"
NN_MODEL_CFG = "src/axion/neural_solver/train/trained_models/mse/<timestamp>/cfg.yaml"
BATCH_SIZE   = 1        # one robot at inference; rebuild if you need more
DEVICE       = "cuda:0"
```

### Step 3 — Run the export script

Press **F5** (or run `python src/axion/neural_solver/fast_inference/export_to_onnx.py` from the repo root).

On success the output is:
```
ONNX export successful: .../nn/best_valid_valid_model.onnx
```

Note the low_dim key concatenation order printed in the export summary — the inference caller must assemble the input tensor in this exact order.

### Step 4 — Build the TensorRT engine

Use the Python builder script (`trtexec` from the CLI also works but isn't on the path on this workstation):

```bash
python src/axion/neural_solver/fast_inference/build_tensorrt_engine.py
```

Edit the constants at the top of the file:

```python
ONNX_PATH         = "src/.../nn/best_valid_valid_model.onnx"
CHECKPOINT_PT     = "src/.../nn/best_valid_valid_model.pt"
NN_MODEL_CFG      = "src/.../cfg.yaml"
FP16              = True       # falls back to FP32 if the GPU has no fast FP16
WORKSPACE_GB      = 1
OUTPUT_PLAN       = None       # → <onnx_stem>.plan beside the ONNX
OUTPUT_META       = None       # → <onnx_stem>.engine_meta.pt beside the ONNX
RUN_PARITY_CHECK  = True
```

The script:

1. Parses the ONNX with the TensorRT 10 `OnnxParser`, builds a serialized engine via `builder.build_serialized_network(network, config)`, and writes the bytes to `<stem>.plan`.
2. Loads the matching `.pt` to extract per-key `input_rms` (mean, var) plus `state_output_dim`, `lambda_output_dim`, `block_size`, and the low-dim concatenation order from `cfg.yaml`, and saves them as `<stem>.engine_meta.pt`. This sidecar is what `TensorRTMSEEngine` reads at inference time — the engine itself does **not** carry normalization.
3. Optionally runs a one-shot parity check: builds a `FastMSEModel` from the same checkpoint, feeds the same random input through both PyTorch and TRT (with normalization skipped on both sides), and prints the max-abs error.

Expected tolerances vs. PyTorch (FastMSEModel):
- FP32 engine → max-abs ~`1e-5`
- FP16 engine → max-abs ~`1e-3`

After this step the `nn/` directory holds:

```
best_valid_valid_model.pt              # original training checkpoint
best_valid_valid_model.onnx (+ .data)  # static-shape ONNX
best_valid_valid_model.plan            # serialized TensorRT engine
best_valid_valid_model.engine_meta.pt  # low_dim_keys, dims, input_rms
```

### Step 5 — Use the engine at simulation time

`TensorRTMSEEngine` (`fast_inference/tensorrt_mse_engine.py`) is a tiny `nn.Module` that duck-types `MSEModel`. It:

- pre-allocates one static input buffer and one static output buffer on the GPU,
- registers their addresses with the execution context once (CUDA-graph-friendly),
- concatenates the dict input in `low_dim_keys` order,
- applies normalization on the GPU using the stored `input_rms`,
- runs `execute_async_v3` on a dedicated stream and returns `(B, 1, regression_output_dim)`.

To swap it in, flip the toggle constant in the engine you're running. For example, in `src/axion/core/axion_engine_with_neural_lambdas.py`:

```python
USE_TENSORRT_ENGINE = True   # was False
```

The same constant exists in `src/axion/core/gpt_engine.py`. When the toggle is on, the engine constructor instantiates `TensorRTMSEEngine(plan_path, meta_path)` and hands it to `NeuralPredictor` in place of the torch model. No other changes are required.

#### Warm-up note (first `num_states_history - 1` steps)

The PyTorch model accepts any sequence length `T <= block_size` and runs the transformer dynamically on whatever history is currently available (1 token at step 0, 2 at step 1, …, `num_states_history` from then on). The TensorRT engine has a fixed `T = num_states_history` baked in, so the wrapper left-pads short inputs by replicating the oldest entry to fill the buffer.

This means torch and TRT outputs match to FP16 tolerance from step `num_states_history - 1` onward, but **differ during the warm-up** — both backends are computing a different thing there (1 token through a 1-token transformer vs. 10 copies of the same token through a 10-token transformer). For predictors used as solver warm-starts (`AxionEngineWithNeuralLambdas`) this is immaterial; for predictors that determine the actual next state (`GPTEngine`) the warm-up will look different for the first few frames before converging.

### Step 6 — Numerical sanity check (recommended)

This is **already done automatically by `build_tensorrt_engine.py`** when `RUN_PARITY_CHECK = True` (the default). The script builds a `FastMSEModel` from the same checkpoint, runs both PyTorch and TRT on a random tensor of the engine's exact `(B, T, D)` shape, and prints a line like:

```
parity vs FastMSEModel: max_abs=2.023e+00, mean_abs=5.947e-02, rel_max=1.405e-03  (|torch|_max=1.438e+03, |trt|_max=1.440e+03, expected rel ~1e-3)
```

Expected tolerances (matching the FP16/FP32 tradeoff above):
- FP32 engine → `rel_max` ~ `1e-5`
- FP16 engine → `rel_max` ~ `1e-3`

**Read the `rel_max` field, not `max_abs`.** With un-normalized random input the network's lambda head produces values of magnitude `~1e3`, so the absolute number looks large; the relative error is what tells you whether FP16 quantization is behaving. If you want to skip the check (e.g. faster iteration on the build step), set `RUN_PARITY_CHECK = False` at the top of `build_tensorrt_engine.py`.

If you'd rather run a one-shot parity check yourself (e.g. with normalized in-distribution inputs), the same pattern works:

```python
import numpy as np

torch_out = fast_model(dummy_input).cpu().numpy()
trt_out   = trt_model.infer(dummy_input.cpu().numpy())

max_err = np.max(np.abs(torch_out - trt_out))
print(f"Max absolute error: {max_err:.2e}")
# FP32 engine → expect ~1e-5
# FP16 engine → expect ~1e-3
```

### Step 7 — Run with CUDA Graph capture (TODO)

Status: **not yet implemented**. The pieces below are what's still missing for `GPTEngine` / `HybridGPTEngine` to run under `examples/conf/gpt_pendulum.yaml` with `execution: cuda_graph` (i.e. `use_cuda_graph: True`).

The interactive simulator wraps `steps_per_segment` calls to `solver.step(...)` inside a `wp.ScopedCapture()` (`src/axion/simulation/interactive_simulator.py:147-153`). Everything that runs during capture must be (a) pure stream work — no host syncs, no host allocations, no Python branches on tensor values — and (b) shape-static — every kernel/launch must see the same shapes on every iteration. The current torch-only path in `NeuralPredictor` violates both. The TensorRT engine swap takes care of the model itself, but the rest of the predictor / engine plumbing still needs work.

#### What already works (no changes needed)

- `TensorRTMSEEngine` already uses one static input buffer, one static output buffer, fixed tensor addresses (`set_tensor_address` is called once at init), and `execute_async_v3` on a CUDA stream — the model call itself is capturable.
- The TRT engine left-pads short inputs to the engine's fixed `T`, so output shape is constant.
- `GPTEngine` and `HybridGPTEngine` both have `USE_TENSORRT_ENGINE` toggles that swap the wrapper in.
- The Warp portions of `_convert_newton_contacts_to_contacts_for_nn_model` and `_solve` are already kernel launches and are capturable as-is.

#### What needs to change

1. **Pre-warm the history before the first capture.**  `NeuralPredictor.states_history` is a `deque(maxlen=num_states_history)`. At simulation start `len(history) = 0` and grows by 1 per step. The engine T is fixed (= `num_states_history`); the wrapper handles short inputs via left-padding, but every step before `num_states_history - 1` still calls Python code with different intermediate shapes. The simulator must run `num_states_history - 1` warm-up steps **outside** the captured region so the first captured segment is fully shape-stable. Likely path: add a `warmup_steps()` method on `GPTEngine` / `HybridGPTEngine` that the simulator calls before `_capture_cuda_graphs` (or pre-fill the deque with the initial state).

2. **Eliminate host allocations / `.clone()` inside `process_inputs`.**  `src/axion/neural_solver/standalone/neural_predictor.py` currently does, every step:
   - `wp.to_torch(...)` + `torch.cat(...)` + `.unsqueeze(0)` (lines ~362-365)
   - `history_entry = {... "states": states.clone(), ...}` (lines ~375-385)
   - `torch.stack([entry[key] for entry in self.states_history], dim=1)` (line ~390)
   - `tensor.view(B, T, -1)` reshape (lines ~395-396)
   - `wrap2PI(states_flat, ...)` which writes into a freshly-viewed tensor (lines ~401-403)
   - `self._embed_states(...)` which calls `.clone()` (line ~489)

   Each `clone()` / `cat()` / `stack()` allocates a fresh torch tensor; that allocation is not capturable. Refactor to keep one fixed-shape ring buffer per low-dim key (e.g. `(1, T, D_key)`), advance an index, and `copy_` new data in. The dict `nn_model_inputs` should then be a *constant* set of views into those ring buffers.

3. **Drop `self._stream.synchronize()` in the TRT wrapper.**  `TensorRTMSEEngine.evaluate(...)` calls `stream.synchronize()` after `execute_async_v3`. Stream syncs are illegal during capture. Two options:
   - Detect capture mode and skip the sync (CUDA Graph already orders launches; the next kernel won't run until the engine finishes).
   - Run TRT on the **same stream Warp is capturing into** (`wp.get_stream()` ⇄ `torch.cuda.ExternalStream(int(wp_stream.cuda_stream))`). That way no extra stream → no sync needed.

4. **Use the Warp capture stream for torch ops.**  Inside `wp.ScopedCapture()`, Warp captures kernel launches onto its current stream. Torch ops run on their own stream by default; mixing them under a single capture means routing both through one stream. Wrap the relevant torch math (normalization, concat-view, slicing) under `torch.cuda.stream(torch_stream_from_warp)` where `torch_stream_from_warp = torch.cuda.ExternalStream(int(wp.get_stream().cuda_stream))`.

5. **Replace `state_out.joint_q = wp.from_torch(...)` with `wp.copy(...)` into a preallocated buffer.**  `GPTEngine.step` (lines 138-139) does:
   ```python
   state_out.joint_q = wp.from_torch(state_predicted[0,:2].reshape(2,))
   state_out.joint_qd = wp.from_torch(state_predicted[0,2:].reshape(2,))
   ```
   This rebinds `state_out`'s arrays to new pointers each step — incompatible with graph replay (the captured graph hard-codes addresses). Use `wp.copy(dest=state_out.joint_q, src=wp.from_torch(state_predicted_slice_view))` instead. `HybridGPTEngine` already does this correctly; mirror the same pattern in `GPTEngine`.

6. **Make `HybridGPTEngine._neural_init_state_fn` capture-safe.**  Two specific lines need attention (`src/axion/core/hybrid_gpt_engine.py:173-178`):
   ```python
   next_lambdas = torch.where(torch.abs(next_lambdas) < 0.01, ..., next_lambdas)
   next_lambdas[..., 11:] = 0.0
   ```
   `torch.where` allocates; the slice-assign `[..., 11:] = 0.0` does not, so that's fine. Switch to an in-place form: pre-allocate `next_lambdas_buf`, mask in place via `next_lambdas_buf.mul_(mask)`. Also the diagnostic captures `last_predicted_next_body_pose = state_out.body_q.numpy().copy()` (line 197) — `.numpy()` is a host transfer, must be skipped or done outside capture.

7. **Pin the engine's `step()` to a single CUDA stream.**  `wp.copy(...)` calls inside `step()` run on Warp's current stream, which during capture is the capture stream. As long as we don't push torch ops onto a *different* stream (see point 4), Warp + torch + TRT all serialize correctly into one captured graph.

8. **Disable the rendering / HDF5 logger inside the captured region.**  `examples/conf/gpt_pendulum.yaml` defaults to `rendering: gl` and `logging: disabled`, which is already capture-friendly. But `AxionEngineWithNeuralLambdas`'s `_log_neural_step` calls `.numpy()` — that engine should not be combined with `cuda_graph: True` until point 2 above is also resolved end-to-end.

9. **Configure `rendering: headless` for benchmarking.**  ViewerGL begins/ends frames per segment, which is fine, but rendering throughput will dominate if `gl` is on. For pure throughput measurement, switch the gpt_pendulum config to `rendering: headless`.

#### Minimal end-to-end happy-path order

Once all of the above are in place, the run should be:

```bash
# 1. Train an MSE checkpoint that GPTEngine is happy with (Step 1).
# 2. python src/axion/neural_solver/fast_inference/export_to_onnx.py
# 3. python src/axion/neural_solver/fast_inference/build_tensorrt_engine.py
# 4. Flip USE_TENSORRT_ENGINE = True in src/axion/core/gpt_engine.py
#    (or src/axion/core/hybrid_gpt_engine.py).
# 5. Edit examples/conf/gpt_pendulum.yaml:
#       - execution: cuda_graph
# 6. python examples/double_pendulum/<GPTEngine driver script>.py
```

#### Quick smoke test for graph capture before integration

A small standalone test that captures `engine.step(...)` directly (without the simulator) on a fixed state is the fastest way to flush out remaining shape / allocation issues. Pattern:

```python
# warm up
for _ in range(num_states_history):
    engine.step(state_in, state_out, control, contacts, dt)

# capture
with wp.ScopedCapture() as cap:
    engine.step(state_in, state_out, control, contacts, dt)

# replay
for _ in range(1000):
    wp.capture_launch(cap.graph)
```

If this loop runs without "operation not permitted while stream is capturing" errors and produces the same `state_out` as the eager path, the simulator-level capture in Step 7 will succeed too.

---

## Key Constraints

| Constraint | Detail |
|---|---|
| Input concatenation order | `inputs.low_dim` keys must be concatenated in the exact order from `cfg.yaml`. |
| Normalization is caller's responsibility | If `normalize_input: true` was used during training, the caller must apply the same RMS normalization before passing tensors to the engine. The fast model does not normalize. |
| Static shapes are baked in | `B` (batch), `T` (sequence length = `block_size`), `D` (input dim) are fixed at export time. Changing any of them requires re-running Steps 2–4. |
| Opset 18 required | Do not lower `opset_version` below 18; the `Split` op used in attention will fail schema validation. |
| Checkpoint authoritative | On any yaml/checkpoint mismatch, the checkpoint wins. The yaml is only used for cross-validation. |
