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

It does **not** run a forward pass or check numerical correctness. That requires a separate comparison against the PyTorch model (see Section 3, Step 5).

---

## 3. End-to-End Reproducible Workflow

```
Train PyTorch model
        │
        ▼
Export to ONNX  (export_to_onnx.py)
        │
        ▼
Build TensorRT engine  (trtexec)
        │
        ▼
Numerical sanity check
        │
        ▼
Fast inference
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

```bash
trtexec \
  --onnx=src/.../nn/best_valid_valid_model.onnx \
  --saveEngine=src/.../nn/best_valid_valid_model.plan \
  --fp16
```

This step is not yet implemented as a Python script. See `docs/model_to_tensor_rt.md` for the Python API alternative.

### Step 5 — Numerical sanity check (recommended)

Before using the engine in production, verify that TRT outputs match PyTorch:

```python
import numpy as np

torch_out = fast_model(dummy_input).cpu().numpy()
trt_out   = trt_model.infer(dummy_input.cpu().numpy())

max_err = np.max(np.abs(torch_out - trt_out))
print(f"Max absolute error: {max_err:.2e}")
# FP32 engine → expect ~1e-5
# FP16 engine → expect ~1e-3
```

---

## Key Constraints

| Constraint | Detail |
|---|---|
| Input concatenation order | `inputs.low_dim` keys must be concatenated in the exact order from `cfg.yaml`. |
| Normalization is caller's responsibility | If `normalize_input: true` was used during training, the caller must apply the same RMS normalization before passing tensors to the engine. The fast model does not normalize. |
| Static shapes are baked in | `B` (batch), `T` (sequence length = `block_size`), `D` (input dim) are fixed at export time. Changing any of them requires re-running Steps 2–4. |
| Opset 18 required | Do not lower `opset_version` below 18; the `Split` op used in attention will fail schema validation. |
| Checkpoint authoritative | On any yaml/checkpoint mismatch, the checkpoint wins. The yaml is only used for cross-validation. |
