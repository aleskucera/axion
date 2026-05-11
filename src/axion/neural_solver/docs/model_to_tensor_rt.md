# 🚀 End-to-End Plan: PyTorch → ONNX → TensorRT → CUDA Graph (with Warp)

This is a **practical, copyable checklist + code guide** to get your transformer model into a CUDA-graph-friendly inference pipeline.

---

# 🧱 0. Install Dependencies

## Python packages

```bash
pip install onnx onnxruntime
```

## TensorRT (NVIDIA)

You have two options:

### Option A — System install (recommended)

* Install via NVIDIA package (depends on your CUDA version)

### Option B — Python wheels (simpler)

```bash
pip install tensorrt
```

👉 Verify:

```python
import tensorrt as trt
print(trt.__version__)
```

---

# 🛠️ 1. Fix Your GPT Model (CRITICAL)

Apply these changes **before export**.

---

## 🔴 Disable Flash Attention

```python
# in CausalSelfAttention.__init__
self.flash = False
```

---

## 🔴 Replace masked_fill(-inf)

### ❌ Original:

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

### ✅ Replace with:

```python
mask = self.bias[:, :, :T, :T]
att = att + (mask == 0) * (-1e9)   # FP32 safe
```

👉 If using FP16:

```python
att = att + (mask == 0) * (-1e4)
```

---

## 🟡 Precompute position IDs

In `GPT.__init__`:

```python
self.register_buffer("pos_ids", torch.arange(config.block_size))
```

In `forward`:

```python
pos = self.pos_ids[:t]
```

---

## 🟡 Ensure inference mode removes dropout

Before export:

```python
model.eval()
```

---

## 🟡 Optional: remove dynamic slicing (if needed)

If sequence length is fixed:

```python
# Instead of slicing bias each forward
att = att + (self.bias == 0) * (-1e9)
```

---

# 🛠️ 2. Create Export Wrapper (REMOVE dict input)

TensorRT does NOT support dict inputs.

---

## Example wrapper

```python
import torch
import torch.nn as nn

class ExportWrapper(nn.Module):
    def __init__(self, model, low_dim_names):
        super().__init__()
        self.model = model
        self.low_dim_names = low_dim_names

    def forward(self, x):
        # x shape: (B, T, total_low_dim)

        # Split back into dict if needed
        input_dict = {}
        start = 0
        for name, size in self.low_dim_names:
            input_dict[name] = x[..., start:start+size]
            start += size

        return self.model(input_dict)
```

👉 Alternatively (simpler): refactor model to take a single tensor.

---

# 🧪 3. Prepare Dummy Input

```python
B = 1
T = FIXED_SEQ_LEN
D = INPUT_DIM

dummy_input = torch.randn(B, T, D).cuda()
```

👉 MUST match runtime shape exactly (for CUDA Graphs).

---

# 📤 4. Export to ONNX

```python
torch.onnx.export(
    export_model,                 # wrapped model
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,           # IMPORTANT: keep static
    do_constant_folding=True,
)
```

---

## ✅ Validate ONNX

```python
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

---

# ⚙️ 5. Build TensorRT Engine

## Option A — CLI (simplest)

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --fp16
```

---

## Option B — Python API

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_engine(network, config)

with open("model.plan", "wb") as f:
    f.write(engine.serialize())
```

---

# 🧠 6. TensorRT Inference Wrapper

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append(device_mem)
            else:
                self.outputs.append(device_mem)

    def infer(self, input_array):
        cuda.memcpy_htod(self.inputs[0], input_array)

        self.context.execute_v2(self.bindings)

        output = np.empty_like(input_array)  # adjust shape!
        cuda.memcpy_dtoh(output, self.outputs[0])
        return output
```

---

# 🔗 7. Integration with Warp / CUDA Graph

## Key idea:

* allocate buffers ONCE
* reuse them
* capture graph

---

## Example (pseudo-flow)

```python
# allocate static buffers
input_gpu = cuda.mem_alloc(size)
output_gpu = cuda.mem_alloc(size)

# warmup
for _ in range(3):
    trt_model.context.execute_v2(bindings)

# capture graph
graph = cuda_graph_capture_begin()

trt_model.context.execute_v2(bindings)

# your Warp kernels here

cuda_graph_capture_end(graph)

# replay
for step in range(N):
    cuda_graph_launch(graph)
```

---

# ⚠️ 8. Critical Constraints

## MUST satisfy:

* fixed batch size
* fixed sequence length
* fixed input dimensions

---

## If any of these change:

👉 you MUST rebuild the TensorRT engine

---

# 🧪 9. Validation Step (IMPORTANT)

Before trusting TensorRT:

```python
torch_out = model(dummy_input)
trt_out = trt_model.infer(dummy_input.cpu().numpy())

print(np.max(np.abs(torch_out.cpu().numpy() - trt_out)))
```

Expected:

* FP32 → ~1e-5
* FP16 → ~1e-3

---

# 🧭 Final Workflow Summary

```text
1. Train model in PyTorch
2. Apply export fixes (attention, masking, etc.)
3. Wrap model → tensor input only
4. Export to ONNX
5. Build TensorRT engine
6. Integrate into Warp pipeline
7. Capture CUDA Graph
8. Run fast inference
```

---

# 💡 Pro Tips

* Start with **FP32**, then switch to FP16
* Use **small sequence length first** to debug
* Keep a **reference PyTorch output** for sanity checks
* Expect **1–2 iterations fixing ONNX issues**

---

# 🎯 What you get at the end

* 🚫 No Python overhead
* ⚡ Fully GPU-native inference
* 🔒 CUDA Graph compatible
* 🚀 Maximum performance

---

If something fails during ONNX or TensorRT build, the error message will usually point to the exact op—those are fixable case-by-case.
