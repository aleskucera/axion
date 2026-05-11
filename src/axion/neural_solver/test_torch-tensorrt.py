import torch
import torch_tensorrt

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Torch-TensorRT version:", torch_tensorrt.__version__)

# Simple model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 5),
).eval().cuda()

# Example input
inputs = [torch.randn(1, 10).cuda()]

# Compile with Torch-TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=inputs,
    enabled_precisions={torch.float},
)

# Run inference
output = trt_model(inputs[0])

print("Compilation SUCCESS")
print("Output shape:", output.shape)
print(output)