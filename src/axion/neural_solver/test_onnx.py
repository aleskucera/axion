import torch
import onnx

model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 5),
).eval()

x = torch.randn(1, 10)

torch.onnx.export(
    model,
    x,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)

print("ONNX export successful")