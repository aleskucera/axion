import torch
path = "src/axion/nn_prediction/trained_models/NeRD_pretrained/pendulum/model.pt"
model, robot_name = torch.load(path, map_location='cpu', weights_only=False)

# Check what keys are in input_rms
if hasattr(model, 'input_rms') and model.input_rms:
    for key, rms in model.input_rms.items():
        print(f"{key}: mean shape = {rms.mean.shape}, var shape = {rms.var.shape}")

# Or check the model's state_dict
for key in model.state_dict().keys():
    if 'input_rms' in key or 'mean' in key or 'var' in key:
        print(f"{key}: {model.state_dict()[key].shape}")

# states_embedding: mean shape = torch.Size([4]), var shape = torch.Size([4])
# contact_normals: mean shape = torch.Size([12]), var shape = torch.Size([12])
# contact_points_1: mean shape = torch.Size([12]), var shape = torch.Size([12])
# contact_depths: mean shape = torch.Size([4]), var shape = torch.Size([4])
# joint_acts: mean shape = torch.Size([2]), var shape = torch.Size([2])
# gravity_dir: mean shape = torch.Size([3]), var shape = torch.Size([3])