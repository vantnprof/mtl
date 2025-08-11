import torch
import torch.nn as nn
import sys
sys.path.append("/work/vantn/tensor4cnn/")

# Load your YOLO11x model
from ultralytics import YOLO

# Adjust this to the correct model path if needed
model = YOLO("/work/vantn/tensor4cnn/src/yolo11/config/yolo11x-obb_mlconv.yaml").model  # or YOLO("path/to/weights.pt").model

def count_params_by_type(model):
    conv_params = 0
    fc_params = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
        elif isinstance(module, nn.Linear):
            fc_params += sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params:,}")
    print(f"Convolutional layer parameters: {conv_params:,}")
    print(f"Fully connected (Linear) layer parameters: {fc_params:,}")
    print(f"Other trainable parameters: {total_params - conv_params - fc_params:,}")

count_params_by_type(model)
