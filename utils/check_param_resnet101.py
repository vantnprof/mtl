import torch
import torchvision.models as models
import torch.nn as nn
import sys
sys.path.append("/work/vantn/tensor4cnn/")
from src.mlconv import do_cnn2mlconv


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

    print(f"Total parameters: {total_params:,}")
    print(f"Convolutional layer parameters: {conv_params:,}")
    print(f"Fully connected layer parameters: {fc_params:,}")
    print(f"Other parameters (e.g., BN, etc.): {total_params - conv_params - fc_params:,}")

rank = 200
model = do_cnn2mlconv(model_name="resnet101",
                    num_classes=10,
                    rank=rank
)
# Count parameters
count_params_by_type(model)
