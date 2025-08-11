from tabulate import tabulate
from src.mtl import MultilinearTransformationLayerSameNumberParam 
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import time
from typing import Tuple


class MTLWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Tuple[int, int], 
                 padding: Tuple[int, int], stride=Tuple[int, int],
        ):
        super(MTLWrapper, self).__init__()
        self.dynamic_mtl = MultilinearTransformationLayerSameNumberParam(
            in_channels, out_channels, 
            kernel_size=kernel_size, padding=padding, stride=stride,
        )
        self.stride = stride
        if stride[0] > 1 and stride[1] > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.dynamic_mtl(x)
        if self.downsample is not None and min(out.shape[-2:]) > 1:
            out = self.downsample(out)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out


def transform_resnet2mtl(model_name: str="resnet18", num_classes:int=10, 
                        same_size: bool=False, same_num_params: bool=True):
    # First, load the model and calculate total number of Conv2d parameters
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # 1. Gather all Conv2d layers and their parameter counts
    conv_layers = []
    def gather_conv_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                conv_layers.append((module, name, child))
            else:
                gather_conv_layers(child)
    gather_conv_layers(model)
    
    for module, name, conv in conv_layers:
        if 'downsample' in name:
            continue
        kernel_size = conv.kernel_size
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        stride = conv.stride
        padding = conv.padding
        # Proportion of parameters for this layer
        # The overall target is to keep total DynamicMTL params equal to original total conv params
        # Scale each DynamicMTL's target_num_params accordingly
        mtl_module = MTLWrapper(in_channels, out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, padding=padding,
        )
        setattr(module, name, mtl_module)

    return model


# --------- Parameter Comparison Test Script ---------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Regular ResNet
    resnet = models.resnet18(num_classes=10)
    mtl_resnet = transform_resnet2mtl("resnet18", num_classes=10, same_size=False, same_num_params=True)

    resnet.to(device)
    mtl_resnet.to(device)

    dummy_input = torch.randn(1, 3, 224, 224)  # standard ResNet input
    dummy_input = dummy_input.to(device)
    
    resnet_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    mtl_params = sum(p.numel() for p in mtl_resnet.parameters() if p.requires_grad)

    print("\n--- Parameter Comparison ---")
    print(f"Original ResNet18 parameters: {resnet_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if resnet_params == mtl_params else 'NO'}")
    print("Before")
    input()

    with torch.no_grad():
        _ = resnet(dummy_input)
        _ = mtl_resnet(dummy_input)

    resnet_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    mtl_params = sum(p.numel() for p in mtl_resnet.parameters() if p.requires_grad)

    print("\n--- Parameter Comparison ---")
    print(f"Original ResNet18 parameters: {resnet_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if resnet_params == mtl_params else 'NO'}")
