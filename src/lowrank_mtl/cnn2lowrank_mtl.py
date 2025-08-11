import torch
import torch.nn as nn
from torchvision import models
from src.lowrank_mtl.lowrank_mtl import LowRankMTL
from typing import Optional


def do_cnn2lowrank_mtl(
    model: nn.Module,
    replaces=[],
    rank: int=1,
) -> nn.Module:
    parent = ""
    def replace_conv_with_lowrank_mtl(module, parent=''):
        for name, child in module.named_children():
            full_name = f"{parent}.{name}" if parent else name
            if isinstance(child, nn.Conv2d) and ('all' in replaces or full_name in replaces or any(full_name.startswith(l) for l in replaces)):
                print(f"Replacing {full_name} with LR-MTL (P = {rank}) layer")
                mtl_layer = LowRankMTL(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    padding=child.padding,
                    stride=child.stride,
                    bias=child.bias,
                    rank=rank
                )
                setattr(module, name, mtl_layer)
            else:
                replace_conv_with_lowrank_mtl(child, full_name)

    # Apply the replacement to the entire model
    replace_conv_with_lowrank_mtl(model, parent)
    return model




if __name__ == "__main__":
    import torch
    from src.lowrank_mtl.lowrank_mtl import LowRankMTL

    def count_mtl_layers(model):
        return sum(1 for m in model.modules() if isinstance(m, LowRankMTL))

    def count_conv2d_layers(model):
        return sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))

    model_name = "resnet18"
    num_classes = 10
    rank = 4

    print("Creating low-rank MTL converted model...")
    model = do_cnn2lowrank_mtl(model_name=model_name, num_classes=num_classes, rank=rank)
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Count and compare
    n_mtl = count_mtl_layers(model)
    n_conv = count_conv2d_layers(model)

    print(f"Number of LowRankMTL layers: {n_mtl}")
    print(f"Number of Conv2d layers (should be 0): {n_conv}")

    # Confirm output works
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)