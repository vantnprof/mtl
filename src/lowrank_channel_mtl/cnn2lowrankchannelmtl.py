# src/lowrank/cnn2lowrank.py
import torch
import torch.nn as nn
from torchvision import models
from src.lowrank_channel_mtl.lowrank_channel_mtl import LowRankChannelMTL
from typing import Dict, List


def do_cnn2lowrankchannel_mtl(
    model: nn.Module,
    rank_budget: Dict[str, int]=None,
    replaces: List[str]=[],
) -> nn.Module:
    parent = ""
    def replace_conv_with_lowrank_channel_mtl(module, parent=''):
        for name, child in module.named_children():
            full_name = f"{parent}.{name}" if parent else name
            if isinstance(child, nn.Conv2d) and ('all' in replaces or full_name in replaces
                                                 or any(full_name.startswith(r) for r in replaces)):
                r = rank_budget[full_name] if full_name in rank_budget.keys() else min(child.in_channels, child.out_channels)
                print(f"Replacing {full_name} with LR channel MTL (Rank budget = {r}) layer")
                mtl_layer = LowRankChannelMTL(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    padding=child.padding,
                    stride=child.stride,
                    bias=child.bias,
                    rank_budget=r
                )
                setattr(module, name, mtl_layer)
            else:
                replace_conv_with_lowrank_channel_mtl(child, full_name)

    # Apply the replacement to the entire model
    replace_conv_with_lowrank_channel_mtl(model, parent)
    return model


if __name__ == "__main__":
    # --------------------
    # 🧪 Basic sanity test
    # --------------------
    def count_trainable_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
     
    import argparse
    from torchvision.models import resnet18

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2, help="Low-rank approximation rank")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Height/width of input")
    args = parser.parse_args()

    # Test output shape equivalence
    model_name = "resnet18"
    num_classes = 10
    same_params = True
    input_size = (3, 32, 32)
    rank_budget = {
        'layer4.1.conv1': 10,
        'layer4.1.conv2': 5,
    }
    # Feed an input tensor to both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, *input_size).to(device)  # Example input tensor

    # Load the original ResNet18 model
    original_model = models.resnet18(weights=None, num_classes=10).to(device)
    print(original_model)
    original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)

    lowrank_channel_mtl_model = do_cnn2lowrankchannel_mtl(model_name, num_classes, replaces=['layer4.1'], rank_budget=rank_budget)
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(lowrank_channel_mtl_model)

    # Print results
    print("\n--- Parameter Count Comparison Before feeding input tensor ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"Lowrank Channel MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")

    # Move models to the same device
    original_model = original_model.to(device)
    lowrank_channel_mtl_model = lowrank_channel_mtl_model.to(device)
    original_model.eval()
    lowrank_channel_mtl_model.eval()

    # Forward pass to initialize all layers
    with torch.no_grad():
        cnn_output = original_model(input_tensor)
        mtl_output = lowrank_channel_mtl_model(input_tensor)

    print("CNN's output shape: {}".format(cnn_output.shape))
    print("MTL's output shape: {}".format(mtl_output.shape))

    # Count parameters for both models
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(lowrank_channel_mtl_model)

    # Print results
    print("\n--- Parameter Count Comparison ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")