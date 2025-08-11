# src/lowrank/cnn2lowrank.py
import torch
import torch.nn as nn
from torchvision import models
from src.lowrank.lowrank import LowRankConv2d


def do_cnn2lowrank(
    model: nn.Module,
    rank: int,
) -> nn.Module:
    parent = ""
    def replace_conv_with_lowrank(module, parent=''):
        for name, child in module.named_children():
            full_name = f"{parent}.{name}" if parent else name
            if isinstance(child, nn.Conv2d):
                print(f"Replacing {full_name} ({child.in_channels},{child.out_channels}, {child.kernel_size}, {child.padding}, {child.stride}) with LR (R = {rank}) layer")
                mtl_layer = LowRankConv2d(
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
                replace_conv_with_lowrank(child, full_name)

    # Apply the replacement to the entire model
    replace_conv_with_lowrank(model, parent)
    return model


if __name__ == "__main__":
    # --------------------
    # 🧪 Basic sanity test
    # --------------------
    import argparse
    from torchvision.models import resnet18

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2, help="Low-rank approximation rank")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Height/width of input")
    args = parser.parse_args()

    # Test output shape equivalence
    model_orig = resnet18(weights=None, num_classes=10)
    print(model_orig)
    x = torch.randn(args.batch, 3, args.img_size, args.img_size)
    with torch.no_grad():
        y_orig = model_orig(x)
    print("Original model output shape:", y_orig.shape)

    # Instantiate pretrained ResNet18 (random weights for demo)
    model = do_cnn2lowrank(model_name="resnet18", num_classes=10, rank=args.rank)
    with torch.no_grad():
        y = model(x)
    print("LowRank model output shape:", y.shape)
    print("Output shape match:", y.shape == y_orig.shape)