import torch
import torch.nn as nn
from torchvision import models
from src.mlconv.mlconv import MLConv
from typing import List


def do_cnn2mlconv(
        model: nn.Module, 
        rank: int=1,
    ):
    parent = ""
    def replace_conv_with_mlconv(module, parent=''):
        for name, child in module.named_children():
            full_name = f"{parent}.{name}" if parent else name
            if isinstance(child, nn.Conv2d):
                print(f"Replacing {full_name} with MLConv layer (R={rank})")
                mtl_layer = MLConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    padding=child.padding,
                    stride=child.stride,
                    bias=child.bias,
                    rank=rank,
                )
                setattr(module, name, mtl_layer)
            else:
                replace_conv_with_mlconv(child, full_name)

    # Apply the replacement to the entire model
    replace_conv_with_mlconv(model, parent)

    return model


if __name__ == "__main__":
    # Function to count trainable parameters
    def count_trainable_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Example usage
    model_name = "resnet18"
    num_classes = 10
    same_params = True

    # Load the original ResNet18 model
    original_model = models.resnet18(pretrained=True)
    original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)
    
    # Convert the model to MTL
    mlconv_model = cnn2mlconv(model_name, num_classes, rank=1, replaces=['all'])

    # Count parameters for both models
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(mlconv_model)

    # Print results
    print("\n--- Parameter Count Comparison Before feeding input tensor ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")

 
    # Feed an input tensor to both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor

    # Move models to the same device
    original_model = original_model.to(device)
    mlconv_model = mlconv_model.to(device)

    # Forward pass to initialize all layers
    with torch.no_grad():
        _ = original_model(input_tensor)
        _ = mlconv_model(input_tensor)

    # Count parameters for both models
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(mlconv_model)

    # Print results
    print("\n--- Parameter Count Comparison ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"MLConv ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")