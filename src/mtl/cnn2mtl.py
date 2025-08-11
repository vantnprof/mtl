import torch
import torch.nn as nn
from torchvision import models
from typing import List
from src.mtl.mtl import MTL
from typing import Tuple
import torch.nn.functional as F
# from src.mtl.utils import collect_layer_outputs, gather_in_out_channels


# class MTLSameFeatureMapSizeWrapper(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
#                  padding: Tuple[int, int], stride=Tuple[int, int], bias: bool=False
#         ):
#         super(MTLSameFeatureMapSizeWrapper, self).__init__()
#         self.mtlsameparam = MTL(
#             in_channels, out_channels, kernel_size=kernel_size, 
#             padding=padding, stride=stride, bias=bias
#         )
#         self.stride = stride
#         if stride[0] > 1 and stride[1]> 1:
#             self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
#         else:
#             self.downsample = None

#     def forward(self, x):
#         out = self.mtlsameparam(x)
#         if self.downsample is not None and min(out.shape[-2:]) > 1:
#             out = self.downsample(out)
#         if out.shape[-2:] != x.shape[-2:]:
#             out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
#         return out
    

# This function works but not really the ideal one, need to improve to work with all model_name
def do_cnn2mtl(model: nn.Module, 
            replaces: List[str]=['all'], 
    ):
    """
    Converts a CNN model (e.g., ResNet18) to an MLConv version by replacing all Conv2d layers with MLConv layers.

    Args:
        model_name (str): Name of the model (e.g., "resnet18", "resnet34").
        num_classes (int): Number of output classes for the model.
        rank (int): Rank for the MLConv layers.

    Returns:
        nn.Module: The modified model with MLConv layers.
    """    
    parent = ""
    def replace_conv_with_mtl(module, parent=''):
        for name, child in module.named_children():
            full_name = f"{parent}.{name}" if parent else name
            if isinstance(child, nn.Conv2d) and ('all' in replaces or full_name in replaces
                                                 or any(full_name.startswith(r) for r in replaces)):
                print(f"Replacing {full_name} with MTL layer")
                mtl_layer = MTL(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    padding=child.padding,
                    stride=child.stride,
                    bias=child.bias
                )
                setattr(module, name, mtl_layer)
            else:
                replace_conv_with_mtl(child, full_name)

    # Apply the replacement to the entire model
    replace_conv_with_mtl(model, parent)

    return model


if __name__ == "__main__":
    # Function to count trainable parameters
    def count_trainable_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Example usage
    model_name = "resnet18"
    num_classes = 10
    same_params = True
    input_size = (3, 32, 32)
    # Feed an input tensor to both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, *input_size).to(device)  # Example input tensor

    # Load the original ResNet18 model
    original_model = models.resnet18(weights=None, num_classes=10).to(device)
    original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)
    # original_model = nn.Sequential(
    #     original_model.conv1,
    #     original_model.bn1,
    #     original_model.relu,
    # #     original_model.maxpool,
    # #     original_model.layer1,
    # #     original_model.layer2,
    # #     original_model.layer3,
    # #     original_model.layer4
    # )
    # print("original_model", original_model)
    # Convert the model to MTL
    mtl_model = do_cnn2mtl(model_name, num_classes, replaces=['all'])
    # mtl_model = nn.Sequential(
    #     mtl_model.conv1,
    #     mtl_model.bn1,
    #     mtl_model.relu,
    #     mtl_model.maxpool,
    #     mtl_model.layer1,
    #     mtl_model.layer2,
    #     mtl_model.layer3,
    #     mtl_model.layer4,
    # )
    # print("mtl_model", mtl_model)
    # input()
    # Count parameters for both models
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(mtl_model)

    # Print results
    print("\n--- Parameter Count Comparison Before feeding input tensor ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")

    # Move models to the same device
    original_model = original_model.to(device)
    mtl_model = mtl_model.to(device)
    original_model.eval()
    mtl_model.eval()

    # Forward pass to initialize all layers
    with torch.no_grad():
        cnn_output = original_model(input_tensor)
        mtl_output = mtl_model(input_tensor)

    print("CNN's output shape: {}".format(cnn_output.shape))
    print("MTL's output shape: {}".format(mtl_output.shape))

    # Count parameters for both models
    original_params = count_trainable_params(original_model)
    mtl_params = count_trainable_params(mtl_model)

    # Print results
    print("\n--- Parameter Count Comparison ---")
    print(f"Original ResNet18 parameters: {original_params}")
    print(f"MTL ResNet18 parameters:      {mtl_params}")
    print(f"Match? {'YES' if original_params == mtl_params else 'NO'}")