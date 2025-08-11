import torch
from src.mtl.mtl import MTL


def mtl_replacement(model, replaces):
    for name, module in model.named_modules():
        if 'dfl.conv' in name:
            continue
        if any(name == r or name.startswith(r + ".") for r in replaces):
            if isinstance(module, torch.nn.Conv2d):
                print(f"Replacing layer: {name} (Conv2d -> MTL)")
                parent_module = model
                sub_names = name.split('.')
                for sub_name in sub_names[:-1]:
                    parent_module = getattr(parent_module, sub_name)
                target_name = sub_names[-1]
                old_layer = getattr(parent_module, target_name)

                in_channels = old_layer.in_channels
                out_channels = old_layer.out_channels
                kernel_size = old_layer.kernel_size
                stride = old_layer.stride
                padding = old_layer.padding
                bias = old_layer.bias is not None

                mtl_layer = MTL(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
                setattr(parent_module, target_name, mtl_layer)
    return model


if __name__ == "__main__":
    from ultralytics import YOLO

    # Load YOLOv11 model
    model = YOLO("yolo11n.yaml").model

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Original model trainable parameters:", count_trainable_parameters(model))

    dummy_input = torch.randn(1, 3, 640, 640)
    original_output = model(dummy_input)
    print("Original output shape:", original_output[0].shape if isinstance(original_output, list) else original_output.shape)

    # Specify a top-level module name to replace all Conv2d layers inside recursively
    replaces = ["model.19", "model.20", "model.21", "model.22"]

    # Apply the MTL replacement
    modified_model = mtl_replacement(model, replaces)

    print("MTL model trainable parameters:", count_trainable_parameters(modified_model))

    mtl_output = modified_model(dummy_input)
    print("MTL model output shape:", mtl_output[0].shape if isinstance(mtl_output, list) else mtl_output.shape)

    # Print the modified model to verify replacement
    # print("\nModified model:")
    # print(modified_model)