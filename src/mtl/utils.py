import math
from typing import Optional
import torch
import torch.nn as nn

def find_config_for_matrices(target_params: int,
                            D1: int,
                            D2: int,
                            D3: int,
                            lowest_d1: int = 1,
                            max_d1: int = 1000,
                            max_d3: Optional[int] = 1000):
    """
    Returns integers d1, d2, d3 such that:
    - d1 * D1 + d2 * D2 + d3 * D3 is as close as possible to target_params
    - d1 == d2 >= lowest_d1
    - d1 <= max_d1, d3 <= max_d3 (if provided)
    """
    best = None
    denom = D1 + D2
    # set search limit for d3
    max_d3 = max_d3 if max_d3 is not None else target_params // D3
    for d3_candidate in range(1, max_d3 + 1):
        remaining = target_params - d3_candidate * D3
        # compute ideal d1
        d1_ideal = remaining / denom
        # consider floor and ceil candidates
        for d1_candidate in (math.floor(d1_ideal), math.ceil(d1_ideal)):
            if d1_candidate < lowest_d1 or d1_candidate > max_d1:
                continue
            total = d1_candidate * (D1 + D2) + d3_candidate * D3
            error = abs(total - target_params)
            if best is None or error < best[0]:
                best = (error, d1_candidate, d3_candidate)
    if best is not None:
        _, best_d1, best_d3 = best
        return best_d1, best_d1, best_d3
    return None



def collect_layer_outputs(model: nn.Module, input_size: tuple, device: torch.device):
    outputs = {}

    for name, module in model.named_modules():
        if name == "":
            continue

        # define a hook that records the shape but DOES NOT return it
        def hook_fn(mod, inp, out, name=name):
            # print("Name: {}  Output shape: {}".format(name, out.shape[1:]))
            outputs[name] = out.shape[1:]
            # no return → original `out` is preserved
        module.register_forward_hook(hook_fn)
    model.to(device)
    model.eval()
    dummy = torch.randn((1, *input_size), device=device)
    with torch.no_grad():
        _ = model(dummy)

    return outputs  


def gather_in_out_channels(module: nn.Module, parent: str, cnn_config_channels: dict):
    for attr_name, child in module.named_children():
        name = parent + "." + attr_name if parent != "" else attr_name
        cnn_config_channels[name] = {
            "is_conv2d": False
        }
        if isinstance(child, nn.Conv2d):
            cnn_config_channels[name] = {
                "total_params": child.kernel_size[0]*child.kernel_size[1]*child.in_channels*child.out_channels,
                "in_channels": child.in_channels,
                "out_channels": child.out_channels,
                "kernel_size": child.kernel_size,
                "padding": child.padding,
                "stride": child.stride,
                "bias": (child.bias is not None),
                "is_conv2d": True,
            }
        else:
            if parent == "":
                new_parent = attr_name
            else:
                new_parent = parent + "." + attr_name
            gather_in_out_channels(child, new_parent, cnn_config_channels)


if __name__ == "__main__":
    test_cases = [
        # (7, 224, 224, 3, 64),       # Conv1
        (3, 56, 56, 64, 64),        # Conv2_x
        (3, 28, 28, 64, 128),       # Conv3_x
        (3, 14, 14, 128, 256),      # Conv4_x
        (3, 7, 7, 256, 512),        # Conv5_x
        # (1, 1, 1, 512, 1000),       # FC (converted to conv1x1 for testing)
    ]
    stride = 1
    padding = 1

    for k, D1, D2, D3, output_channels in test_cases:
        target = k**2 * D3 * output_channels
        lowest_d1 = (D1 - k + 2*padding)//stride + 1
        d3 = output_channels
        d = find_config_for_matrices(target, D1, D2, D3, d3=d3, lowest_d1=lowest_d1)
        if d is not None:
            d1, d2 = d
            total = D1 * d1 + D2 * d2 + D3 * d3
            error = abs(total - target)
            print(f"k={k}, d1={d1}, d2={d2}, d3={d3}, total={total}, error={error}, match={total == target}")
        else:
            print(f"k={k}, No valid d1, d2 found for target={target}")