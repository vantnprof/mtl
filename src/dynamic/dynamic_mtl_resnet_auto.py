from tabulate import tabulate
from src.dynamic.dynamic_mtl import DynamicMTL
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import time


class DynamicMTLWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, target_num_params, stride=1):
        super(DynamicMTLWrapper, self).__init__()
        self.dynamic_mtl = DynamicMTL(in_channels, out_channels, target_num_params)
        self.stride = stride
        if stride > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.dynamic_mtl(x)
        if self.downsample is not None:
            out = self.downsample(out)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out


def transform_resnet2dynamicmtl(model_name: str="resnet18", num_classes:int=10):
    # First, load the model and calculate total number of Conv2d parameters
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # 1. Gather all Conv2d layers and their parameter counts
    conv_layers = []
    total_conv_params = 0
    def gather_conv_layers(module):
        nonlocal total_conv_params
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                k = child.kernel_size[0]
                in_channels = child.in_channels
                out_channels = child.out_channels
                num_params = k**2 * in_channels * out_channels
                conv_layers.append((module, name, child, num_params))
                total_conv_params += num_params
            else:
                gather_conv_layers(child)
    gather_conv_layers(model)
    
    # 2. Replace each Conv2d with DynamicMTLWrapper, scaling target_num_params by parameter proportion
    for module, name, conv, num_params in conv_layers:
        # Skip downsample paths (usually contain conv layers + BN expecting specific dimensions)
        if 'downsample' in name:
            continue
        k = conv.kernel_size[0]
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        stride = conv.stride[0]
        # Proportion of parameters for this layer
        param_proportion = num_params / total_conv_params if total_conv_params > 0 else 1.0
        # The overall target is to keep total DynamicMTL params equal to original total conv params
        # Scale each DynamicMTL's target_num_params accordingly
        scaled_target_num_params = int(round(param_proportion * total_conv_params))
        dynamic_mtl_module = DynamicMTLWrapper(in_channels, out_channels, scaled_target_num_params, stride=stride)
        setattr(module, name, dynamic_mtl_module)

    return model



def test_models(model_names, num_classes=10, input_size=(1, 3, 128, 128)):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(*input_size).to(device)

    for model_name in model_names:
        print(f"\nTesting model: {model_name}")

        # Load original model
        torch.cuda.reset_peak_memory_stats()
        original_model = models.__dict__[model_name](weights=None, num_classes=num_classes).to(device).train()
        with torch.no_grad():
            original_output = original_model(input_tensor)
        orig_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        orig_mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)

        # Timing original model forward and backward
        fwd_times_orig = []
        bwd_times_orig = []
        for _ in range(10):
            original_model.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_fwd = time.time()
            output = original_model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_fwd = time.time()
            loss = output.sum()
            start_bwd = time.time()
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_bwd = time.time()
            fwd_times_orig.append((end_fwd - start_fwd) * 1000)
            bwd_times_orig.append((end_bwd - start_bwd) * 1000)
        avg_fwd_time_orig = sum(fwd_times_orig) / len(fwd_times_orig)
        avg_bwd_time_orig = sum(bwd_times_orig) / len(bwd_times_orig)

        # Load DynamicMTL version
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dynamic_model = transform_resnet2dynamicmtl(model_name=model_name, num_classes=num_classes).to(device).train()
        with torch.no_grad():
            dynamic_output = dynamic_model(input_tensor)
        dyn_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        dyn_mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        dynamic_params = sum(p.numel() for p in dynamic_model.parameters() if p.requires_grad)

        # Timing dynamic model forward and backward
        fwd_times_dyn = []
        bwd_times_dyn = []
        for _ in range(10):
            dynamic_model.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_fwd = time.time()
            output = dynamic_model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_fwd = time.time()
            loss = output.sum()
            start_bwd = time.time()
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_bwd = time.time()
            fwd_times_dyn.append((end_fwd - start_fwd) * 1000)
            bwd_times_dyn.append((end_bwd - start_bwd) * 1000)
        avg_fwd_time_dyn = sum(fwd_times_dyn) / len(fwd_times_dyn)
        avg_bwd_time_dyn = sum(bwd_times_dyn) / len(bwd_times_dyn)

        results.append([
            model_name,
            original_params,
            dynamic_params,
            f"{avg_fwd_time_orig:.2f}",
            f"{avg_bwd_time_orig:.2f}",
            f"{avg_fwd_time_dyn:.2f}",
            f"{avg_bwd_time_dyn:.2f}",
            f"{orig_mem_alloc:.2f}",
            f"{dyn_mem_alloc:.2f}"
        ])

        del original_model, dynamic_model
        torch.cuda.empty_cache()

    print("\nModel Parameter and Timing Comparison:")
    headers = ["Model", "Original Params", "DynamicMTL Params",
               "Orig Fwd (ms)", "Orig Bwd (ms)",
               "MTL Fwd (ms)", "MTL Bwd (ms)",
               "Orig Mem (MB)", "MTL Mem (MB)"]
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

def compare_layer_configs(model_name="resnet18", num_classes=10):
    import pandas as pd
    from tabulate import tabulate

    original_model = models.__dict__[model_name](weights=None, num_classes=num_classes)
    dynamic_model = transform_resnet2dynamicmtl(model_name=model_name, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 3, 128, 128).to(device)
    original_model.to(device).eval()
    dynamic_model.to(device).eval()
    with torch.no_grad():
        _ = original_model(input_tensor)
        _ = dynamic_model(input_tensor)

    original_layers = []
    dynamic_layers = []

    def collect_conv_configs(model, prefix=""):
        configs = []
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Conv2d):
                configs.append((full_name, type(module).__name__, module.in_channels, module.out_channels, module.kernel_size, None, None, None, None, None, None, None, None))
            elif isinstance(module, DynamicMTLWrapper):
                dm = module.dynamic_mtl
                configs.append((full_name, type(dm).__name__,
                                dm.D3, None, None, dm.D1, dm.D2,
                                dm.d1, dm.d2, dm.d3, dm.r1, dm.r2, dm.r3))
            else:
                configs.extend(collect_conv_configs(module, full_name))
        return configs

    original_layers = collect_conv_configs(original_model)
    dynamic_layers = collect_conv_configs(dynamic_model)

    # Align lengths
    max_len = max(len(original_layers), len(dynamic_layers))
    original_layers += [("", "", "", "", "", "", "", "", "", "", "", "", "")] * (max_len - len(original_layers))
    dynamic_layers += [("", "", "", "", "", "", "", "", "", "", "", "", "")] * (max_len - len(dynamic_layers))

    headers = [
        "Layer", "Orig.Type", "Dyn.Type",
        "Kernel", "InCh", "OutCh",
        "D1", "D2", "D3", "d1", "d2", "d3", "r1", "r2", "r3"
    ]
    comparison_data = []
    for orig, dyn in zip(original_layers, dynamic_layers):
        layer_name = orig[0] if orig[0] else dyn[0]
        orig_type = orig[1] if orig[1] else "-"
        dyn_type = dyn[1] if dyn[1] else "-"

        kernel = orig[4] if orig_type == "Conv2d" else "-"
        in_ch = orig[2] if orig_type == "Conv2d" else "-"
        out_ch = orig[3] if orig_type == "Conv2d" else "-"

        D1 = dyn[5] if dyn_type == "DynamicMTL" else "-"
        D2 = dyn[6] if dyn_type == "DynamicMTL" else "-"
        D3 = dyn[2] if dyn_type == "DynamicMTL" else "-"
        d1 = dyn[7] if dyn_type == "DynamicMTL" else "-"
        d2 = dyn[8] if dyn_type == "DynamicMTL" else "-"
        d3 = dyn[9] if dyn_type == "DynamicMTL" else "-"
        r1 = dyn[10] if dyn_type == "DynamicMTL" else "-"
        r2 = dyn[11] if dyn_type == "DynamicMTL" else "-"
        r3 = dyn[12] if dyn_type == "DynamicMTL" else "-"

        comparison_data.append([
            layer_name, orig_type, dyn_type,
            kernel, in_ch, out_ch,
            D1, D2, D3, d1, d2, d3, r1, r2, r3
        ])

    print("\nClean Layer Configuration Comparison:")
    print(tabulate(comparison_data, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    test_models(["resnet18", "resnet34"])
    compare_layer_configs("resnet18")
