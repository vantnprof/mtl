import torch
from typing import Tuple
import torch.nn as nn
from src.dynamic.utils import find_config
import time


class DynamicMTL(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, target_params: int, weights=None):
        super().__init__()
        self.D3 = input_channels
        self.D1 = None
        self.D2 = None
        self.d1, self.d2 = None, None
        self.d3 = output_channels
        self.target_params = target_params
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.initialized = False
        self.cached_U1 = None
        self.cached_U2 = None
        self.cached_U3 = None
        self.device = None
        self.cache_enabled = True

        self.initialized = False
        self.cached_U1 = None
        self.cached_U2 = None
        self.cached_U3 = None
        self.device = None

        self.m1 = nn.Parameter(torch.empty(0))
        self.L1 = nn.Parameter(torch.empty(0))
        self.z1 = nn.Parameter(torch.empty(0))
        self.R1 = nn.Parameter(torch.empty(0))
        self.m2 = nn.Parameter(torch.empty(0))
        self.L2 = nn.Parameter(torch.empty(0))
        self.z2 = nn.Parameter(torch.empty(0))
        self.R2 = nn.Parameter(torch.empty(0))
        self.m3 = nn.Parameter(torch.empty(0))
        self.L3 = nn.Parameter(torch.empty(0))
        self.z3 = nn.Parameter(torch.empty(0))
        self.R3 = nn.Parameter(torch.empty(0))

        self.weights = weights

    def _initialize_weights(self, H_in: int = None, W_in: int = None, device=None):
        self.device = device
        if self.D1 is None or self.D2 is None:
            if H_in is None or W_in is None:
                raise ValueError("H_in and W_in must be provided to initialize weights.")
            self.D1 = H_in
            self.D2 = W_in
        self.d1, self.d2, self.r1, self.r2, self.r3, _ = find_config(
            target_params=self.target_params,
            D1=self.D1,
            D2=self.D2,
            D3=self.D3,
            d3=self.d3,
        )
        # Ensure each matrix has correct shape
        if self.weights is not None:
            pass
        else:
            # Initialize all parameters with correct shapes and values, using .data assignments
            self.m1.data = torch.ones(self.d1, device=device)
            self.L1.data = torch.empty(self.d1, self.r1, device=device)
            torch.nn.init.kaiming_normal_(self.L1.data, mode='fan_out', nonlinearity='relu')
            self.z1.data = torch.ones(self.r1, device=device)
            self.R1.data = torch.empty(self.r1, self.D1, device=device)
            torch.nn.init.kaiming_normal_(self.R1.data, mode='fan_out', nonlinearity='relu')

            self.m2.data = torch.ones(self.d2, device=device)
            self.L2.data = torch.empty(self.d2, self.r2, device=device)
            torch.nn.init.kaiming_normal_(self.L2.data, mode='fan_out', nonlinearity='relu')
            self.z2.data = torch.ones(self.r2, device=device)
            self.R2.data = torch.empty(self.r2, self.D2, device=device)
            torch.nn.init.kaiming_normal_(self.R2.data, mode='fan_out', nonlinearity='relu')

            self.m3.data = torch.ones(self.d3, device=device)
            self.L3.data = torch.empty(self.d3, self.r3, device=device)
            torch.nn.init.kaiming_normal_(self.L3.data, mode='fan_out', nonlinearity='relu')
            self.z3.data = torch.ones(self.r3, device=device)
            self.R3.data = torch.empty(self.r3, self.D3, device=device)
            torch.nn.init.kaiming_normal_(self.R3.data, mode='fan_out', nonlinearity='relu')
        self.initialized = True
        if not self.training:
            U1 = (self.m1.view(-1, 1) * self.L1) @ (self.z1.view(-1, 1) * self.R1)
            U2 = (self.m2.view(-1, 1) * self.L2) @ (self.z2.view(-1, 1) * self.R2)
            U3 = (self.m3.view(-1, 1) * self.L3) @ (self.z3.view(-1, 1) * self.R3)
            self.cached_U1 = U1
            self.cached_U2 = U2
            self.cached_U3 = U3

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)
    
    def train(self, mode=True):
        self.cached_U1 = None
        self.cached_U2 = None
        self.cached_U3 = None
        return super().train(mode)
    
    def forward(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(0)
        elif X.dim() != 4:
            raise ValueError("Input tensor must be 3D or 4D (B, C, H, W).")

        B, C_in, H_in, W_in = X.shape

        if not self.initialized:
            self._initialize_weights(H_in=H_in, W_in=W_in, device=X.device)

        if not self.training and self.cached_U1 is not None:
            U1 = self.cached_U1
            U2 = self.cached_U2
            U3 = self.cached_U3
        else:
            U1 = (self.m1.view(-1, 1) * self.L1) @ (self.z1.view(-1, 1) * self.R1)
            U2 = (self.m2.view(-1, 1) * self.L2) @ (self.z2.view(-1, 1) * self.R2)
            U3 = (self.m3.view(-1, 1) * self.L3) @ (self.z3.view(-1, 1) * self.R3)
            if not self.training:
                self.cached_U1 = U1
                self.cached_U2 = U2
                self.cached_U3 = U3

        # Step 1: contract along H_in (D1)
        x1 = X.permute(0, 2, 1, 3).reshape(B * C_in * W_in, H_in)
        out1 = x1 @ U1.t()
        out1 = out1.reshape(B, C_in, W_in, -1).permute(0, 3, 1, 2)

        # Step 2: contract along W_in (D2)
        x2 = out1.permute(0, 2, 3, 1).reshape(B * C_in * self.d1, W_in)
        out2 = x2 @ U2.t()
        out2 = out2.reshape(B, C_in, self.d1, -1).permute(0, 2, 3, 1)

        # Step 3: contract along C_in (D3)
        x3 = out2.permute(0, 2, 3, 1).reshape(B * self.d1 * self.d2, C_in)
        out3 = x3 @ U3.t()
        out3 = out3.reshape(B, self.d1, self.d2, self.d3).permute(0, 3, 1, 2).contiguous()

        return out3
    
    

if __name__ == "__main__":
    test_cases = [
        {"input_shape": (16, 32, 32), "output_channels": 32, "kernel_size": (3, 3), "d1": 24, "d2": 24, "d3": 24, "r1": 12, "r2": 12, "r3": 12},
        {"input_shape": (32, 16, 16), "output_channels": 64, "kernel_size": (3, 3), "d1": 36, "d2": 36, "d3": 36, "r1": 18, "r2": 18, "r3": 18},
        {"input_shape": (64, 8, 8), "output_channels": 128, "kernel_size": (3, 3), "d1": 48, "d2": 48, "d3": 48, "r1": 24, "r2": 24, "r3": 24}
    ]

    tolerance_threshold = 100  # example threshold for parameter count difference

    for case in test_cases:
        input_channels, H_in, W_in = case["input_shape"]
        output_channels = case["output_channels"]
        kernel_size = case["kernel_size"]
        # d1, d2, d3 = case["d1"], case["d2"], case["d3"]
        # r1, r2, r3 = case["r1"], case["r2"], case["r3"]

        input_tensor = torch.randn(1, input_channels, H_in, W_in)

        # CNN layer
        conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            bias=False
        )
        _ = conv(input_tensor)
        cnn_param_count = sum(p.numel() for p in conv.parameters() if p.requires_grad)

        # MTL layer initialization (assuming a class DynamicMTL is imported and accepts d1, d2, d3, r1, r2, r3)
        dynamic_mtl = DynamicMTL(
            input_channels=input_channels,
            output_channels=output_channels,
            target_params=cnn_param_count
        )
        # Overwrite the parameters with test case values for d1, d2, d3, r1, r2, r3
        # dynamic_mtl.d1 = d1
        # dynamic_mtl.d2 = d2
        # dynamic_mtl.d3 = d3
        # dynamic_mtl.r1 = r1
        # dynamic_mtl.r2 = r2
        # dynamic_mtl.r3 = r3
        # dynamic_mtl._initialize_weights(H_in=H_in, W_in=W_in, device=input_tensor.device)
        _ = dynamic_mtl(input_tensor)
        mtl_param_count = sum(p.numel() for p in dynamic_mtl.parameters() if p.requires_grad)

        print(f"Test case: input_shape={case['input_shape']}, output_channels={output_channels}, kernel_size={kernel_size}")
        print("CNN params:", cnn_param_count)
        print("MTL params:", mtl_param_count)
        assert abs(cnn_param_count - mtl_param_count) < tolerance_threshold, "Parameter count difference exceeds tolerance"

    # Additional test comparing forward and backward times
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 32
    output_channels = 64
    kernel_size = (3, 3)
    batch_size = 8
    H_in, W_in = 32, 32

    input_tensor = torch.randn(batch_size, input_channels, H_in, W_in, device=device, requires_grad=True)

    conv_layer = nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        bias=False
    ).to(device)
    dynamic_mtl_layer = DynamicMTL(
        input_channels=input_channels,
        output_channels=output_channels,
        target_params=sum(p.numel() for p in conv_layer.parameters() if p.requires_grad)
    ).to(device)

    # Warm-up
    for _ in range(5):
        out_conv = conv_layer(input_tensor)
        out_conv.sum().backward(retain_graph=True)
        conv_layer.zero_grad()
        input_tensor.grad = None

        out_mtl = dynamic_mtl_layer(input_tensor)
        out_mtl.sum().backward(retain_graph=True)
        dynamic_mtl_layer.zero_grad()
        input_tensor.grad = None

    # Timing Conv2d
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_fwd_conv = time.time()
    out_conv = conv_layer(input_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_fwd_conv = time.time()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_bwd_conv = time.time()
    out_conv.sum().backward()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_bwd_conv = time.time()

    conv_layer.zero_grad()
    input_tensor.grad = None

    # Timing DynamicMTL
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_fwd_mtl = time.time()
    out_mtl = dynamic_mtl_layer(input_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_fwd_mtl = time.time()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_bwd_mtl = time.time()
    out_mtl.sum().backward()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_bwd_mtl = time.time()

    dynamic_mtl_layer.zero_grad()
    input_tensor.grad = None

    print("\nPerformance comparison (times in seconds):")
    print(f"{'Layer':<12} {'Forward':>10} {'Backward':>10}")
    print(f"{'Conv2d':<12} {end_fwd_conv - start_fwd_conv:10.6f} {end_bwd_conv - start_bwd_conv:10.6f}")
    print(f"{'DynamicMTL':<12} {end_fwd_mtl - start_fwd_mtl:10.6f} {end_bwd_mtl - start_bwd_mtl:10.6f}")

    # Backward gradient check for DynamicMTL
    input_tensor = torch.randn(batch_size, input_channels, H_in, W_in, device=device, requires_grad=True)
    output = dynamic_mtl_layer(input_tensor)
    loss = output.sum()
    loss.backward()

    # Check if gradients exist
    has_grad = all(p.grad is not None for p in dynamic_mtl_layer.parameters() if p.requires_grad)
    print("\nBackward pass gradient check for DynamicMTL:", "Passed" if has_grad else "Failed")

    # Print gradients of trainable parameters
    print("\nGradients of trainable parameters:")
    for name, param in dynamic_mtl_layer.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f"{name}: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}, shape={tuple(param.grad.shape)}")
            else:
                print(f"{name}: gradient is None")