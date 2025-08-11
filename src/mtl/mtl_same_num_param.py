import torch
import torch.nn as nn
from typing import Tuple
from src.mtl.utils import find_config_for_matrices


class MultilinearTransformationLayerSameNumberParam(nn.Module):
    """
    Implements a multilinear transformation layer:
    O = X ×₁ U1 ×₂ U2 ×₃ U3
    where ×ₙ denotes the n-mode product.
    """
    def __init__(self, desired_param_num, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], 
                 padding: Tuple[int, int], bias=False, config_dim: dict=None):
        super().__init__()
        self.desired_param_num = desired_param_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if not config_dim is None:
            self.D1 = config_dim['D1']
            self.D2 = config_dim['D2']
            self.D3 = config_dim['D3']
            self.d1 = config_dim['d1']
            self.d2 = config_dim['d2']
            self.d3 = config_dim['d3']
        else:
            self.D1 = None
            self.D2 = None
            self.D3 = in_channels
            self.d1 = None
            self.d2 = None
            self.d3 = None
        self.initialized = False
        self.not_match = False
        self.conv2d = None
        self.U1 = None
        self.U2 = None
        self.U3 = None
        
    def forward(self, X):
        # X: (B, C_in, H_in, W_in)
        # Ensure input is 4D
        if X.dim() != 4:
            raise ValueError("Input tensor must be 4D (B, C_in, H_in, W_in)")
        # Unpack input dimensions
        B, C_in, H_in, W_in = X.shape
        if not self.initialized:
            # Initialize spatial dims
            if self.D1 is None and self.D2 is None:
                self.D1, self.D2 = H_in, W_in
                config = find_config_for_matrices(
                    target_params=self.desired_param_num,
                    D1=self.D1, D2=self.D2, D3=self.D3, 
                    # lowest_d1=(self.D1-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1,
                )
                # Fallback if no exact channel match
                if config is None:
                    self.not_match = True
                    self.conv2d = nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        stride=self.stride,
                        bias=self.bias
                    ).to(X.device)
                    # print("Using Conv2D layer instead due to channel mismatch!")
                else:
                    self.d1, self.d2, self.d3 = config
                    self.U1 = nn.Parameter(torch.randn(self.d1, self.D1, device=X.device))
                    self.U2 = nn.Parameter(torch.randn(self.d2, self.D2, device=X.device))
                    self.U3 = nn.Parameter(torch.randn(self.d3, self.D3, device=X.device))
            else:
                self.U1 = nn.Parameter(torch.randn(self.d1, self.D1, device=X.device))
                self.U2 = nn.Parameter(torch.randn(self.d2, self.D2, device=X.device))
                self.U3 = nn.Parameter(torch.randn(self.d3, self.D3, device=X.device))
            self.initialized = True

        X = X.contiguous()

        if self.not_match:
            return self.conv2d(X)  # Use self.conv2d directly
        else:
            return torch.einsum('bchw,dh,ew,fc->bfde', X, self.U1, self.U2, self.U3)


if __name__ == "__main__":
    # Example usage
    B, C_in, H_in, W_in = (1, 3, 224, 224)
    C_out = 64
    kernel_size = (13, 13)
    stride = (1, 1)
    padding = (1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(B, C_in, H_in, W_in).to(device)

    # Initialize MTL
    mtl = MultilinearTransformationLayerSameNumberParam(
        desired_param_num=kernel_size[0]*kernel_size[1]*C_in*C_out,
        in_channels=C_in, out_channels=C_out,
        kernel_size=kernel_size, stride=stride, padding=padding
    ).to(device)

    # Forward pass
    output_tensor = mtl(input_tensor)

    print("d1: ", mtl.d1)
    print("d2: ", mtl.d2)
    print("d3: ", mtl.d3)

    print("Input shape :", input_tensor.shape)
    print("MTL's Output shape:", output_tensor.shape)

    # Comparison test between MTL and Conv2d
    
    conv = torch.nn.Conv2d(in_channels=C_in, out_channels=C_out, 
                           kernel_size=kernel_size, padding=padding, 
                           bias=False).to(device)
    output_tensor_conv = conv(input_tensor)
    print("Conv2d's output shape: ",output_tensor_conv.shape)

    print("\n--- Parameter Count Comparison ---")
    conv_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    mtl_params = sum(p.numel() for p in mtl.parameters() if p.requires_grad)
    print(f"Conv2d parameters: {conv_params}")
    print(f"MTL parameters:    {mtl_params}")
    print(f"Match? {'YES' if conv_params == mtl_params else 'NO'}")