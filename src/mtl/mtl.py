import torch
import torch.nn as nn
from typing import Tuple


class MTL(nn.Module):
    """
    Implements a multilinear transformation layer:
    O = X ×₁ U1 ×₂ U2 ×₃ U3
    where ×ₙ denotes the n-mode product.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int]=(3, 3),
                 stride: Tuple[int, int]=(1, 1),
                 padding: Tuple[int, int]=(0, 0),
                 bias: bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.input_size = input_size
        self.bias = bias
        self.k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        self.s = self.stride if isinstance(self.stride, int) else self.stride[0]
        self.p = self.padding if isinstance(self.padding, int) else self.padding[0]
        self.U1 = nn.Parameter(torch.empty(0))  # Placeholder, requires_grad=True
        self.U2 = nn.Parameter(torch.empty(0))
        self.U3 = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
        self.initialize = False

    def _init_weights(self, H: int, W: int, dtype: torch.dtype, device: torch.device,  C: int=None):
        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        self.U1.data = torch.randn(H_out, H, dtype=dtype, device=device) * 0.5
        self.U2.data = torch.randn(W_out, W, dtype=dtype, device=device) * 0.5
        if not C is None and C != self.in_channels:
            self.in_channels = C
        self.U3.data = torch.randn(self.out_channels, self.in_channels, dtype=dtype, device=device) * 0.5

        self.initialize = True
        print("MTL initialized with input:", (H, W, C))
        # input()

    def forward(self, X: torch.Tensor):
        B, C_in, H, W = X.shape

        if not self.initialize or H != self.U1.shape[1] or W != self.U2.shape[1] or C_in != self.U3.shape[1]:
            self._init_weights(H, W, dtype=X.dtype, device=X.device, C=C_in)
        # if self.U1.numel() == 0:
        #     H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        #     W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        #     self.U1.data = torch.randn(H_out, H, device=X.device) * 0.5
        #     self.U2.data = torch.randn(W_out, W, device=X.device) * 0.5
        #     self.U3.data = torch.randn(self.out_channels, self.in_channels, device=X.device) * 0.5

        X = X.contiguous()     

        # Mode-3 product (Channels)
        out = torch.einsum('bchw,oc->bohw', X, self.U3)
        
        # Mode-1 product (Height)
        out = torch.einsum('bchw,oh->bcow', out, self.U1)
    
        # Mode-2 product (Width)
        out = torch.einsum('bchw,ow->bcho', out, self.U2)

        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")
    
    def __str__(self):
        return str(self.__repr__())
    
    
def test_feature_map_size_and_parameters():
    # Test parameters
    B, C_in, H_in, W_in = 2, 3, 32, 32
    C_out = 4
    kernel_size = (5, 5)
    stride = (1, 1)
    padding = (1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(B, C_in, H_in, W_in, requires_grad=True).to(device)

    mtl = MTL(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    output = mtl(input_tensor)

    # Check if output has 4 dimensions and reasonable shape
    print("MTL's output shape:", output.shape)

    # Create dummy target for testing with MSE or CrossEntropy
    target = torch.randn_like(output)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()

    # Gradient check
    print("\n=== Gradient Check ===")
    for name, param in mtl.named_parameters():
        grad = param.grad
        print(f"{name}: requires_grad={param.requires_grad}, grad_shape={None if grad is None else grad.shape}")

    # Parameter count
    conv2d = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
    cnn_out = conv2d(input_tensor)
    print("CNN's output shape:", cnn_out.shape)

    mtl_params = sum(p.numel() for p in mtl.parameters() if p.requires_grad)
    conv2d_params = sum(p.numel() for p in conv2d.parameters() if p.requires_grad)
    print(f"\nMTL parameter count: {mtl_params}")
    print(f"Conv2D parameter count: {conv2d_params}")


if __name__ == "__main__":
    test_feature_map_size_and_parameters()
