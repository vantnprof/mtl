import torch
import torch.nn as nn
from typing import Tuple


class LowRankConv2d(nn.Module):
    def __init__(self,
                in_channels: int, 
                out_channels: int, 
                kernel_size: Tuple[int]=(3, 3),
                stride: Tuple[int, int]=(1, 1),
                padding: Tuple[int, int]=(0, 0),
                bias: bool=False, 
                rank:int=265,
        ):
        super().__init__()
        # Standardize kernel, stride, padding to tuples
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding

        # Vertical conv applies vertical stride only
        self.vertical_conv = nn.Conv2d(
            in_channels, rank, kernel_size=(kernel_h, 1),
            stride=(stride_h, 1), padding=(pad_h, 0),
            bias=False
        )
        # Horizontal conv applies horizontal stride only
        self.horizontal_conv = nn.Conv2d(
            rank, out_channels, kernel_size=(1, kernel_w),
            stride=(1, stride_w), padding=(0, pad_w),
            bias=bias
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.vertical_conv.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.horizontal_conv.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.horizontal_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vertical_conv(x)
        x = self.horizontal_conv(x)
        return x
    

def test_low_rank_conv2d():
    import torch
    import torch.nn as nn

    # Settings
    in_channels = 3
    out_channels = 16
    kernel_size = 11
    stride = 1
    padding = 2
    rank = 100
    input_size = (30, in_channels, 64, 64)  # Batch size 1, 32x32 image

    # Create dummy input
    x = torch.randn(input_size, requires_grad=True)

    # Standard Conv2d for comparison
    standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    y_std = standard_conv(x)
    print(f"Standard Conv2d Output Shape: {y_std.shape}")
    print(f"Standard Conv2d Param Count: {sum(p.numel() for p in standard_conv.parameters())}")

    # LowRankConv2d
    lowrank_conv = LowRankConv2d(in_channels, out_channels, kernel_size, rank, stride, padding)
    y_lr = lowrank_conv(x)
    print(f"LowRankConv2d Output Shape: {y_lr.shape}")
    print(f"LowRankConv2d Param Count: {sum(p.numel() for p in lowrank_conv.parameters())}")

    # Verify gradient flow
    loss = y_lr.mean()
    loss.backward()
    print("Backward pass successful: gradients computed.")


if __name__ == '__main__':
    test_low_rank_conv2d()