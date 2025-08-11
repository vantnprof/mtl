import torch.nn as nn
from src.mlconv.mlconv_bottleneck import MLconvConv
import torch


class MLconvSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.mlconv1 = MLconvConv(c1, c_, 1, 1)
        self.mlconv2 = MLconvConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.mlconv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.mlconv2(torch.cat(y, 1))