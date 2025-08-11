from torch import nn
import torch
from src.mtl.mtl import MTL
from typing import Tuple


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MTLConv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.mtl = MTL(c1, c2, k, s, autopad(k, p, d), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        B, C, H, W = x.shape
        if not self.mtl.initialize or H != self.mtl.U1.shape[1] or W != self.mtl.U2.shape[1] or C != self.mtl.U3.shape[1]:
            self.mtl._init_weights(H, W, dtype=x.dtype, device=x.device, C=C)

        return self.act(self.bn(self.mtl(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        B, C, H, W = x.shape
        # if not self.mtl.initialize or H != self.mtl.U1.shape[1] or W != self.mtl.U2.shape[1] or C != self.mtl.U3.shape[1]:
        #     self.mtl._init_weights(H, W, dtype=x.dtype, device=x.device, C=C)
        return self.act(self.mtl(x))
    

class MTLBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.mtl1 = MTLConv(c1, c_, k[0], 1)
        self.mtl2 = MTLConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        # B, C, H, W = x.shape
        # if not self.mtl1.mtl.initialize or H != self.mtl1.mtl.U1.shape[1] or W != self.mtl1.U2.shape[1] or C != self.mtl1.U3.shape[1]:
        #     self.mtl1.mtl._init_weights(H, W, dtype=x.dtype, device=x.device, C=C)
        # if not self.mtl2.initialize or H != self.mtl2.U1.shape[1] or W != self.mtl2.U2.shape[1] or C != self.mtl2.U3.shape[1]:
        #     self.mtl2.mtl._init_weights(H, W, dtype=x.dtype, device=x.device, C=C)
        return x + self.mtl2(self.mtl1(x)) if self.add else self.mtl2(self.mtl1(x))
    