import torch
import torch.nn as nn
from src.lowrank_channel_mtl.lowrank_channel_mtl_bottleneck import LowRankChannelMTLConv, LowRankChannelMTLConvBottleneck


class LowRankChannelMTLConvC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.lowrank1 = LowRankChannelMTLConv(c1, 2 * self.c, 1, 1)
        self.lowrank2 = LowRankChannelMTLConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(LowRankChannelMTLConvBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        B, C, H, W = x.shape
        y = list(self.lowrank1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.lowrank2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.lowrank1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        B, C, H, W = (torch.cat(y, 1)).shape
        return self.lowrank2(torch.cat(y, 1))


class LowRankChannelMTLConvC3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.lowrank1 = LowRankChannelMTLConv(c1, c_, 1, 1)
        self.lowrank2 = LowRankChannelMTLConv(c1, c_, 1, 1)
        self.lowrank3 = LowRankChannelMTLConv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(LowRankChannelMTLConvBottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.lowrank3(torch.cat((self.m(self.lowrank1(x)), self.lowrank2(x)), 1))
    

class LowRankChannelMTLConvC3k(LowRankChannelMTLConvC2f):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(LowRankChannelMTLConvBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class LowRankChannelMTLConvC3k2(LowRankChannelMTLConvC2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            LowRankChannelMTLConvC3k(self.c, self.c, 2, shortcut, g) if c3k else LowRankChannelMTLConvBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )