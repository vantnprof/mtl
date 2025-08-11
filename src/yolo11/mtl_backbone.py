# mtl_backbone.py
import torch.nn as nn
from src.mtl.cnn2mtl import MTL as MTLLayer


class MTLBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, 
                 kernel_size=3, stride=1, padding=1, bias=False):
        super(MTLBackbone, self).__init__()
        # Define your MTL layers here
        self.mtl = MTLLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        x = self.mtl(x)
        return x