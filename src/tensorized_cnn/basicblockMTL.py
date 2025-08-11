import torch.nn as nn
from src.mtl import MultilinearTransformationLayer as MTL

class BasicBlockMTL(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockMTL, self).__init__()
        self.conv1 = MTL(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MTL(
            input_channels=out_channels,
            output_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    