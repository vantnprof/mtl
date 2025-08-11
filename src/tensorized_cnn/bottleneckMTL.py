import torch.nn as nn
from src.mtl import MultilinearTransformationLayer as MTL


class BottleneckMTL(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckMTL, self).__init__()
        self.conv1 = MTL(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = MTL(out_channels, out_channels, kernel_size=(3, 3),
                                      stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = MTL(out_channels, out_channels * self.expansion, kernel_size=(1, 1), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if out.shape != identity.shape:
            raise ValueError(f"Output shape {out.shape} and identity shape {identity.shape} do not match.")

        out += identity
        out = self.relu(out)
        
        return out
