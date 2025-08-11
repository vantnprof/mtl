import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, expansion is 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Downsample layer to match dimensions
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # Save input for the residual connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual connection
        out += identity
        out = self.relu(out)

        return out
