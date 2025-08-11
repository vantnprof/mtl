import torch.nn as nn
from src.dynamic.dynamic_mtl import DynamicMTL

class DynamicBottleNeckMTL(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None):
        super(DynamicBottleNeckMTL, self).__init__()
        # 1x1 DynamicMTL to reduce channels
        self.mtl1 = DynamicMTL(
            input_channels=in_channels,
            output_channels=out_channels,
            target_params=1**2*in_channels*out_channels
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 DynamicMTL (spatial)
        self.mtl2 = DynamicMTL(
            input_channels=out_channels,
            output_channels=out_channels,
            target_params=3**2*out_channels*out_channels
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 DynamicMTL to expand channels
        self.mtl3 = DynamicMTL(
            input_channels=out_channels,
            output_channels=out_channels*self.expansion,
            target_params=1**2*out_channels*out_channels*self.expansion
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.mtl1(x)))
        out = self.relu(self.bn2(self.mtl2(out)))
        out = self.bn3(self.mtl3(out))

        # If shape mismatch, interpolate and project identity
        if identity.shape != out.shape:
            identity = nn.functional.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
            if identity.shape[1] != out.shape[1]:
                conv = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1, stride=1, bias=False).to(identity.device)
                identity = conv(identity)

        out += identity
        out = self.relu(out)
        return out

if __name__ == "__main__":
    import torch

    batch_size = 2
    in_channels = 16
    input_height = 32
    input_width = 32
    out_channels = 32

    model = DynamicBottleNeckMTL(
        in_channels=in_channels,
        out_channels=out_channels,
    )

    x = torch.randn(batch_size, in_channels, input_height, input_width)
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)