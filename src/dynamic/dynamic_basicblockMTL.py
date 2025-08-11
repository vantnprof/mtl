import torch.nn as nn
from src.dynamic.dynamic_mtl import DynamicMTL

class DynamicBasicBlockMTL(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(DynamicBasicBlockMTL, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.mtl1 = DynamicMTL(
            input_channels=in_channels,
            output_channels=out_channels,
            target_params=3**2*in_channels*out_channels
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mtl2 = DynamicMTL(
            input_channels=out_channels,
            output_channels=out_channels,
            target_params=3**2*out_channels*out_channels
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample if needed (channels or spatial size mismatch)
        # if downsample is None and (stride != 1 or in_channels != out_channels):
        #     self.downsample = nn.Sequential(
        #         DynamicMTL(
        #             input_channels=in_channels,
        #             output_channels=output_channels,
        #             target_params=
        #         ),
        #         nn.BatchNorm2d(out_channels),
        #     )
        # else:
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.mtl1(x)))
        if self.stride != 1:
            out = nn.functional.avg_pool2d(out, kernel_size=2, stride=self.stride)
        out = self.bn2(self.mtl2(out))

        # If downsample is defined, apply it to identity
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

if __name__ == "__main__":
    import torch

    batch_size = 1
    input_channels = 10
    input_height = 64
    input_width = 64
    output_channels = 120
    output_height = 122
    output_width = 122
    max_rank = 120

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    input_tensor = input_tensor.to(device)
    print("Expected output shape:", (batch_size, output_channels, output_height, output_width))

    tensorized_conv = DynamicBasicBlockMTL(
        in_channels=input_channels,
        out_channels=output_channels,
        
    )
    tensorized_conv = tensorized_conv.to(device)
    output_tensor = tensorized_conv(input_tensor)
    print("Input shape :", input_tensor.shape)
    print("Final output shape:", output_tensor.shape)