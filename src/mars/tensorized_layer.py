import torch
import torch.nn as nn


class TensorizedLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(TensorizedLayer, self).__init__()
        in_channels, in_height, in_width = input_shape
        out_channels, out_height, out_width = output_shape

        # Define a convolutional layer to handle image data
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # If output dimensions differ from input, adjust using interpolation
        self.output_size = (out_height, out_width)

    def forward(self, x, mask=None):
        # Apply convolution
        x = self.conv(x)

        # If mask is provided, apply it to the output
        if mask is not None:
            x = x * mask

        # Resize output to match desired output dimensions
        x = nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x