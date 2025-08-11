# test_multilinear_transformation.py

import unittest
import torch
from src.mtl import MultilinearTransformationLayer

class TestMultilinearTransformationLayer(unittest.TestCase):
    def setUp(self):
        # Define input parameters
        self.batch_size = 100
        self.input_channels = 3
        self.height = 32
        self.width = 32
        self.output_channels = 5
        self.kernel_size = (5, 5)
        self.stride = 1
        self.padding = 1

        # Create a random input tensor
        self.input_tensor = torch.randn(
            self.batch_size, self.input_channels, self.height, self.width, requires_grad=True
        )

        # Initialize the MTL layer
        self.mtl = MultilinearTransformationLayer(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

    def test_output_shape(self):
        # Forward pass
        output_tensor = self.mtl(self.input_tensor)

        # Calculate expected output dimensions
        expected_height = (self.height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        expected_width = (self.width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1

        # Assert output shape
        expected_shape = (self.batch_size, self.output_channels, expected_height, expected_width)
        self.assertEqual(output_tensor.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output_tensor.shape}")

    def test_gradient_computation(self):
        # Forward pass
        output_tensor = self.mtl(self.input_tensor)

        # Compute a simple loss (sum of outputs)
        loss = output_tensor.sum()

        # Backward pass
        loss.backward()

        # Assert that gradients are computed for input
        self.assertIsNotNone(self.input_tensor.grad, "Gradients not computed for input tensor")

        # Assert that gradients are computed for all parameters
        for name, param in self.mtl.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradients not computed for parameter {name}")

if __name__ == "__main__":
    unittest.main()