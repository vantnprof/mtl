import torch
import torch.nn as nn
from typing import Tuple


class MLConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Tuple[int]=(3, 3),
                 stride: Tuple[int, int]=(1, 1),
                 padding: Tuple[int, int]=(0, 0),
                 bias: bool=False, 
                 rank:int=350,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.rank = rank
        self.k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        self.s = self.stride if isinstance(self.stride, int) else self.stride[0]
        self.p = self.padding if isinstance(self.padding, int) else self.padding[0]
        self.pointwise_in = nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.rank,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)
        self.depthwise = nn.Conv2d(in_channels=self.rank,
                                   out_channels=self.rank,
                                   kernel_size=self.k,
                                   stride=self.s,
                                   padding=self.p,
                                   groups=self.rank,
                                   bias=False)
        self.pointwise_out = nn.Conv2d(in_channels=self.rank,
                                       out_channels=self.out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.depthwise(x)
        x = self.pointwise_out(x)
        return x


if __name__ == "__main__":
    import torch.nn.functional as F
    import tensorly as tl
    from tensorly.decomposition import parafac
    import matplotlib.pyplot as plt

    from tensorly.cp_tensor import cp_to_tensor as kruskal_to_tensor

    # Set TensorLy to use PyTorch backend
    tl.set_backend('pytorch')

    in_channels = 3
    out_channels = 6
    kernel_size = 3
    stride = 1
    padding = 1

    # Set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move input tensor to CUDA device
    input_tensor = torch.randn(1, in_channels, 32, 32).to(device)

    # Initialize Conv2d and move it to CUDA device
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False).to(device)
    conv_params = sum(p.numel() for p in conv.parameters())

    # Measure inference time for Conv2d
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    _ = conv(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    conv_inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds

    ranks = list(range(1, out_channels * 2 + 1))
    errors = []
    ratios = []
    mlconv_inference_times = []

    weight_tensor = conv.weight.data.permute(2, 3, 1, 0)
    original_weight_frobenius = torch.norm(weight_tensor) ** 2

    for rank in ranks:
        factors = parafac(weight_tensor, rank=rank, init='svd')
        reconstructed = kruskal_to_tensor(factors)
        error = torch.norm(weight_tensor - reconstructed) ** 2 / original_weight_frobenius
        errors.append(error.item())

        spatial_1, spatial_2, input_channel, output_channel = factors.factors

        # Initialize MLConv and move it to CUDA device
        mlconv = MLConv(in_channels, out_channels, kernel_size, padding, stride, rank).to(device)
        mlconv.pointwise_in.weight.data = input_channel.T.unsqueeze(-1).unsqueeze(-1)
        depthwise_weights = torch.einsum('ir,jr->rij', spatial_1, spatial_2)
        mlconv.depthwise.weight.data = depthwise_weights.unsqueeze(1)
        mlconv.pointwise_out.weight.data = output_channel.unsqueeze(-1).unsqueeze(-1)

        # Measure inference time for MLConv
        torch.cuda.synchronize()
        start_event.record()
        _ = mlconv(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        mlconv_inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        mlconv_inference_times.append(mlconv_inference_time)

        mlconv_params = sum(p.numel() for p in mlconv.parameters())
        ratios.append(mlconv_params / conv_params)

    first_rank = ranks[0]
    first_error = errors[0]
    first_ratio = ratios[0]

    print(f"Conv2d Inference Time: {conv_inference_time:.4f} ms")
    print(f"Rank 1: Relative Error = {first_error:.4f}, Param Ratio = {first_ratio:.4f}, MLConv Inference Time = {mlconv_inference_times[0]:.4f} ms")

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('CP Rank')
    ax1.set_ylabel('Relative Error', color=color)
    ax1.plot(ranks, errors, color=color, label='Relative Error')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Parameter Ratio (MLConv / Conv2d)', color=color)
    ax2.plot(ranks, ratios, color=color, label='Param Ratio')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Approximation Error and Parameter Ratio vs. CP Rank", pad=20)
    plt.savefig("./src/mlconv/approximate-ratio_and_number_of_parameter_versus_rank.pdf")

    # Print average MLConv inference time across ranks
    avg_mlconv_time = sum(mlconv_inference_times) / len(mlconv_inference_times)
    print(f"Average MLConv Inference Time: {avg_mlconv_time:.4f} ms")
