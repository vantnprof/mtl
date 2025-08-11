# import torch
# import torch.nn as nn
# from typing import Tuple
# from src.mtl.mtl import MTL


# class LowRankMTL(nn.Module):
#     """
#     Implements a multilinear transformation layer with low-rank decomposition:
#     O = X ×₁ U1 ×₂ U2 ×₃ U3
#     where U1 = sum_k H1_k V1_k^T, U2 = sum_k H2_k V2_k^T, U3 = sum_k H3_k V3_k^T.
#     """
#     def __init__(self, in_channels: int, out_channels: int,
#                  kernel_size: Tuple[int]=(3, 3),
#                  stride: Tuple[int, int]=(1, 1),
#                  padding: Tuple[int, int]=(0, 0),
#                  bias: bool=False,
#                  rank: int=1):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.bias = bias
#         self.rank = rank
#         self.k = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
#         self.s = self.stride[0] if isinstance(self.stride, tuple) else self.stride
#         self.p = self.padding[0] if isinstance(self.padding, tuple) else self.padding

#         self.mtl1 = MTL(
#             in_channels=self.in_channels,
#             out_channels=self.rank,
#             kernel_size=(self.k, self.k),
#             stride=(self.s, self.s),
#             padding=(self.p, self.p),
#             bias=False
#         )

#         self.mtl2 = MTL(
#             in_channels=self.rank,
#             out_channels=self.out_channels,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#             bias=False
#         )

#     def forward(self, X: torch.Tensor):
#         B, C_in, H, W = X.shape

#         out = self.mtl1(X)

#         out = self.mtl2(out)

#         return out

#     def __repr__(self):
#         return (f"{self.__class__.__name__}(in_channels={self.in_channels}, "
#                 f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
#                 f"stride={self.stride}, padding={self.padding})")
    
#     def __str__(self):
#         return str(self.__repr__())
import torch
import torch.nn as nn
from typing import Tuple


class LowRankMTL(nn.Module):
    """
    Implements a multilinear transformation layer with true low-rank Tucker decomposition.
    This is a two-stage process:
    1. Analysis: Project X into a small core tensor G.
       G = X ×₁ V₁ᵀ ×₂ V₂ᵀ ×₃ V₃ᵀ
    2. Synthesis: Expand G into the output tensor Y.
       Y = G ×₁ H₁ ×₂ H₂ ×₃ H₃
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 bias: bool = False, # Bias is not implemented in this version for simplicity
                 rank: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        # Ensure kernel, stride, padding are integers for calculation
        self.k = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.s = stride[0] if isinstance(stride, tuple) else stride
        self.p = padding[0] if isinstance(padding, tuple) else padding

        # --- Parameters for Stage 1 (Analysis / Projection) ---
        # These correspond to the V matrices, mapping input dims to the rank.
        # User request: U1 in R^(rank x H_in) -> This is V₁ᵀ
        self.V1 = nn.Parameter(torch.empty(0)) # Shape: (H_in, rank)
        self.V2 = nn.Parameter(torch.empty(0)) # Shape: (W_in, rank)
        self.V3 = nn.Parameter(torch.empty(0)) # Shape: (C_in, rank)

        # --- Parameters for Stage 2 (Synthesis / Expansion) ---
        # These correspond to the H matrices, mapping rank to output dims.
        # User request: U1 in R^(H_out x rank) -> This is H₁
        self.H1 = nn.Parameter(torch.empty(0)) # Shape: (H_out, rank)
        self.H2 = nn.Parameter(torch.empty(0)) # Shape: (W_out, rank)
        self.H3 = nn.Parameter(torch.empty(self.out_channels, self.rank)) # Shape: (C_out, rank)

        self.initialized = False

    def _init_weights(self, H_in: int, W_in: int, C_in: int, dtype: torch.dtype, device: torch.device):
        """Lazy initialization of weights based on the first input tensor."""
        # Calculate output spatial dimensions
        H_out = (H_in + 2 * self.p - self.k) // self.s + 1
        W_out = (W_in + 2 * self.p - self.k) // self.s + 1

        # Initialize Analysis (V) matrices
        self.V1.data = torch.randn(H_in, self.rank, dtype=dtype, device=device) * 0.02
        self.V2.data = torch.randn(W_in, self.rank, dtype=dtype, device=device) * 0.02
        self.V3.data = torch.randn(C_in, self.rank, dtype=dtype, device=device) * 0.02

        # Initialize Synthesis (H) matrices
        self.H1.data = torch.randn(H_out, self.rank, dtype=dtype, device=device) * 0.02
        self.H2.data = torch.randn(W_out, self.rank, dtype=dtype, device=device) * 0.02
        # H3 is already partially defined, just need to re-initialize data
        self.H3.data = torch.randn(self.out_channels, self.rank, dtype=dtype, device=device) * 0.02
        
        self.initialized = True
        print(f"LowRankMTL weights initialized for input H={H_in}, W={W_in}, C={C_in}")

    def forward(self, X: torch.Tensor):
        B, C_in, H_in, W_in = X.shape

        # Lazy initialize weights on first forward pass or if input shape changes
        if not self.initialized or H_in != self.V1.shape[0] or W_in != self.V2.shape[0] or C_in != self.V3.shape[0]:
            self._init_weights(H_in, W_in, C_in, X.dtype, X.device)

        # --- Stage 1: Analysis ---
        # Project X into the core tensor G using V matrices.
        # G = X ×₁ V₁ᵀ ×₂ V₂ᵀ ×₃ V₃ᵀ
        # The einsum operation below performs all 3 contractions at once.
        # bchw (X), cr (V3), hr (V1), wr (V2) -> bijk (core_tensor)
        # i, j, k are the new dimensions of size 'rank'
        core_tensor = torch.einsum('bchw,ci,hj,wk -> bijk', X, self.V3, self.V1, self.V2)

        # --- Stage 2: Synthesis ---
        # Expand the core tensor G into the output Y using H matrices.
        # Y = G ×₁ H₁ ×₂ H₂ ×₃ H₃
        # bijk (core_tensor), oi (H1), pj (H2), qi (H3) -> bqop (output)
        # q, o, p are the new dimensions C_out, H_out, W_out
        output = torch.einsum('bijk,oi,pj,qk -> bqop', core_tensor, self.H1, self.H2, self.H3)

        return output

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, rank={self.rank}, kernel_size=({self.k},{self.k}), "
                f"stride=({self.s},{self.s}), padding=({self.p},{self.p}))")

    def __str__(self):
        return str(self.__repr__())
    
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from src.mtl.mtl import MTL
    
    # Sample input: batch size = 2, channels = 64, height = 512, width = 512
    x = torch.randn(2, 64, 512, 512)

    # Parameters
    in_channels = 64
    out_channels = 128
    kernel_size = (3, 3)
    stride = (3, 3)
    padding = (1, 1)
    rank = 9

    # LowRankMTL
    mtl_layer = LowRankMTL(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        rank=rank
    )
    mtl_output = mtl_layer(x)
    print("LowRankMTL output shape:", mtl_output.shape)
    mtl_params = sum(p.numel() for p in mtl_layer.parameters())
    print("LowRankMTL parameter count:", mtl_params)
    print("LowRankMTL trainable parameters:")
    for name, param in mtl_layer.named_parameters():
        print(f"  {name}: {param.shape}")

    # Backward check
    mtl_output.mean().backward()
    print("LowRankMTL backward pass successful.")

    # MTL for comparison
    mtl_true_layer = MTL(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    mtl_true_output = mtl_true_layer(x)
    print("MTL output shape:", mtl_true_output.shape)
    mtl_true_params = sum(p.numel() for p in mtl_true_layer.parameters())
    print("MTL parameter count:", mtl_true_params)
    print("MTL trainable parameters:")
    for name, param in mtl_true_layer.named_parameters():
        print(f"  {name}: {param.shape}")

    # Compare
    print("Shapes match:", mtl_output.shape == mtl_true_output.shape)