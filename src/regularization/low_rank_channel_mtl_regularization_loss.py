import torch.nn as nn
import torch
from src.lowrank_channel_mtl.lowrank_channel_mtl import LowRankChannelMTL
import math
import numpy as np


def group_lasso_regularization(model: nn.Module):
    reg_loss = 0.0
    # sum_all_A_norm = 0
    # sum_all_mtl_params = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
    #         A = module.A3 
    #         B = module.B3
    #         h = A.shape[1]
    #         sum_all_mtl_params += (A.shape[0]+B.shape[1])*h
            # sum_all_A_norm += torch.norm(A, p='fro')
    for name, module in model.named_modules():
        if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
            A = module.A3 
            B = module.B3
            h = A.shape[1]
            penalty = A.shape[0]*h/torch.norm(A, p='fro')
            reg_loss += penalty * torch.norm(A, dim=0).sum()
    return reg_loss


# def group_lasso_regularization(model: nn.Module, normalize: bool = True):
#     reg_loss = 0.0
#     all_norms = []

#     for name, module in model.named_modules():
#         if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
#             A = module.A3  # shape: (out_channels, h)
#             B = module.B3  # shape: (h, in_channels)

#             U3 = A @ B  # Effective conv weight: shape (out_channels, in_channels)
#             group_norms = torch.norm(U3, dim=1, p=2)  # L2 norm per output channel

#             if normalize:
#                 norm_factor = torch.norm(group_norms, p=2).detach() + 1e-6
#                 group_norms = group_norms / norm_factor

#             reg_loss += group_norms.sum()
#             all_norms.append(group_norms.sum().item())

#     return reg_loss