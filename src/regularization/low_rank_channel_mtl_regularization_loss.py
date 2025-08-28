import torch.nn as nn
import torch
from src.lowrank_channel_mtl.lowrank_channel_mtl import LowRankChannelMTL


def group_lasso_regularization(model: nn.Module):
    reg_loss = 0.0
    for _, module in model.named_modules():
        if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
            A = module.A3 
            h = A.shape[1]
            penalty = A.shape[0]*h/torch.norm(A, p='fro')
            reg_loss += penalty * torch.norm(A, dim=0).sum()
    return reg_loss