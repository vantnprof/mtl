import torch.nn as nn
from src.lowrank_channel_mtl.lowrank_channel_mtl import LowRankChannelMTL
import torch


def do_clipping(model: nn.Module, threshold: float = 1e-4, use_absolute: bool=False):
    h_dict = {
        'total': 0
    }
    print("[Clipping] Current threshold: {}".format(threshold))
    with torch.no_grad():
        for name, module in model.named_modules():
            # print("Nihao...")
            # input()
            if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
                A, B = module.A3.data, module.B3.data
                col_norms = torch.norm(A, dim=0)

                print(f"[DEBUG] Norm range in {name}: min={col_norms.min():.4f}, max={col_norms.max():.4f}, mean={col_norms.mean():.4f}")

                if use_absolute:
                    mask = col_norms > threshold
                else:
                    max_norm = col_norms.max()
                    mask = col_norms > threshold * max_norm

                old_h = A.shape[1]
                new_h = int(mask.sum().item())

                h_dict[name] = new_h
                h_dict['total'] += new_h
                
                if 1 <= new_h < old_h:
                    print(f"[Clipping] {name}: rank {old_h} → {new_h}")
                    module.A3 = nn.Parameter(A[:, mask])
                    module.B3 = nn.Parameter(B[mask, :])
                else:
                    print(f"[Clipping] {name}: no change (rank {old_h})")
    return h_dict


def compute_excess_parameters(model: nn.Module):
    """
    Compute excess parameters E(n) for each LowRankChannelMTL layer after clipping.

    Returns:
        total_excess: total excess across all layers
        layer_info: list of (layer_name, E(n), J(n), K(n), h) for inspection
    """
    total_excess = 0
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, LowRankChannelMTL) and hasattr(module, 'A3') and module.A3 is not None:
            C_out, h = module.A3.shape
            h_check, C_in = module.B3.shape

            assert h == h_check, f"A3 columns and B3 rows mismatch in {name}"

            K = C_out + C_in
            J = C_out * C_in
            E = J - K * h

            total_excess += E
            layer_info.append((name, E, J, K, h))

    return total_excess, layer_info