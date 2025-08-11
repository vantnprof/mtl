import torch


def mars_loss(output, target, mask_logits, alpha=1e-4):
    task_loss = nn.CrossEntropyLoss()(output, target)
    sparsity_loss = alpha * torch.sum(torch.sigmoid(mask_logits))
    return task_loss + sparsity_loss