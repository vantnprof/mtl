import torch
import torch.nn as nn

class MARSWrapper(nn.Module):
    def __init__(self, tensor_layer, mask_shape, temperature=1.0, hard=False):
        super(MARSWrapper, self).__init__()
        self.tensor_layer = tensor_layer
        self.temperature = temperature
        self.hard = hard
        self.mask_logits = nn.Parameter(torch.zeros(mask_shape))

    def sample_mask(self):
        uniform_noise = torch.rand_like(self.mask_logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
        logits = self.mask_logits + gumbel_noise
        probs = torch.sigmoid(logits / self.temperature)
        return (probs > 0.5).float() if self.hard else probs

    def forward(self, x):
        mask = self.sample_mask()
        return self.tensor_layer(x, mask=mask)