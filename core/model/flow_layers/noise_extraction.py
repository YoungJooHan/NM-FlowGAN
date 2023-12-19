import torch
from torch import nn
from . import regist_layer

@regist_layer
class NoiseExtraction(nn.Module):
    def __init__(self, device='cpu', name='noise_extraction'):
        super(NoiseExtraction, self).__init__()
        self.name = name
        self.device = device

    def _inverse(self, z, **kwargs):
        x = z + kwargs['clean']
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        z = x - kwargs['clean']
        ldj = torch.zeros(x.shape[0], device=self.device)
        return z, ldj
