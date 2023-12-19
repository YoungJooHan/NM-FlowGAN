import torch
from torch import nn
from . import regist_layer

@regist_layer
class UniformDequantization(nn.Module):
    def __init__(self, num_bits=8, device='cpu', name='uniform_dequantization'):
        super(UniformDequantization, self).__init__()
        self.num_bits = num_bits
        self.quantization_bins = 2**num_bits
        self.register_buffer(
            'ldj_per_dim',
            - num_bits * torch.log(torch.tensor(2, device=device, dtype=torch.float))
        )
        self.name = name

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def _inverse(self, z, **kwargs):
        z = self.quantization_bins * z
        return z.floor().clamp(min=0, max=self.quantization_bins-1)

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        u = torch.rand(x.shape, device=self.ldj_per_dim.device, dtype=self.ldj_per_dim.dtype)
        z = (x.type(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape)
        return z, ldj
