import math
import torch
from torch import nn

class StandardNormal(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self):
        super(StandardNormal, self).__init__()
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, shape):
        return torch.randn(*shape, device=self.buffer.device, dtype=self.buffer.dtype)

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)