from . import regist_loss
import torch
import torch.nn as nn
from util.standard_normal_dist import StandardNormal

@regist_loss
class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = StandardNormal()

    def forward(self, input_data, model_output, data, module):
        z = model_output['z']
        ldj = model_output['ldj']

        log_z = self.dist.log_prob(z)
        objectives = ldj + log_z
        return torch.mean(-objectives)

@regist_loss
class std_z(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = StandardNormal()

    def forward(self, input_data, model_output, data, module):
        z = model_output['z']
        var_z = torch.var(z, dim=[1,2,3])
        sd_z = torch.mean(torch.sqrt(var_z))
        return sd_z