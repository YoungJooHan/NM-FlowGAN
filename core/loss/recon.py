from . import regist_loss
import torch
import torch.nn as nn


def _mse(x, y, level=2):
    assert x.shape == y.shape
    err = (x - y) ** level
    err = torch.abs(err)
    return err.mean()


@regist_loss
class L1Loss(nn.Module):
    def forward(self, input_data, model_output, data, module):
        fx = model_output['recon']
        y = data['clean']
        return _mse(fx, y, level=1)


@regist_loss
class L2Loss(nn.Module):
    def forward(self, input_data, model_output, data, module):
        fx = model_output['recon']
        y = data['clean']
        return _mse(fx, y, level=2)
