import torch.nn as nn
import torch.nn.functional as F
from . import regist_model
from .NMFlowGAN import NMFlowGANDenoiser

@regist_model
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=18,features=64):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = kernel_size//2
        self.bias=True
        self.residual=True
        layers = list()
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=self.bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=self.bias))
            layers.append(nn.BatchNorm2d(features, momentum=0.9, eps=1e-04, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=self.bias))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x, kwargs=None):
        if self.residual:
            out = x-self.dncnn(x)
        else:
            out = self.dncnn(x)
        return out
    
@regist_model
class DnCNNFlowGAN(NMFlowGANDenoiser):
    def __init__(
        self,
        kwargs_dncnn,
        kwargs_unet,
        kwargs_flow,
        pretrained_path,
        num_bits=8
        ):
        super().__init__(
            DnCNN(**kwargs_dncnn),
            kwargs_flow,
            kwargs_unet,
            pretrained_path,
            num_bits,
        )