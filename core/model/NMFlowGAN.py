import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator
from torch.nn.parameter import Parameter

from core.model import regist_model, get_model_class
from .NMFlow import NMFlow
from .UNet import UNet

import os 

@regist_model
class NMFlowGANGenerator(NMFlow):
    def __init__(
        self,
        kwargs_unet,
        kwargs_flow,
    ):
        super(NMFlowGANGenerator, self).__init__(
            **kwargs_flow
        )
        self.generator = UNet(
             **kwargs_unet
        )

    def forward(self, noisy, clean, kwargs=dict()):
        z, objectives = super().forward(noisy, clean, kwargs)
        kwargs['clean']=clean
        with torch.no_grad():
            x = super().sample(kwargs) - kwargs['clean'] 
        x_scaled = x / (2**self.num_bits) # x_scaled: -1 ~ 1
        y = (self.generator(x_scaled) * (2**self.num_bits) + kwargs['clean']).requires_grad_(True)
        return z, objectives, y, x
    
    def sample(self, kwargs=dict()):
        x = super().sample(kwargs) - kwargs['clean'] # pixelwise noise
        x_scaled = x / (2**self.num_bits) # x_scaled: -1 ~ 1
        y = self.generator(x_scaled) * (2**self.num_bits) + kwargs['clean']
        y = torch.clip(y, 0, 2**self.num_bits)
        return y

@regist_model
class NMFlowGANCritic(nn.Module):
    def __init__(
            self,
            in_ch=1,
            nc=64,
            num_bits=8
    ):
        super(NMFlowGANCritic, self).__init__()
        self.num_bits = num_bits
        self.critic = Discriminator_96(in_ch, nc)

    def forward(self, x):
         x_scaled = x / (2**self.num_bits)
         return self.critic(x_scaled)

class Discriminator_96(nn.Module):
    """Discriminator with 96x96 input, refer to Kai Zhang, https://github.com/cszn/KAIR"""
    def __init__(self, in_nc=3, nc=64):
        super(Discriminator_96, self).__init__()
        conv0 = nn.Conv2d(in_nc, nc, kernel_size=7, padding=3)
        conv1 = self._get_basic_module(nc, nc, kernel_size=4, stride=2)
        # 48, 64
        conv2 = self._get_basic_module(nc, nc*2, kernel_size=3, stride=1)
        conv3 = self._get_basic_module(nc*2, nc*2, kernel_size=4, stride=2)
        # 24, 128
        conv4 = self._get_basic_module(nc*2, nc*4, kernel_size=3, stride=1)
        conv5 = self._get_basic_module(nc*4, nc*4, kernel_size=4, stride=2)
        # 12, 256
        conv6 = self._get_basic_module(nc*4, nc*8, kernel_size=3, stride=1)
        conv7 = self._get_basic_module(nc*8, nc*8, kernel_size=4, stride=2)
        # 6, 512
        conv8 = self._get_basic_module(nc*8, nc*8, kernel_size=3, stride=1)
        conv9 = self._get_basic_module(nc*8, nc*8, kernel_size=4, stride=2)
        # 3, 512
        self.features = nn.Sequential(*[conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def _get_basic_module(self, in_ch, out_ch, kernel_size=1, stride=1, padding=1, negative_slope=0.2):
            return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.InstanceNorm2d(out_ch, affine=True), #batch normalization?
                    nn.LeakyReLU(negative_slope, inplace=True)
            )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class NMFlowGANDenoiser(nn.Module):
    def __init__(
            self,
            denoiser,
            kwargs_flow,
            kwargs_unet,
            pretrained_path,
            num_bits=14,
        ):
        super().__init__()
        self.denoiser = denoiser
        self.kwargs_flow = kwargs_flow
        self.pretrained_path = pretrained_path
        self.num_bits = num_bits
        self.noise_model = get_model_class("NMFlowGANGenerator")(kwargs_unet, kwargs_flow)
        self._load_checkpoint(self.noise_model, self.pretrained_path)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.denoiser.parameters(recurse) # the parameters of denoiser will be trained only.
    
    def _load_checkpoint(self, module, path):
        if not os.path.exists(path):
            print(os.path.exists(path), f"WARNING: {path} is not exist.")
            return
        pth = torch.load(path)
        module.load_state_dict(pth['model_weight']['generator'])
        module.eval()
        
    def forward(self, x, kwargs=dict()):
        # x: clean image
        x_scaled = x / (2**self.num_bits) # x_scaled: 0 ~ 1
        x_scaled = x_scaled * (2**self.noise_model.num_bits) # x_scaled: 0 ~ noise model's max GL.
        
        kwargs['clean'] = x_scaled
        with torch.no_grad(): 
            n = self.noise_model.sample(kwargs) # noisy image

        n_scaled = n / (2**self.noise_model.num_bits) # n_scaled: 0 ~ 1
        n_scaled = torch.clip(n_scaled, 0., 1.)
        y = self.denoiser(n_scaled)
        y = y * (2**self.num_bits) # y: 0 ~ denoiser's max GL.
        return y
    
    def denoise(self, x, kwargs=None):
        # x: noisy image
        if kwargs is None or 'num_bits' not in kwargs: num_bits = self.num_bits
        else: num_bits = kwargs['num_bits']

        x_scaled = x / (2**num_bits) # x_scaled: 0 ~ 1
        y =  self.denoiser(x_scaled) 
        y = torch.clip(y, 0., 1.)
        y *= (2**num_bits) # x_scaled: 0 ~ denoiser's max GL.
        return y
    
    def sample(self, x, kwargs=None):
        # x: clean image
        if kwargs is None or 'num_bits' not in kwargs: num_bits = self.num_bits
        else: num_bits = kwargs['num_bits']

        x_scaled = x / (2**num_bits) # x_scaled: 0 ~ 1
        x_scaled = x_scaled * (2**self.noise_model.num_bits) # x_scaled: 0 ~ noise model's max GL.

        kwargs = dict()
        kwargs['clean'] = x_scaled
        n = self.noise_model.sample(kwargs) # n: 0 ~ noise model's max GL.
        
        n_scaled = n / (2**self.noise_model.num_bits) # n_scaled: 0 ~ 1
        n_scaled = torch.clip(n_scaled, 0., 1.)
        n_scaled = n_scaled * (2**num_bits) # n_scaled: 0 ~ denoiser's max GL.
        return n_scaled