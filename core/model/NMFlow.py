from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from core.model import regist_model, get_model_class
from core.model.flow_layers import get_flow_layer
from util.standard_normal_dist import StandardNormal
import os

@regist_model
class NMFlow(nn.Module):
    def __init__(
        self,
        in_ch=1,
        ch_exp_coef = 1.,
        width_exp_coef = 2.,
        num_bits=16,
        conv_net_feats=32,
        pre_arch="CD|UD",
        arch="NE|AN|C|CAC|AN|C|CAC|AN|C|CAC|AN|C|CAC|AN|C|CAC" # glow-style model
    ):
        super(NMFlow, self).__init__()
        self.num_bits=num_bits

        self.in_ch = in_ch
        self.ch_exp_coef = ch_exp_coef
        self.width_exp_coef = width_exp_coef
        self.conv_net_feats = conv_net_feats

        self.pre_bijectors = list()
        pre_arch_lyrs = pre_arch.split('|')
        for lyr in pre_arch_lyrs:
            self.pre_bijectors.append(self.get_flow_layer(lyr))
        self.pre_bijectors = nn.Sequential(*self.pre_bijectors)

        self.bijectors = list()
        arch_lyrs = arch.split('|')
        for lyr in arch_lyrs:
            self.bijectors.append(self.get_flow_layer(lyr))
        self.bijectors = nn.Sequential(*self.bijectors)
        self.dist = StandardNormal()

    def internal_channels(self):
        return int(self.in_ch * self.ch_exp_coef)
    
    def internal_widths(self):
        return int(self.in_ch * self.width_exp_coef)

    def get_flow_layer(self, name):
        if name == "UD":
            return get_flow_layer("UniformDequantization")(device='cuda', num_bits=self.num_bits)
        elif name == "NE":
            return get_flow_layer("NoiseExtraction")(device='cuda')
        elif name == "CL2":
            return get_flow_layer("ConditionalLinearExp2")(
                in_ch=self.internal_channels(),
                device='cuda'
            )
        elif name == "SDL":
            return get_flow_layer("SignalDependentConditionalLinear")(
                meta_encoder=lambda in_features, out_features: get_flow_layer("ResidualNet")(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=5,
                    num_blocks=3,
                    use_batch_norm=True,
                    dropout_probability=0.0
                ),
                scale_and_bias=lambda in_features, out_features: get_flow_layer("PointwiseConvs")(
                    in_features=in_features,
                    out_features=out_features,
                    feats=self.conv_net_feats
                ),
                in_ch=self.internal_channels(),
                device='cuda'
            )
        elif name == "SAL":
            return get_flow_layer("StructureAwareConditionalLinearLayer")(
                meta_encoder=lambda in_features, out_features: get_flow_layer("ResidualNet")(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=5,
                    num_blocks=3,
                    use_batch_norm=True,
                    dropout_probability=0.0
                ),
                structure_encoder=lambda in_features, out_features: get_flow_layer("SpatialConvs")(
                    in_features=in_features,
                    out_features=out_features,
                    receptive_field=9,
                    feats=self.conv_net_feats
                ),
                in_ch=self.internal_channels(),
                device='cuda'
            )
        else:
            assert False, f"Invalid layer name : {name}"

    def forward(self, noisy, clean, kwargs=dict()):
        x = noisy
        kwargs['clean'] = clean.clone()

        objectives = 0.
        for bijector in self.pre_bijectors:
            if isinstance(bijector, get_flow_layer("UniformDequantization")):
                kwargs['clean'], _ = bijector._forward_and_log_det_jacobian(kwargs['clean'])

            x, ldj = bijector._forward_and_log_det_jacobian(x, **kwargs)
            objectives += ldj

        for bijector in self.bijectors:
            x, ldj = bijector._forward_and_log_det_jacobian(x, **kwargs)
            objectives += ldj
        return x, objectives

    def sample(self, kwargs=dict()):
        for bijector in self.pre_bijectors:
            if isinstance(bijector, get_flow_layer("UniformDequantization")):
                kwargs['clean'], _ = bijector._forward_and_log_det_jacobian(kwargs['clean'], **kwargs)

        b,_,h,w = kwargs['clean'].shape
        x = self.dist.sample((b,self.internal_channels(),h,w))
        for bijector in reversed(self.bijectors):
            x = bijector._inverse(x, **kwargs)

        for bijector in reversed(self.pre_bijectors):
            if isinstance(bijector, get_flow_layer("UniformDequantization")):
                kwargs['clean'] = bijector._inverse(kwargs['clean'], **kwargs)
            x = bijector._inverse(x, **kwargs)
        x = torch.clip(x, 0, 2**self.num_bits)
        return x 

class NMFlowDenoiser(nn.Module):
    def __init__(
            self,
            denoiser,
            kwargs_flow,
            flow_pth_path,
            num_bits=14,
        ):
        super().__init__()
        self.denoiser = denoiser
        self.kwargs_flow = kwargs_flow
        self.flow_pth_path = flow_pth_path
        self.num_bits = num_bits
        self.noise_model = get_model_class("NMFlow")(**kwargs_flow)
        self._load_checkpoint(self.noise_model, flow_pth_path)

    def _load_checkpoint(self, module, path, name='noise_model'):
        assert os.path.exists(path), f"{path} is not exist."
        pth = torch.load(path)
        module.load_state_dict(pth['model_weight'][name])
        module.eval()
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.denoiser.parameters(recurse) # the parameters of denoiser will be trained only.
        
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
        n_scaled = n_scaled * (2**num_bits) # n_scaled: 0 ~ denoiser's max GL.
        return n_scaled
    