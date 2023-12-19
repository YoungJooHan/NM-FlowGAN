import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from . import regist_layer

@regist_layer
class ConditionalLinear(nn.Module):
    def __init__(self, device='cpu', name='linear_transformation'):
        super(ConditionalLinear, self).__init__()
        self.name = name

        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

        self.log_scale = nn.Parameter(torch.zeros(25), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(25), requires_grad=True)

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]

        x = (z - bias.reshape((-1, 1, 1, 1))) / torch.exp(log_scale.reshape((-1, 1, 1, 1)))
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        
        z = x * torch.exp(log_scale.reshape((-1, 1, 1, 1))) + bias.reshape((-1, 1, 1, 1))
        log_abs_det_J_inv = log_scale * np.prod(x.shape[1:])

        return z, log_abs_det_J_inv

@regist_layer
class ConditionalLinearExp2(nn.Module):
    def __init__(self, in_ch=3, device='cpu', name='linear_transformation_exp2'):
        super(ConditionalLinearExp2, self).__init__()
        self.name = name
        self.device = device 

        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

        self.log_scale = nn.Parameter(torch.zeros(25, in_ch), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(25, in_ch), requires_grad=True)

    def _inverse(self, z, **kwargs):
        b,_,_,_ = z.shape

        iso = torch.zeros([b], device=self.device, dtype=torch.float32)
        for iso_idx, iso_val in enumerate(self.iso_vals):
            iso += torch.where(kwargs['ISO-level'] == iso_val, iso_idx, 0.0)

        cam = torch.zeros([b], device=self.device, dtype=torch.float32)
        for cam_idx, cam_val in enumerate(self.cam_vals):
            cam += torch.where(kwargs['smartphone-code'] == cam_val, cam_idx, 0.0)

        iso_cam = iso * self.iso_vals.shape[0] + cam
        iso_cam = torch.arange(0, self.iso_vals.shape[0] * self.cam_vals.shape[0]).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]

        x = (z - bias.reshape((-1, z.shape[1], 1, 1))) / torch.exp(log_scale.reshape((-1, z.shape[1], 1, 1)))
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        b,_,_,_ = x.shape

        iso = torch.zeros([b], device=x.device, dtype=torch.float32)
        for iso_idx, iso_val in enumerate(self.iso_vals):
            iso += torch.where(kwargs['ISO-level'] == iso_val, iso_idx, 0.0)

        cam = torch.zeros([b], device=x.device, dtype=torch.float32)
        for cam_idx, cam_val in enumerate(self.cam_vals):
            cam += torch.where(kwargs['smartphone-code'] == cam_val, cam_idx, 0.0)

        iso_cam = iso * self.iso_vals.shape[0] + cam
        iso_cam = torch.arange(0, self.iso_vals.shape[0] * self.cam_vals.shape[0]).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        z = x * torch.exp(log_scale.reshape((-1, x.shape[1], 1, 1))) + bias.reshape((-1, x.shape[1], 1, 1))
        log_abs_det_J_inv = torch.sum(log_scale * np.prod(x.shape[2:]), dim=1)

        return z, log_abs_det_J_inv
    

@regist_layer
class SignalDependentConditionalLinear(nn.Module):
    def __init__(self, meta_encoder, scale_and_bias, in_ch=3, device='cpu', name='signal_dependent_condition_linear'):
        super(SignalDependentConditionalLinear, self).__init__()
        self.name = name
        self.device = device 

        self.in_ch = in_ch
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        self.encode_ch = 3
        self.meta_encoder = meta_encoder(10, self.encode_ch)
        self.scale_and_bias = scale_and_bias(self.encode_ch+in_ch, in_ch*2) # scale, bias per channels

    def _get_embeddings(self, x, **kwargs):
        b,_,_,_ = x.shape

        iso = torch.zeros([b], device=x.device, dtype=torch.float32)
        for iso_idx, iso_val in enumerate(self.iso_vals):
            iso += torch.where(kwargs['ISO-level'] == iso_val, iso_idx, 0.0)

        cam = torch.zeros([b], device=x.device, dtype=torch.float32)
        for cam_idx, cam_val in enumerate(self.cam_vals):
            cam += torch.where(kwargs['smartphone-code'] == cam_val, cam_idx, 0.0)

        iso_one_hot = F.one_hot(iso.to(torch.int64), num_classes=self.iso_vals.shape[0]).to(torch.float32)
        cam_one_hot = F.one_hot(cam.to(torch.int64), num_classes=self.cam_vals.shape[0]).to(torch.float32)

        embedding = self.meta_encoder(torch.cat((iso_one_hot, cam_one_hot), dim=1)) # [b, 10] -> [b,encode_ch]
        embedding = embedding.reshape((-1, self.encode_ch, 1, 1))
        embedding = torch.repeat_interleave(embedding, x.shape[-2], dim=-2)# [b, encode_ch, 1, 1] -> [b, encode_ch, h, 1]
        embedding = torch.repeat_interleave(embedding, x.shape[-1], dim=-1)# [b, encode_ch, h, 1] -> [b, encode_ch, h, w]

        embedding = torch.cat((embedding, kwargs['clean']), dim=1) # [b, encode_ch, h, w], [b, c, h, w] -> [b, c+encode_ch, h, w]

        embedding = self.scale_and_bias(embedding)
        return embedding
    
    def _inverse(self, z, **kwargs):
        embedding = self._get_embeddings(z, **kwargs)

        log_scale = embedding[:,:self.in_ch, ...]
        bias = embedding[:,self.in_ch:, ...]
        z = (z - bias)/torch.exp(log_scale)
        return z

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        embedding = self._get_embeddings(x, **kwargs)

        log_scale = embedding[:,:self.in_ch, ...]
        bias = embedding[:,self.in_ch:, ...]
        
        z = torch.exp(log_scale)*x + bias
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1,2,3])
        return z, log_abs_det_J_inv

@regist_layer
class StructureAwareConditionalLinearLayer(nn.Module):
    def __init__(self, meta_encoder, structure_encoder, in_ch=3, device='cpu', name='signal_dependent_condition_linear'):
        super(StructureAwareConditionalLinearLayer, self).__init__()
        self.in_ch = in_ch
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

        self.meta_encoder = meta_encoder(10, in_ch*2)
        self.structure_encoder = structure_encoder(in_ch, in_ch*2)

    def _get_embeddings(self, x, **kwargs):
        b,_,_,_ = x.shape

        iso = torch.zeros([b], device=x.device, dtype=torch.float32)
        for iso_idx, iso_val in enumerate(self.iso_vals):
            iso += torch.where(kwargs['ISO-level'] == iso_val, iso_idx, 0.0)

        cam = torch.zeros([b], device=x.device, dtype=torch.float32)
        for cam_idx, cam_val in enumerate(self.cam_vals):
            cam += torch.where(kwargs['smartphone-code'] == cam_val, cam_idx, 0.0)

        iso_one_hot = F.one_hot(iso.to(torch.int64), num_classes=self.iso_vals.shape[0]).to(torch.float32)
        cam_one_hot = F.one_hot(cam.to(torch.int64), num_classes=self.cam_vals.shape[0]).to(torch.float32)

        meta_embedding = self.meta_encoder(torch.cat((iso_one_hot, cam_one_hot), dim=1)) # [b, 10] -> [b,encode_ch]
        meta_embedding = meta_embedding.reshape((-1, self.in_ch*2, 1, 1))
        
        structure_embedding = self.structure_encoder(kwargs['clean'])
        embedding = structure_embedding * meta_embedding
        return embedding
    
    def _inverse(self, z, **kwargs):
        embedding = self._get_embeddings(z, **kwargs)

        log_scale = embedding[:,:self.in_ch, ...]
        bias = embedding[:,self.in_ch:, ...]
        z = (z - bias)/torch.exp(log_scale)
        return z

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        embedding = self._get_embeddings(x, **kwargs)

        log_scale = embedding[:,:self.in_ch, ...]
        bias = embedding[:,self.in_ch:, ...]
        
        z = torch.exp(log_scale)*x + bias
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1,2,3])
        return z, log_abs_det_J_inv


@regist_layer
class PointwiseConvs(nn.Module):
    def __init__(self, in_features=3, out_features=3, feats=32, device='cpu', name='pointwise_convs'):
        super(PointwiseConvs, self).__init__()
        self.name = name
        self.device = device 
        self.body = nn.Sequential(
            nn.Conv2d(in_features, feats, kernel_size=1, stride=1, padding=0),
            self._get_basic_module(feats, feats*2, k_size=1, stride=1, padding=0),
            self._get_basic_module(feats*2, feats*2, k_size=1, stride=1, padding=0),
            self._get_basic_module(feats*2, feats, k_size=1, stride=1, padding=0),
            nn.Conv2d(feats, out_features, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _get_basic_module(self, in_ch, out_ch, k_size=1, stride=1, padding=1, negative_slope=0.2):
            return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padding),
                    nn.InstanceNorm2d(out_ch, affine=True), #batch normalization?
                    nn.LeakyReLU(negative_slope, inplace=True)
            )
    
    def forward(self, x):
        return self.body(x)
    
@regist_layer
class SpatialConvs(nn.Module):
    def __init__(self, in_features=3, out_features=3, feats=32, receptive_field=9, device='cpu', name='pointwise_convs'):
        super(SpatialConvs, self).__init__()
        self.name = name
        self.device = device 

        self.receptive_field = receptive_field

        self.body = list()
        self.body.append(nn.Conv2d(in_features, feats, kernel_size=1, stride=1, padding=0))
        self.body.append(nn.ReLU(inplace=True))

        for _ in range(self.receptive_field//2):
            self.body.append(nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1))
            self.body.append(nn.ReLU(inplace=True))
        
        self.body.append(nn.Conv2d(feats, out_features, kernel_size=1, stride=1, padding=0))
        self.body.append(nn.Tanh())
        self.body = nn.Sequential(*self.body)

    def _get_basic_module(self, in_ch, out_ch, k_size=1, stride=1, padding=1, negative_slope=0.2):
            return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padding),
                    nn.InstanceNorm2d(out_ch, affine=True), #batch normalization?
                    nn.LeakyReLU(negative_slope, inplace=True)
            )
    
    def forward(self, x):
        return self.body(x)