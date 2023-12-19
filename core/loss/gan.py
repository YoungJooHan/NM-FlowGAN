from . import regist_loss
import torch
import torch.nn as nn
from util.standard_normal_dist import StandardNormal

@regist_loss
class GANLoss(nn.Module):
    def __init__(self, lambda_gp=10., lambda_gen=1.0):
        super(GANLoss, self).__init__()
        self.lambda_gen = lambda_gen
        self.lambda_gp = lambda_gp
        
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, input_data, model_output, data, module):
        training_mode = model_output['training_mode']
        if training_mode == 'generator':
            loss = -model_output['critic_fake'].mean() * self.lambda_gen
        elif training_mode == 'critic':
            if model_output['critic_noise']:
                #REMARKS: If this part is changed, _forward_fn must also be changed.
                fake_noise = (model_output['fake']-data['clean']).requires_grad_(True)
                real_noise = (model_output['real']-data['clean']).requires_grad_(True)
                gp_loss = self._gradient_penalty(
                    module['critic'],
                    torch.cat([real_noise, data['clean']],dim=1).requires_grad_(True),
                    torch.cat([fake_noise, data['clean']],dim=1).requires_grad_(True)
                )
            else:
                gp_loss = self._gradient_penalty(module['critic'], model_output['real'], model_output['fake'])
            loss = model_output['critic_fake'].mean() - model_output['critic_real'].mean() \
                + self.lambda_gp * gp_loss
        else:
            assert False, f'Invalid training mode: {training_mode}'

        return loss

    def _gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones([real_samples.shape[0], 1], device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
@regist_loss
class real_sub_fake(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data, model_output, data, module):
        return model_output['critic_real'].mean() - model_output['critic_fake'].mean()