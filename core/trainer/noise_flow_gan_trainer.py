import os
import datetime
from .base_trainer import BaseTrainer
from . import regist_trainer
from core.model import get_model_class
import torch
from util import AverageMeter, tensor2np, np2tensor, kl_div_3_data, get_file_name_from_path
import cv2
import numpy as np

@regist_trainer
class NoiseFlowGANTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.TRAIN.NOISE_GENERATOR_TRAINER:
            self.trainer_config = config.TRAIN.NOISE_GENERATOR_TRAINER

    def _set_module(self):
        kwargs_gen_flow, kwargs_gen_generator, kwargs_critic = None, None, None
        if self.config.BASE.model.lower() == 'nmflowgan':
            kwargs_gen_flow = self.config.MODEL.NMFLOW
            kwargs_gen_generator = self.config.MODEL.UNET
            kwargs_critic = self.config.MODEL.NMFLOWGAN_CRITIC
        else:
            assert False, "Invalid model name."

        module = {}
        if kwargs_gen_flow is None or kwargs_gen_generator is None: 
            module['generator'] = get_model_class(self.config.BASE.model + "generator")(dict(), dict())
        else: 
            module['generator'] = get_model_class(self.config.BASE.model+ "generator")(kwargs_gen_generator, kwargs_gen_flow)

        if kwargs_critic is None: 
            module['critic'] = get_model_class(self.config.BASE.model + "critic")(dict())
        else: 
            module['critic'] = get_model_class(self.config.BASE.model+ "critic")(**kwargs_critic)

        return module
    
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(opt        = self.config.OPTIMIZER, 
                                                     parameters = self.module[key].parameters(), 
                                                     lr         = float(self.config.TRAIN.init_lr))
        return optimizer
    
    def _forward_fn(self, module, loss, data, mode='generator'):
        assert 'clean' in data['dataset'] and 'noisy' in data['dataset']

        input_data = [data['dataset'][arg] for arg in self.config.MODEL.input_type] # noisy, clean, kwargs..
        z, ldj, noisy, px_noisy = module['generator'](*input_data)
        model_output = {'z':z, 'ldj':ldj, 'noisy':noisy, 'px_noisy': px_noisy} 

        model_output['fake'] = model_output['noisy']
        model_output['real'] = data['dataset']['noisy']
        model_output['critic_noise'] = self.trainer_config.critic_noise
        if model_output['critic_noise']:
            #REMARKS: If this part is changed, the logic for calculating the gradient penalty must also be changed.
            fake_noise = (model_output['fake']-data['dataset']['clean']).requires_grad_(True)
            real_noise = (model_output['real']-data['dataset']['clean']).requires_grad_(True)
            model_output['critic_fake'] = module['critic'](torch.cat([fake_noise, data['dataset']['clean']],dim=1).requires_grad_(True))
            model_output['critic_real'] = module['critic'](torch.cat([real_noise, data['dataset']['clean']],dim=1).requires_grad_(True))
        else:
            model_output['critic_fake'] = module['critic'](model_output['fake'])
            model_output['critic_real'] = module['critic'](model_output['real'])

        model_output['training_mode'] = mode
        
        losses, tmp_info = loss(input_data, model_output, data['dataset'], module, ratio=1.)
        return losses, tmp_info

    def _set_main_module(self):
        if len(self.device) > 1:
            module = self.model['generator'].module
        else:
            module = self.model['generator']
            
        if hasattr(module, 'sample'):
            self.sampler = module.sample
        else:
            self.sampler = module

    def test(self):
        dataset_load = (self.config.TEST.test_img is None) and (self.config.TEST.test_dir is None) 
        self._before_test(dataset_load)
        
        # set image save path
        for i in range(60):
            test_time = datetime.datetime.now().strftime('%m-%d-%H-%M') + '-%02d'%i
            img_save_path = 'img/test_%s_%03d_%s' % (self.config.DATA.test_dataset, self.epoch, test_time)
            if not self.file_manager.is_dir_exist(img_save_path): break

        kld = None
        if self.config.TEST.test_img is not None:
            self.test_img(self.config.TEST.test_img, self.config.TEST.save_dir)
            exit()
        elif self.config.TEST.test_dir is not None:
            self.test_dir(self.config.TEST.test_dir)
        else:
            kld = self._test_dataloader_process(    dataloader    = self.test_dataloader,
                                                    add_con       = 0.  if not 'add_con' in self.config.TEST else self.config.TEST.add_con,
                                                    floor         = False if not 'floor' in self.config.TEST else self.config.TEST.floor,   
                                                    scale         = 1. if not 'scale' in self.config.TEST else self.config.TEST.scale,
                                                    using_bits    = None if not 'using_bits' in self.config.TEST else self.config.TEST.using_bits,
                                                    img_save_path = img_save_path,
                                                    img_save      = self.config.TEST.save_image)
        if kld is not None:
            with open(os.path.join(self.file_manager.get_dir(img_save_path), 'kld-%.result'%(kld)), 'w') as f:
                f.write('KLD: %f'%(kld))

    def validation(self):
        # set denoiser
        self._set_main_module()

        # make directories for image saving
        img_save_path = 'img/val_%03d' % self.epoch
        self.file_manager.make_dir(img_save_path)
        
        kld = self._test_dataloader_process(    dataloader    = self.val_dataloader,
                                                add_con       = 0.  if not 'add_con' in self.config.VALIDATION else self.config.VALIDATION.add_con,
                                                floor         = False if not 'floor' in self.config.VALIDATION else self.config.VALIDATION.floor,   
                                                scale         = 1. if not 'scale' in self.config.VALIDATION else self.config.VALIDATION.scale,
                                                using_bits    = None if not 'using_bits' in self.config.VALIDATION else self.config.VALIDATION.using_bits,
                                                img_save_path = img_save_path,
                                                img_save      = self.config.VALIDATION.save_image)
        if kld is not None:
            with open(os.path.join(self.file_manager.get_dir(img_save_path), 'kld-%.3f.result'%(kld)), 'w') as f:
                f.write('KLD: %f'%(kld))

    def _test_dataloader_process(self, dataloader, add_con=0., floor=False, scale=1., using_bits=None, img_save=True, img_save_path=None, info=True):
        # make directory
        self.file_manager.make_dir(img_save_path)
        kld_avg = AverageMeter()
        for idx, data in enumerate(dataloader['dataset']):
            # to device
            kwargs = dict()
            if len(self.device) > 0:
                for key in data:
                    if isinstance(data[key], dict):
                        dictdata = data[key]
                        for k in dictdata:
                            if isinstance(dictdata[k], torch.Tensor):
                                kwargs[k] = dictdata[k] = dictdata[k].cuda()
                    else:
                        kwargs[key] = data[key] = data[key].cuda()
            
            #generated = self.module['noise_model'](*[data[arg] for arg in self.config.MODEL.input_type]).detach()
            fake = self.sampler(kwargs=kwargs).detach() 
            
            # evaluation
            real_noise=None
            if 'real_noisy' in data:
                real_noise = data['real_noisy']-data['clean']
                generated = fake - data['clean']
                # KLD
                quantization_bins=2**using_bits if using_bits is not None else 16
                kld = kl_div_3_data(
                    tensor2np(real_noise).flatten(), 
                    tensor2np(generated).flatten(),
                    None,
                    -quantization_bins-4, # add padding
                    quantization_bins+5 # add padding
                    )
                kld_avg.update(kld)

            # apply scale
            if scale is not None:
                for key, value in data.items():
                    data[key] = value * scale
                fake = fake * scale

            # add constant and floor (if floor is on)
            fake += add_con
            if floor: fake = torch.floor(fake)

            if using_bits==8:
                for key, value in data.items():
                    if key in ['clean', 'real_noisy', 'noisy', 'syn_noisy']:
                        data[key] = torch.clip(value,0,255).type(torch.uint8)
                fake = torch.clip(fake,0,255).type(torch.uint8)

            # image save
            if img_save:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze(0).cpu()
                if 'real_noisy' in data: noisy_img = data['real_noisy']
                elif 'syn_noisy' in data: noisy_img = data['syn_noisy']
                elif 'noisy' in data: noisy_img = data['noisy']
                else: noisy_img = None
                if noisy_img is not None: noisy_img = noisy_img.squeeze(0).cpu()
                fake_img = fake.squeeze(0).cpu()

                # write psnr value on file name
                gen_name = '%05d_GEN_%.6f'%(idx, kld) if 'real_noisy' in data else '%05d_GEN'%idx

                # imwrite
                if 'clean' in data:         self.file_manager.save_img_tensor(img_save_path, '%05d_CL'%idx, clean_img, ext=self.config.TEST.save_ext)
                if noisy_img is not None: self.file_manager.save_img_tensor(img_save_path, '%05d_N'%idx, noisy_img, ext=self.config.TEST.save_ext)
                self.file_manager.save_img_tensor(img_save_path, gen_name, fake_img, ext=self.config.TEST.save_ext)
                
            if info:
                if 'real_noisy' in data:
                    self.logger.note('[%s] testing... %05d/%05d. KLD : %.6f'%(self.status, idx, dataloader['dataset'].__len__(), kld), end='\r')
                else:
                    self.logger.note('[%s] testing... %05d/%05d.'%(self.status, idx, dataloader['dataset'].__len__()), end='\r')
                               
        # final log msg
        if kld_avg.count > 0:
            self.logger.val('[%s] Done! KLD: %.6f'%(self.status, kld_avg.avg))
        else:
            self.logger.val('[%s] Done!'%self.status)
            
        return kld_avg.avg
    
    def test_img(self, image_dir, save_dir='./'):
        '''
        Inference a single image.
        '''
        # load image
        if self.config.TEST.imread.lower() == 'gray' or self.config.TEST.imread.lower() == 'grey':
            clean = np2tensor(cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH).astype(np.float32))
        else:
            clean = np2tensor(cv2.imread(image_dir))
        clean = clean.unsqueeze(0).float()

        # to device
        if len(self.device) > 0:
            clean = clean.cuda()

        # forward
        kwargs = {'clean':clean}
        fake = self.sampler(kwargs) 

        # post-process
        fake += self.config.TEST.add_con
        if self.config.TEST.floor: fake = torch.floor(fake)
            
        # save image
        fake = tensor2np(fake)
        fake = fake.squeeze(0)
        if self.config.TEST.floor: fake = fake.astype(np.uint16)
        name = get_file_name_from_path(image_dir)
        cv2.imwrite(os.path.join(save_dir, name+'_GEN.'+self.config.TEST.save_ext), fake)

        # print message
        self.logger.note('[%s] saved : %s'%(self.status, os.path.join(save_dir, name+'_GEN.'+self.config.TEST.save_ext)))
      
    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = {}
        self._train_mode()
        for key in self.train_dataloader_iter:
            try:
                data[key] = next(self.train_dataloader_iter[key])
            except StopIteration:
                self.train_dataloader_iter[key] = iter(self.train_dataloader[key])
                data[key] = next(self.train_dataloader_iter[key])
                            
        # to device 
        if len(self.device) > 0 :
            for dataset_key in data:
                for key in data[dataset_key]:
                    if isinstance(data[dataset_key][key], dict):
                        dictdata = data[dataset_key][key]
                        for k in dictdata:
                            if isinstance(dictdata[k], torch.Tensor):
                                dictdata[k] = dictdata[k].cuda()        
                    else:
                        data[dataset_key][key] = data[dataset_key][key].cuda()

        # forward, cal losses, backward)
        # generator
        losses = dict()
        tmp_infos = dict()
        for mode in self.module:
            if mode == 'generator' and self.iter % self.config.TRAIN.NOISE_GENERATOR_TRAINER.generator_iter_step != 0:
                continue

            # zero grad
            self.optimizer[mode].zero_grad(set_to_none=True) 

            loss, tmp_info = self._forward_fn(self.model, self.loss, data, mode)
            losses.update({f'{mode}_'+key: loss[key].mean() for key in loss})
            tmp_infos.update({f'{mode}_'+key: tmp_info[key].mean() for key in tmp_info})

            # backward
            total_loss = sum(v for v in loss.values())
            total_loss.backward()

            # optimizer step
            self.optimizer[mode].step()

        # save losses and tmp_info
        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])
        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        self.loss_dict['count'] += 1
