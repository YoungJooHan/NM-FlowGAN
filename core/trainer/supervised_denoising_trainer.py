import os
import datetime
from .base_trainer import BaseTrainer
from . import regist_trainer
from core.model import get_model_class
import torch
from util import psnr, ssim, AverageMeter

@regist_trainer
class SLDenoisingTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
    def test(self):
        dataset_load = (self.config.TEST.test_img is None) and (self.config.TEST.test_dir is None) 
        self._before_test(dataset_load)
        
        # set image save path
        for i in range(60):
            test_time = datetime.datetime.now().strftime('%m-%d-%H-%M') + '-%02d'%i
            img_save_path = 'img/test_%s_%03d_%s' % (self.config.DATA.test_dataset, self.epoch, test_time)
            if not self.file_manager.is_dir_exist(img_save_path): break

        psnr = None
        ssim = None
        if self.config.TEST.test_img is not None:
            self.test_img(self.config.TEST.test_img, self.config.TEST.save_dir)
            exit()
        elif self.config.TEST.test_dir is not None:
            self.test_dir(self.config.TEST.test_dir)
        else:
            psnr, ssim = self._test_dataloader_process(     dataloader    = self.test_dataloader,
                                                            scale         = 1.  if not 'scale' in self.config.TEST else self.config.TEST.scale,
                                                            add_con       = 0.  if not 'add_con' in self.config.TEST else self.config.TEST.add_con,
                                                            floor         = False if not 'floor' in self.config.TEST else self.config.TEST.floor,   
                                                            using_bits    = None if not 'using_bits' in self.config.TEST else self.config.TEST.using_bits,
                                                            img_save_path = img_save_path,
                                                            img_save      = self.config.TEST.save_image)
        if psnr is not None and ssim is not None:
            with open(os.path.join(self.file_manager.get_dir(img_save_path), '_psnr-%.2f_ssim-%.3f.result'%(psnr, ssim)), 'w') as f:
                f.write('PSNR: %f\nSSIM: %f'%(psnr, ssim))
    
    def validation(self):
        # set denoiser
        self._set_main_module()

        # make directories for image saving
        img_save_path = 'img/val_%03d' % self.epoch
        self.file_manager.make_dir(img_save_path)
        
        psnr, ssim = self._test_dataloader_process(     dataloader    = self.val_dataloader,
                                                        scale         = 1.  if not 'scale' in self.config.TEST else self.config.TEST.scale,
                                                        add_con       = 0.  if not 'add_con' in self.config.VALIDATION else self.config.VALIDATION.add_con,
                                                        floor         = False if not 'floor' in self.config.VALIDATION else self.config.VALIDATION.floor,  
                                                        using_bits    = None if not 'using_bits' in self.config.VALIDATION else self.config.VALIDATION.using_bits, 
                                                        img_save_path = img_save_path,
                                                        img_save      = self.config.VALIDATION.save_image)
        if psnr is not None and ssim is not None:
            with open(os.path.join(self.file_manager.get_dir(img_save_path), '_psnr-%.2f_ssim-%.3f.result'%(psnr, ssim)), 'w') as f:
                f.write('PSNR: %f\nSSIM: %f'%(psnr, ssim))

    def _set_module(self):
        kwargs = None
        if self.config.BASE.model.lower() == 'dncnnflowgan':
            kwargs = self.config.MODEL.DNCNNFLOWGAN
            kwargs['kwargs_dncnn'] = self.config.MODEL.DNCNN
            kwargs['kwargs_flow'] = self.config.MODEL.NMFLOW
            kwargs['kwargs_unet'] = self.config.MODEL.UNET
        else:
            assert False, f"Invalid model: {self.config.BASE.model}"
        module = {}
        if kwargs is None:
            module['denoiser'] = get_model_class(self.config.BASE.model)()
        else:   
            module['denoiser'] = get_model_class(self.config.BASE.model)(**kwargs)
        return module
    
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(opt        = self.config.OPTIMIZER, 
                                                     parameters = self.module[key].parameters(), 
                                                     lr         = float(self.config.TRAIN.init_lr))
        return optimizer
    
    def _forward_fn(self, module, loss, data):
        input_data = [data['dataset'][arg] for arg in self.config.MODEL.input_type]
        denoised_img = module['denoiser'](*input_data)
        model_output = {'recon':denoised_img}
        if hasattr(module['denoiser'], 'num_bits'): model_output['num_bits'] = module['denoiser'].num_bits

        losses, tmp_info = loss(input_data, model_output, data['dataset'], module, \
                                    ratio=1.)
        
        return losses, tmp_info
    
    def _test_dataloader_process(self, dataloader, scale=1., add_con=0., floor=False, using_bits=None, img_save=True, img_save_path=None, info=True):
        # make directory
        self.file_manager.make_dir(img_save_path)
        psnr_avg = AverageMeter()
        ssim_avg = AverageMeter()
        for idx, data in enumerate(dataloader['dataset']):
            # to device
            if len(self.device) > 0:
                for key in data:
                    data[key] = data[key].cuda()
                    
            input_data = [data[arg] for arg in self.config.MODEL.test_input_type]
            if using_bits is not None: input_data.append({'num_bits':using_bits})
            denoised_image = self.denoiser(*input_data).detach()        

            # add constant and floor (if floor is on)
            if scale: denoised_image *= scale
            if add_con: denoised_image += add_con
            if floor: denoised_image = torch.floor(denoised_image)

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'], 
                                  max_val = 2**using_bits-1 if using_bits else 255)
                ssim_value = ssim(denoised_image, data['clean'], 
                                  data_range= 2**using_bits-1 if using_bits else 255)

                psnr_avg.update(psnr_value)
                ssim_avg.update(ssim_value)

            # image save
            if img_save:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze(0).cpu()
                if 'real_noisy' in self.config.MODEL.test_input_type: noisy_img = data['real_noisy']
                elif 'syn_noisy' in self.config.MODEL.test_input_type: noisy_img = data['syn_noisy']
                elif 'noisy' in self.config.MODEL.test_input_type: noisy_img = data['noisy']
                else: noisy_img = None
                if noisy_img is not None: noisy_img = noisy_img.squeeze(0).cpu()
                denoi_img = denoised_image.squeeze(0).cpu()

                # write psnr value on file name
                denoi_name = '%05d_DN_%.2f'%(idx, psnr_value) if 'clean' in data else '%05d_DN'%idx

                # imwrite
                if 'clean' in data:         self.file_manager.save_img_tensor(img_save_path, '%05d_CL'%idx, clean_img, ext=self.config.TEST.save_ext)
                if noisy_img is not None: self.file_manager.save_img_tensor(img_save_path, '%05d_N'%idx, noisy_img, ext=self.config.TEST.save_ext)
                self.file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img, ext=self.config.TEST.save_ext)
                
            if info:
                if 'clean' in data:
                    self.logger.note('[%s] testing... %05d/%05d. PSNR : %.2f dB'%(self.status, idx, dataloader['dataset'].__len__(), psnr_value), end='\r')
                else:
                    self.logger.note('[%s] testing... %05d/%05d.'%(self.status, idx, dataloader['dataset'].__len__()), end='\r')
                    
        # final log msg
        if psnr_avg.count > 0:
            self.logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f'%(self.status, psnr_avg.avg, ssim_avg.avg))
        else:
            self.logger.val('[%s] Done!'%self.status)
            
        return psnr_avg.avg, ssim_avg.avg

