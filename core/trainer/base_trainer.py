from util import FileManager, Logger, setup_determinism, rot_hflip_img, np2tensor, np2tensor_multi, tensor2np, get_file_name_from_path, psnr, ssim, make_predefiend_1d_to_2d, load_numpy_from_raw, save_img
import cv2
from core.loss import Loss
from core.dataset import get_dataset_class
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from tqdm import tqdm
from einops import rearrange

class BaseTrainer():
    '''
    Base trainer class to implement other trainer classes.
    below function should be implemented in each of trainer class.
    '''
    def test(self):
        raise NotImplementedError('define this function for each trainer')
    def validation(self):
        raise NotImplementedError('define this function for each trainer')
    def export(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')
    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')
    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    def __init__(self, config):
        self.session_name = config.LOG.session_name

        self.checkpoint_folder = 'checkpoint'

        # get file manager and logger class
        self.file_manager = FileManager(self.session_name, config.DATA.output_path)
        self.logger = Logger()        
        self.config = config
        self.device = None
        
    def _log_configs(self, config, prefix=""):
        for key, value in config.items():
            if isinstance(value, dict):
                self._log_configs(value, prefix+"."+key)
            else:
                msg = '{}.{}: {}'.format(prefix, key, value)
                self.logger.highlight(msg)

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.config.TRAIN.warmup:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch+1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()
        
        self._after_train()
        
    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()
                    
    def _before_epoch(self):
        self._set_status('epoch %04d/%04d'%(self.epoch, self.max_epoch))

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        self._train_mode()
        
    def _run_epoch(self):
        for self.iter in range(1, self.max_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # save checkpoint
        if self.epoch >= self.config.CKPT.start_epoch:
            if (self.epoch-self.config.CKPT.start_epoch)%self.config.CKPT.interval_epoch == 0:
                self.save_checkpoint()

        # validation
        if self.config.DATA.validation_dataset_path is not None:
            if self.epoch >= self.config.VALIDATION.start_epoch:
                if (self.epoch-self.config.VALIDATION.start_epoch) % self.config.VALIDATION.interval_epoch == 0:
                    self._eval_mode()
                    self._set_status('val %03d'%self.epoch)
                    self.validation()
                    
    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())
                    
    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        if len(self.device) > 1:
            torch.save({'epoch': self.epoch,
                        'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                        'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                        os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))
        else:
            torch.save({'epoch': self.epoch,
                        'model_weight': {key:self.model[key].state_dict() for key in self.model},
                        'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                        os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))
            
    def load_checkpoint(self, load_epoch=0, name=None):
        if name is None:
            # if scratch, return
            if load_epoch == 0: return
            # load from local checkpoint folder
            file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        else:
            # load from global checkpoint folder
            file_name = os.path.join('./ckpt', name)
        
        # check file exist
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

        # print message 
        self.logger.note('[%s] model loaded : %s'%(self.status, file_name))

    def _checkpoint_name(self, epoch, extension='pth'):
        return self.session_name + '_%03d'%epoch + f'.{extension}'
        
    def set_device(self, device):
        assert isinstance(device, str)
        self.device = [int(i) for i in device.split(',')] 

    def _set_loss(self):
        self.loss = Loss(self.config.TRAIN.loss, self.config.TRAIN.tmp_info)
    
    def _before_train(self):
        # setup determinism
        if self.config.BASE.seed > 0:
            setup_determinism(self.config.BASE.seed)       

        # initialing
        self.module = self._set_module()

        # training dataset loader
        self.logger.info('Prepare training dataloader...')
        self.train_dataloader = self._set_dataloader(
            self.config.DATA.train_dataset, 
            self.config.DATA.train_dataset_path, 
            self.config.DATA.TRAIN_DATALOADER, 
            batch_size=self.config.DATA.batch_size, 
            shuffle=True, 
            num_workers=self.config.DATA.threads
            )
        self.logger.info('Done!')

        # validation dataset loader
        self.logger.info('Prepare validation dataloader...')
        self.val_dataloader = self._set_dataloader(
            self.config.DATA.validation_dataset, 
            self.config.DATA.validation_dataset_path, 
            self.config.DATA.VALIDATION_DATALOADER, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.config.DATA.threads
            )
        self.logger.info('Done!')

        # other configuration
        self.max_epoch = self.config.TRAIN.max_epochs
        self.epoch = self.start_epoch = 1
        max_len = self.train_dataloader['dataset'].dataset.__len__() # base number of iteration works for dataset named 'dataset'
        self.max_iter = math.ceil(max_len / self.config.DATA.batch_size)

        self._set_loss()
        self.loss_dict = {'count':0}
        self.tmp_info = {}
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        if self.config.BASE.resume:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # wrapping and device setting
        assert len(self.device) > 0, "There is not available device."
        if len(self.device) > 1:
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key], self.device).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
            os.environ["CUDA_VISIBLE_DEVICES"]= str(self.device[0])
            self.model = {key: self.module[key].cuda() for key in self.module}

        # start message
        #self.logger.info(self.summary())
        self.logger.start((self.epoch-1, 0))
        self.logger.highlight(self.logger.get_start_msg())
        self._log_configs(self.config)

    def _warmup(self):
        self._set_status('warmup')
        self._train_mode()
        
        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.config.TRAIN.warmup_iters
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter+1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()
            
    def _before_step(self):
        pass
    
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
        losses, tmp_info = self._forward_fn(self.model, self.loss, data)
        losses   = {key: losses[key].mean()   for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        for opt in self.optimizer.values():
            opt.step()

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True) 

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

    def _after_step(self):
        # adjust learning rate
        self._adjust_lr()

        # print loss
        if (self.iter%self.config.LOG.interval_iter==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch-1, self.iter-1))
        
    def _adjust_lr(self):
        sched = self.config.SCHEDULER.type

        if sched == 'step':
            '''
            step decreasing scheduler
            Args:
                step_size: step size(epoch) to decay the learning rate
                gamma: decay rate
            '''
            if self.iter == self.max_iter:
                args = self.config.SCHEDULER.STEP
                if self.epoch % args.step_size == 0:
                    for optimizer in self.optimizer.values():
                        lr_before = optimizer.param_groups[0]['lr']
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_before * float(args.gamma)
        elif sched == 'linear':
            '''
            linear decreasing scheduler
            Args:
                step_size: step size(epoch) to decrease the learning rate
                gamma: decay rate for reset learning rate
            '''
            args = self.config.SCHEDULER.LINEAR
            if not hasattr(self, 'reset_lr'):
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args.gamma)**((self.epoch-1)//args.step_size)

            # reset lr to initial value
            if self.epoch % args.step_size == 0 and self.iter == self.max_iter:
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args.gamma)**(self.epoch//args.step_size)
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.reset_lr
            # linear decaying
            else:
                ratio = ((self.epoch + (self.iter)/self.max_iter - 1) % args['step_size']) / args['step_size']
                curr_lr = (1-ratio) * self.reset_lr
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched))
    
    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']
            
    def print_loss(self):
        temporal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                    temporal_loss += self.loss_dict[key]/self.loss_dict['count']
        self.loss_log += [temporal_loss]
        if len(self.loss_log) > 100: self.loss_log.pop(0)

        # print status and learning rate
        loss_out_str = '[%s] %04d/%04d, lr:%s ∣ '%(self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))
        global_iter = (self.epoch-1)*self.max_iter + self.iter

        # print losses
        avg_loss = np.mean(self.loss_log)
        loss_out_str += 'avg_100 : %.6f ∣ '%(avg_loss)

        for key in self.loss_dict:
            if key != 'count':
                loss = self.loss_dict[key]/self.loss_dict['count']
                loss_out_str += '%s : %.6f ∣ '%(key, loss)
                self.loss_dict[key] = 0.

        # print temporal information
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            for key in self.tmp_info:
                loss_out_str += '  %s : %.2f'%(key, self.tmp_info[key]/self.loss_dict['count'])
                self.tmp_info[key] = 0.
            loss_out_str += ' ]'

        # reset
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)
    
    def _set_status(self, status:str):
        status_len = 15
        assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)'%(status_len, len(status))

        if len(status.split(' ')) == 2:
            s0, s1 = status.split(' ')
            self.status = '%s'%s0.rjust(status_len//2) + ' '\
                          '%s'%s1.ljust(status_len//2)
        else:
            sp = status_len - len(status)
            self.status = ''.ljust(sp//2) + status + ''.ljust((sp+1)//2)
            
    def _adjust_warmup_lr(self, warmup_iter):
        init_lr = float(self.config.TRAIN.init_lr)
        warmup_lr = init_lr * self.iter / warmup_iter

        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
                
    def _set_dataloader(self, dataset_class, path, dataset_args, batch_size, shuffle, num_workers):
        dataloader = {}
        
        dataset_args['path'] = path
        dataset = get_dataset_class(dataset_class)(**dataset_args)
        dataloader['dataset'] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
        return dataloader
    
    def _set_one_optimizer(self, opt, parameters, lr):
        if opt.type == 'SGD':
            return torch.optim.SGD(parameters, lr=lr, momentum=float(opt.SGD.momentum), weight_decay=float(opt.SGD.weight_decay))
        elif opt.type == 'Adam':
            return torch.optim.Adam(parameters, lr=lr, betas=opt.ADAM.betas)
        elif opt.type == 'AdamW':
            return torch.optim.Adam(parameters, lr=lr, betas=opt.ADAMW.betas, weight_decay=float(opt.ADAMW.weight_decay))
        else:
            raise RuntimeError()
    
    def _set_main_module(self):
        if len(self.device) > 1:
            module = self.model['denoiser'].module
        else:
            module = self.model['denoiser']
            
        if hasattr(module, 'denoise'):
            self.denoiser = module.denoise
        else:
            self.denoiser = module
            
    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        if len(epochs) <= 0: return 0
        return max(epochs)
            
    def _before_test(self, dataset_load):
        # initialing
        self.module = self._set_module()
        self._set_status('test')

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.config.TEST.ckpt_epoch == -1 else self.config.TEST.ckpt_epoch
        ckpt_name  = self.config.BASE.pretrained if self.config.BASE.pretrained is not None else None
        self.load_checkpoint(ckpt_epoch, ckpt_name)
        self.epoch = ckpt_epoch # for print or saving file name.

        # test dataset loader
        if dataset_load:
            self.logger.info('Prepare test dataloader...')
            self.test_dataloader = self._set_dataloader(
                self.config.DATA.test_dataset, 
                self.config.DATA.test_dataset_path, 
                self.config.DATA.TEST_DATALOADER, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.config.DATA.threads
                )
            self.logger.info('Done!')


        # wrapping and device setting
        assert len(self.device) > 0, "There is not available device."
        if len(self.device) > 1:
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key], self.device).cuda() for key in self.module}
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
            os.environ["CUDA_VISIBLE_DEVICES"]= str(self.device[0])
            self.model = {key: self.module[key].cuda() for key in self.module}

        # evaluation mode and set status
        self._eval_mode()
        self._set_status('test %04d'%self.epoch)

        # start message
        self.logger.highlight(self.logger.get_start_msg())
        self._log_configs(self.config)
        
        # set denoiser
        self._set_main_module()

        # wrapping denoiser w/ crop test
        if self.config.TEST.crop:
            crop_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(crop_fn, 
                                                               *input_data, 
                                                               size=self.config.TEST.CROP.size, 
                                                               overlap=self.config.TEST.CROP.overlap
                                                               )

    def _get_determine_patch_indices(self, idx, idy, crop_w, crop_h, img_w, img_h):
        # get horizontal info.
        if idx + crop_w >= img_w:
            start_x = img_w-crop_w
            end_x = img_w
        else:
            start_x=idx
            end_x = idx+crop_w
        # get vertical info.
        if idy + crop_h >= img_h:
            start_y = img_h-crop_h
            end_y = img_h
        else:
            start_y=idy
            end_y = idy+crop_h
        return start_x, end_x, start_y, end_y

    def _generate_image_patches(self, img, roi, c_w, c_h, s_x, s_y):
        '''
        img : original image (tensor)
        roi : ROI of original image [top, left, height, width]
        c_w : width of the cropped images
        c_h : height of the cropped images
        s_x : stride (x-axis)
        s_y : stride (y-axis)
        '''
        roi_t, roi_l, roi_h, roi_w = roi
        _, _, img_h, img_w = img.shape

        assert img_h >= roi_t + roi_h and img_w >= roi_l + roi_w

        img_roi = img[:,:,roi_t:roi_t+roi_h, roi_l:roi_l+roi_w]
        idx, idy = 0, 0
        results = list()
        while idy + c_h <= roi_h:
            while idx + c_w <= roi_w:
                start_x, end_x, start_y, end_y = self._get_determine_patch_indices(idx, idy, c_w, c_h, roi_w, roi_h)
                cropped = img_roi[:,:,start_y:end_y, start_x:end_x] # img_roi : C, H, W
                results.append(cropped)
                idx += s_x
            idx = 0
            idy += s_y
        return results

    def _generate_weight_kernel(self, c_w, c_h, stride):
        kernel = np.zeros((c_h, c_w), dtype=np.float32)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                val = float(min(min(i, kernel.shape[0] - 1 - i), min(j, kernel.shape[1] - 1 - j)))
                kernel[i, j] = val
        kernel += 1.0
        kernel = np.minimum(kernel, float(stride))
        kernel /= float(stride)
        return torch.tensor(kernel)
    
    def _generate_weights_matrix(self, shape, kernel, c_w, c_h, s_x, s_y, device):
        _, _, roi_h, roi_w = shape
        weights = torch.zeros(shape, device=device, dtype=torch.float32)
        idx, idy = 0, 0
        while idy + s_y <= roi_h:
            while idx + s_x <= roi_w:
                i, j = idy, idx
                if j + c_w > roi_w: j = roi_w - c_w
                if i + c_h > roi_h: i = roi_h - c_h
                weights[:, :, i:i+c_h, j:j+c_w] += kernel
                idx += s_x
            idx = 0
            idy += s_y
        return weights
                            
    def _crop_operator(self, img, fn, roi, c_w, c_h, s_x, s_y):
        _, _, roi_h, roi_w = roi
        b, _, _, _ = img.shape
        weights = None
        reconst = None

        # generate weights kernel
        kernel = self._generate_weight_kernel(c_w, c_h, s_x).to(img.device)

        idx, idy = 0, 0
        while idy + s_y <= roi_h:
            while idx + s_x <= roi_w:
                i, j = idy, idx
                if j + c_w > roi_w: j = roi_w - c_w
                if i + c_h > roi_h: i = roi_h - c_h

                noisy = img[:,:,i:i+c_h, j:j+c_w]
                denoised = fn(noisy)

                if weights is None: weights = self._generate_weights_matrix([denoised.shape[0], denoised.shape[1], roi_h, roi_w], kernel, c_w, c_h, s_x, s_y, img.device)
                if reconst is None: reconst = torch.zeros([denoised.shape[0], denoised.shape[1], roi_h, roi_w], device = img.device, dtype=torch.float32)

                reconst[:,:,i:i+c_h, j:j+c_w] += denoised * (kernel / weights[:,:,i:i+c_h, j:j+c_w])
                idx += s_x
            idx = 0 
            idy += s_y
        return reconst
    
    @torch.no_grad()
    def crop_test(self, fn, x, kwargs=None, size=512, overlap=0):
        _,_,h,w = x.shape
        assert size > overlap, "Invalid parameter. (size <= overlap)"
        reconst = self._crop_operator(x, fn, [0,0,h,w], size, size, size-overlap, size-overlap)
        return reconst
    
    def test_img(self, image_dir, save_dir='./'):
        '''
        Inference a single image.
        '''      
        # load image (noisy)
        noisy = None
        if image_dir[-4:] == '.raw':
            noisy = np2tensor(make_predefiend_1d_to_2d(load_numpy_from_raw(image_dir, 'uint16')).astype(np.float32)).unsqueeze(0)
        elif self.config.TEST.imread.lower() == 'gray' or self.config.TEST.imread.lower() == 'grey':
            ret, noisy = cv2.imreadmulti(image_dir, flags=cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH) # it can be multi-stacked image.
            assert ret > 0, f"Failed to load image: {image_dir}"
            noisy = np2tensor_multi(noisy)
        else:
            noisy = np2tensor(cv2.imread(image_dir)).unsqueeze(0)

        # load image (clean, optional)
        clean = None
        if image_dir.find("_N.") > 0: # if not exist, -1 is returned, 
            clean_path = image_dir.replace("_N.", "_CL.")
            clean_path = clean_path if os.path.exists(clean_path) else None
            if clean_path[-4:] == '.raw':
                clean = np2tensor(make_predefiend_1d_to_2d(load_numpy_from_raw(clean_path, 'uint16')).astype(np.float32)).unsqueeze(0)
            elif self.config.TEST.imread.lower() == 'gray' or self.config.TEST.imread.lower() == 'grey':
                ret, clean = cv2.imreadmulti(clean_path, flags=cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH) # it can be multi-stacked image.
                assert ret > 0, f"Failed to load image: {clean_path}"
                clean = np2tensor_multi(clean)
            else:
                clean = np2tensor(cv2.imread(clean_path)).unsqueeze(0)

        if len(self.device) > 0:
            noisy = noisy.cuda()

        # multi-frame input
        if self.config.TEST.TEST_DIR.no_input_frames is None: no_input_frames = 1
        else: no_input_frames = self.config.TEST.TEST_DIR.no_input_frames

        denoised = None
        if noisy.shape[0] > 1 : pbar = tqdm(range(noisy.shape[0] - no_input_frames + 1))
        else : pbar = range(noisy.shape[0] - no_input_frames + 1)
        for batch_idx in pbar:
            noisy_batch = noisy[batch_idx:batch_idx+no_input_frames, ...]
            noisy_batch = rearrange(noisy_batch, 'b c h w -> 1 (b c) h w') # batch size is always 1. number of multi-frame denotes number of channels.

            torch.cuda.synchronize()
            tic = time.time()

            # denoising
            if denoised is None : denoised = self.denoiser(noisy_batch, {'num_bits':self.config.TEST.using_bits}).cpu()
            else: denoised = torch.cat([denoised, self.denoiser(noisy_batch, {'num_bits':self.config.TEST.using_bits}).cpu()], dim=0)

            torch.cuda.synchronize()
            toc = time.time()
            time_span_ms = (toc-tic)*1000.
            if isinstance(pbar, tqdm):
                pbar.set_description('Time span (ms): %.2f'%(time_span_ms))
            else:
                self.logger.note('Time span (ms): %.2f'%(time_span_ms))

        # calculate PSNR and SSIM, if possible
        PSNR, SSIM = None, None
        if clean: # if not exist, -1 is returned, 
            #max_val = torch.max(denoised).item() if torch.max(denoised) > torch.max(clean) else torch.max(clean).item()
            PSNR = psnr(denoised, clean, max_val=2**self.config.TEST.using_bits-1 if self.config.TEST.using_bits else 255),  
            SSIM = ssim(denoised, clean, data_range=2**self.config.TEST.using_bits-1 if self.config.TEST.using_bits else 255)
            
        # post-process
        denoised *= self.config.TEST.scale
        denoised += self.config.TEST.add_con
        if self.config.TEST.floor: denoised = torch.floor(denoised)

        # save image
        denoised = tensor2np(denoised)
        #denoised = denoised.squeeze(0)
        if self.config.TEST.floor: denoised = np.clip(denoised, a_min=0, a_max=2**16-1).astype(np.uint16)
        name = get_file_name_from_path(image_dir)
        if PSNR and SSIM:
            save_img(save_dir, name.replace('_N','_DN')+'_psnr_%.3f_ssim_%.5f'%(PSNR, SSIM)+'.'+self.config.TEST.save_ext, denoised)
        else:
            save_img(save_dir, name + '_DN' + '.'+self.config.TEST.save_ext, denoised)
            if self.config.TEST.TEST_DIR.save_original_img:
                noisy_tmp = tensor2np(noisy[-denoised.shape[0]:, ...].cpu().squeeze(0))*self.config.TEST.scale
                if self.config.TEST.floor: noisy_tmp = np.floor(noisy_tmp+self.config.TEST.add_con).astype(np.uint16)
                save_img(save_dir, name + '_N' + '.'+self.config.TEST.save_ext, noisy_tmp)

        # print message
        if PSNR and SSIM:
            self.logger.note('[%s] saved : %s (psnr: %.3f, ssim: %.5f'% \
                             (self.status, os.path.join(save_dir, name.replace('_N','_DN')+'.'+ self.config.TEST.save_ext), PSNR, SSIM))
        else:
            self.logger.note('[%s] saved : %s'%(self.status, os.path.join(save_dir, name+'_DN.' + self.config.TEST.save_ext)))

    def test_dir(self, direc):
        '''
        Inference all images in the directory.
        '''
        for ff in [f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]:
            if "_CL." in ff: continue
            if ff[-4:] != '.tif' and ff[-5:] != '.tiff' and ff[-4:] != '.raw' and ff[-4:] != '.png' and ff[-4:] != '.jpg': continue
            
            result_dir_name = f'results_{self.session_name}_epoch{self.epoch}'
            if len(self.config.TEST.TEST_DIR.postfix) > 0:
                result_dir_name += f'_{self.config.TEST.TEST_DIR.postfix}' 

            os.makedirs(os.path.join(direc, result_dir_name), exist_ok=True)
            self.test_img(os.path.join(direc, ff), os.path.join(direc, result_dir_name))