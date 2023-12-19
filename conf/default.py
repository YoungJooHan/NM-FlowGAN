from yacs.config import CfgNode as CN

_C = CN()

_C.BASE = CN()
_C.BASE.model = None
_C.BASE.trainer = None
_C.BASE.seed = 2016309
_C.BASE.resume = False
_C.BASE.pretrained = None

_C.MODEL = CN()
_C.MODEL.input_type = ['noisy']
_C.MODEL.test_input_type = ['noisy']

_C.MODEL.DNCNN = CN()
_C.MODEL.DNCNN.channels = 1
_C.MODEL.DNCNN.num_of_layers = 18
_C.MODEL.DNCNN.features = 64

_C.MODEL.UNET = CN()
_C.MODEL.UNET.in_channels = 1
_C.MODEL.UNET.n_classes = 1
_C.MODEL.UNET.depth = 5
_C.MODEL.UNET.wf = 6
_C.MODEL.UNET.padding = True
_C.MODEL.UNET.batch_norm = False
_C.MODEL.UNET.up_mode = 'upconv'
_C.MODEL.UNET.residual = True
_C.MODEL.UNET.drop_p = 0.15

_C.MODEL.NMFLOWGAN_CRITIC = CN()
_C.MODEL.NMFLOWGAN_CRITIC.in_ch = 3
_C.MODEL.NMFLOWGAN_CRITIC.nc = 64
_C.MODEL.NMFLOWGAN_CRITIC.num_bits = 8

_C.MODEL.NMFLOW = CN()
_C.MODEL.NMFLOW.in_ch = 1
_C.MODEL.NMFLOW.num_bits = 14
_C.MODEL.NMFLOW.pre_arch = "UD"
_C.MODEL.NMFLOW.arch = "NE|SAL|SDL|CL2|SAL|SDL|CL2" 
_C.MODEL.NMFLOW.ch_exp_coef = 1
_C.MODEL.NMFLOW.width_exp_coef = 1.334
_C.MODEL.NMFLOW.conv_net_feats = 16

_C.MODEL.DNCNNFLOWGAN = CN()
_C.MODEL.DNCNNFLOWGAN.pretrained_path = ""
_C.MODEL.DNCNNFLOWGAN.num_bits = 0

_C.LOG = CN()
_C.LOG.session_name = None
_C.LOG.interval_iter = 10

_C.DATA = CN()
_C.DATA.train_dataset = "SIDD_HDF"
_C.DATA.validation_dataset = "SIDD_HDF"
_C.DATA.test_dataset = "SIDD_HDF"
_C.DATA.train_dataset_path = None
_C.DATA.validation_dataset_path = None
_C.DATA.test_dataset_path = None
_C.DATA.output_path = None
_C.DATA.threads = 0
_C.DATA.batch_size = 8

_C.DATA.TRAIN_DATALOADER = CN()
_C.DATA.TRAIN_DATALOADER.add_noise = None
_C.DATA.TRAIN_DATALOADER.crop_size = None #[128,128]
_C.DATA.TRAIN_DATALOADER.aug = ['hflip', 'rot']
_C.DATA.TRAIN_DATALOADER.n_repeat = 1
_C.DATA.TRAIN_DATALOADER.n_data = None
_C.DATA.TRAIN_DATALOADER.step = 1
_C.DATA.TRAIN_DATALOADER.scale = None

_C.DATA.VALIDATION_DATALOADER = CN()
_C.DATA.VALIDATION_DATALOADER.add_noise = None
_C.DATA.VALIDATION_DATALOADER.crop_size = None 
_C.DATA.VALIDATION_DATALOADER.aug = None
_C.DATA.VALIDATION_DATALOADER.n_repeat = 1
_C.DATA.VALIDATION_DATALOADER.n_data = None
_C.DATA.VALIDATION_DATALOADER.step = 1
_C.DATA.VALIDATION_DATALOADER.scale = None


_C.DATA.TEST_DATALOADER = CN()
_C.DATA.TEST_DATALOADER.add_noise = None
_C.DATA.TEST_DATALOADER.crop_size = None 
_C.DATA.TEST_DATALOADER.aug = None
_C.DATA.TEST_DATALOADER.n_repeat = 1
_C.DATA.TEST_DATALOADER.n_data = None
_C.DATA.TEST_DATALOADER.step = 1
_C.DATA.TEST_DATALOADER.scale = None

_C.TRAIN = CN()
_C.TRAIN.init_lr = 1e-4
_C.TRAIN.warmup = False
_C.TRAIN.warmup_iters = 200
_C.TRAIN.loss = "1*L1Loss"
_C.TRAIN.max_epochs = 20
_C.TRAIN.tmp_info = []
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.C2N = CN()
_C.TRAIN.LOSS.C2N.gp_lambda = 0.
_C.TRAIN.LOSS.C2N.w_stb = 0.

_C.TRAIN.NOISE_GENERATOR_TRAINER = CN()
_C.TRAIN.NOISE_GENERATOR_TRAINER.generator_iter_step = 5
_C.TRAIN.NOISE_GENERATOR_TRAINER.critic_noise = False

_C.OPTIMIZER = CN()
_C.OPTIMIZER.type = "Adam"

_C.OPTIMIZER.ADAM = CN()
_C.OPTIMIZER.ADAM.betas = [0.9, 0.999]

_C.SCHEDULER = CN()
_C.SCHEDULER.type = "step"

_C.SCHEDULER.STEP = CN()
_C.SCHEDULER.STEP.step_size = 8
_C.SCHEDULER.STEP.gamma = 0.1

_C.CKPT = CN()
_C.CKPT.save = True
_C.CKPT.start_epoch = 1
_C.CKPT.interval_epoch = 1

_C.VALIDATION = CN()
_C.VALIDATION.save_image = True
_C.VALIDATION.start_epoch = 1
_C.VALIDATION.interval_epoch = 1
_C.VALIDATION.add_con = 0.0
_C.VALIDATION.floor = False
_C.VALIDATION.scale = 1.
_C.VALIDATION.using_bits = None

_C.TEST = CN()
_C.TEST.ckpt_epoch = -1
_C.TEST.test_img = None
_C.TEST.test_dir = None
_C.TEST.self_en = None
_C.TEST.add_con = 0.5
_C.TEST.save_image = True
_C.TEST.floor = True
_C.TEST.crop = False
_C.TEST.scale = 1.
_C.TEST.using_bits = None
_C.TEST.proc = False
_C.TEST.imread = 'gray'
_C.TEST.save_dir = "./"
_C.TEST.save_ext = "tif"

_C.TEST.TEST_DIR = CN()
_C.TEST.TEST_DIR.postfix = ''
_C.TEST.TEST_DIR.save_original_img = False 
_C.TEST.TEST_DIR.no_input_frames = 1

_C.TEST.CROP = CN()
_C.TEST.CROP.size = 256
_C.TEST.CROP.overlap = 128

_C.TEST.PROC = CN()
_C.TEST.PROC.bias = 0
_C.TEST.PROC.blending = 1.0


def get_cfg_defaults():
  return _C.clone()
