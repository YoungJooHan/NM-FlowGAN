import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from math import log10, exp
import random
import torch.nn.functional as F
import os

def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        n = np.expand_dims(n, axis=2)
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))
    
def np2tensor_multi(n:np.array):
    t = None
    if len(n) <= 1: # single stacked image
        t = np2tensor(n[0].astype(np.float32)).unsqueeze(0).float()
    else: # multi stacked image
        for mat in n:
            if t is None: t = np2tensor(mat.astype(np.float32)).unsqueeze(0).float()
            else: t = torch.cat([t, np2tensor(mat.astype(np.float32)).unsqueeze(0).float()], dim=0)
    return t

def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s'%name, tensor2np(t.cpu()))

def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s'%name))

def rot_hflip_img(img:torch.Tensor, rot_times:int=0, hflip:int=0):
    '''
    rotate '90 x times degree' & horizontal flip image 
    (shape of img: b,c,h,w or c,h,w)
    '''
    b=0 if len(img.shape)==3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+2).flip(b+1)
        # 270 degrees
        else:               
            return img.flip(b+2).transpose(b+1,b+2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img.flip(b+2)
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).flip(b+2).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+1)
        # 270 degrees
        else:               
            return img.transpose(b+1,b+2)
        
def psnr(x, y, mask=None, max_val=1.):
    if max_val is None : max_val = 1.
    if mask is None:
        mse = torch.mean((x - y) ** 2)
    else:
        mse = torch.sum(((x - y) ** 2) * mask) / mask.sum() 
    return 10 * log10(max_val**2 / mse.item())

def ssim(x, y, mask=None, data_range=None):
    x = x[0,0].cpu().numpy()
    y = y[0,0].cpu().numpy()
    mssim, S = structural_similarity(x, y, full=True, data_range=data_range)
    if mask is not None:
        mask = mask[0,0].cpu().numpy()
        return (S * mask).sum() / mask.sum()
    else:
        return mssim
    
class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def setup_determinism(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    '''
    return 2d gaussian filter window as tensor form
    Arg:
        window_size : filter window size
        sigma : standard deviation
    '''
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = gauss.unsqueeze(1)
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device).unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d/filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)

def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    '''
    return 2d mean filter as tensor form
    Args:
        window_size : filter window size
    '''
    window = torch.ones((window_size, window_size), device=device)
    window = (window/window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)

def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=None, keep_sigma=False, padd=True):
    '''
    color channel-wise 2d mean or gaussian convolution
    Args:
        x : input image
        window_size : filter window size
        filter_type(opt) : 'gau' or 'mean'
        sigma : standard deviation of gaussian filter
    '''
    b_x = x.unsqueeze(0) if len(x.shape) == 3 else x

    if window is None:
        if sigma is None: sigma = (window_size-1)/6
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=b_x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=b_x.shape[1], device=x.device)
    else:
        window_size = window.shape[-1]

    if padd:
        pl = (window_size-1)//2
        b_x = F.pad(b_x, (pl,pl,pl,pl), 'reflect')

    m_b_x = F.conv2d(b_x, window, groups=b_x.shape[1])

    if keep_sigma:
        m_b_x /= (window**2).sum().sqrt()

    if len(x.shape) == 4:
        return m_b_x
    elif len(x.shape) == 3:
        return m_b_x.squeeze(0)
    else:
        raise ValueError('input image shape is not correct')
    
def get_file_name_from_path(path):
    if '/' in path : name = path.split('/')[-1].split('.')[:-1]
    elif '\\' in path: name = path.split('\\')[-1].split('.')[:-1]
    else: assert False, f'Invalid path: {path}'

    if isinstance(name, list):
        merged = ""
        for token in name[:-1]: 
            merged += token + '.'
        merged += name[-1]
        name = merged
    return name

def kl_div_3_data(real_noise, gen_noise, bin_edges=None, left_edge=0.0, right_edge=1.0):
    # Kousha, Shayan, et al. "Modeling srgb camera noise with normalizing flows." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
    # ref) https://github.com/SamsungLabs/Noise2NoiseFlow
    noise_pats = (gen_noise, real_noise)	

    # histograms
    bw = 4
    bin_edges = np.arange(left_edge, right_edge, bw)

    cnt_regr = 1
    hists = [None] * len(noise_pats)	
    klds = np.ndarray([len(noise_pats)])	
    klds[:] = 0.0

    for h in reversed(range(len(noise_pats))):
        hists[h] = get_histogram(noise_pats[h], bin_edges=bin_edges, cnt_regr=cnt_regr)
        klds[h] = kl_div_forward(hists[-1], hists[h])	

    return klds[0]

def get_histogram(data, bin_edges=None, cnt_regr=1):
    n = np.prod(data.shape)	
    hist, _ = np.histogram(data, bin_edges)	
    return (hist + cnt_regr)/(n + cnt_regr * len(hist))

def kl_div_forward(p, q):
    assert (~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))).all()	
    idx = (p > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))

def load_numpy_from_raw(path, dtype='float32'):
    fid = open(path, "rb")
    return np.fromfile(fid, dtype=dtype)

def make_predefiend_1d_to_2d(arr):
    predefined_sizes = [(3072,2560), (3072,3072), (9216,3072), (6144,3072)] # H, W
    assert len(arr.shape) == 1
    for predefined_size in predefined_sizes:
        if arr.shape[0] == (predefined_size[0] * predefined_size[1]):
            arr = np.reshape(arr, predefined_size)
    assert len(arr.shape) == 2, "Error: No matching predefined size exists."
    return arr 

def save_img(dir_name, file_name, img):
    path = os.path.join(dir_name, file_name)
    if 'raw' in path[-3:]:
        os.makedirs(dir_name, exist_ok=True)
        with open(path, 'w') as fid:
            img.tofile(fid)
    else:
        if len(img.shape) == 3 and img.shape[-1] != 3 and img.shape[-1] > 1:
            cv2.imwritemulti(path, img.transpose([2,0,1])) # multi stack image, convert to CHW
        elif len(img.shape) == 4 and img.shape[0] > 1: # batch image, only grey image is available
            img = img.squeeze(-1)
            cv2.imwritemulti(path, img) 
        elif len(img.shape) == 4 and img.shape[0] <= 1: # single batch image
            img = img.squeeze(0)
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, img)
