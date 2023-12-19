import argparse
import os
import h5py
import cv2
from tqdm import tqdm
import random
import numpy as np

######## This dataset splitting strategy is referenced from sRGB denoising (https://github.com/SamsungLabs/Noise2NoiseFlow)
TRAIN_NOISEGEN_INDICES = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                        90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                        138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
                        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    
TRAIN_DENOISER_INDICES = [1,   2,   3,   4,   5,   6,   7,   8,  10,  11,  12, 13,  14,  15,  16,  17,  18,
                        19,  20,  22,  23,  25, 27,  28,  29,  30,  32,  33,  34,  35,  38, 39,  40,  42,
                        43,  44,  45,  47,  48,  51, 52,  54,  55,  57,  59,  60,  62,  63, 66,  75,  77,
                        81,  86,  87,  88,  90, 94,  98,  101, 102, 104, 105, 110, 111, 113, 114, 115, 116,
                        117, 118, 122, 125, 126, 127, 129, 132, 133, 134, 135, 136, 137, 138, 140, 142,
                        147, 149, 150, 151, 152, 154, 155, 156, 159, 160, 161, 163, 164, 165, 166, 167,
                        169, 172, 175, 177, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194,
                        195, 196, 197, 198, 199]

TRAIN_ALL_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 28, 29, 30,
                      32,  33,  34,  35, 36,  38, 39,  40,  42,
                     43,  44,  45,  47,  48,  50, 51, 52,  54,  55,  57,  59,  60,  62,  63, 64, 65, 66, 68,  69, 70, 72, 73, 75, 76, 77, 78,
                     80, 81, 83, 84,  86,  87,  88, 89, 90, 91, 92,  94, 96, 97, 98, 99,  101, 102, 104, 105, 106, 107, 108, 110, 111, 113, 114, 115, 116,
                     117, 118, 120, 121, 122, 123, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142,
                     144, 145, 146, 147, 149, 150, 151, 152, 154, 155, 156,157, 159, 160, 161, 163, 164, 165, 166, 167, 168,
                     169, 172, 173, 175, 177, 178, 179, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194,
                     195, 196, 197, 198, 199, 200]

TEST_INDICES = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 154, 155, 159, 160, 161, 163, 164, 165, 166, 198,199]
###########################################################################################################################################
     
def get_SIDD_img_paths(base_path):
    file_paths = list()
    subdirs = os.listdir(base_path)
    for subdir in subdirs:
        for full_path, _, file_names in os.walk(os.path.join(base_path, subdir)):
            for file_name in file_names:
                if 'GT' in file_name:
                    origin_path = os.path.join(full_path, file_name)
                    counterpath_path = os.path.join(full_path, file_name.replace('GT','NOISY'))
                    if os.path.exists(counterpath_path):
                        file_paths.append({'GT':origin_path,'NOISY': counterpath_path})
    return file_paths

def convert_smartphone_code(model_name):
    models = ['IP', 'GP', 'S6', 'N6', 'G4']
    for idx, model in enumerate(models):
        if model == model_name: return idx
    assert False, "Invalid camera name."

def convert_illuminant_birghtness_code(illum_code):
    codes = ['L','N','H']
    for idx, code in enumerate(codes):
        if illum_code == code: return idx
    assert False, "Invalid illuminant brightness code."

def parse_dir_name(dir_name):
    tokens = dir_name.split('_')
    return {
        'scene-instance-number': int(tokens[0]),
        'scene-number': int(tokens[1]),
        'smartphone-code': int(convert_smartphone_code(tokens[2])),
        'ISO-level': int(tokens[3]),
        'shutter-speed': int(tokens[4]),
        'illuminant-temperature': int(tokens[5]),
        'illuminant-brightness-code':int(convert_illuminant_birghtness_code(tokens[6]))
    }

def crop(img, size, overlap):
    crops = list()
    img = img.transpose(2,0,1)
    _,h,w = img.shape
    i,j = 0, 0
    while i < h:
        while j < w:
            roi_x, roi_y = j, i
            if i + size > h: roi_y = h - size 
            if j + size > w: roi_x = w - size
            crops.append(img[:,roi_y:roi_y+size,roi_x:roi_x+size])
            j+=overlap
        j=0
        i+=overlap
    i=0
    return crops

def find_support_scene(img_paths, current_path):
    random.shuffle(img_paths) # shuffling
    current_config = parse_dir_name(current_path['GT'].split('/')[-2])
    for img_path in img_paths:
        target_config = parse_dir_name(img_path['GT'].split('/')[-2])
        if target_config['smartphone-code'] == current_config['smartphone-code'] and \
        target_config['ISO-level'] == current_config['ISO-level'] and \
        target_config['scene-instance-number'] != current_config['scene-instance-number']:
            return img_path
    assert False, "There is no matching scene."

def main(args):
    img_paths = get_SIDD_img_paths(args.SIDD_path)
    for img_idx, img_path in enumerate(tqdm(img_paths)):
        img_gt = cv2.imread(img_path['GT'], cv2.IMREAD_COLOR)
        if args.rgb_flip:
            img_gt = np.flip(img_gt, axis=2)
        img_noisy = cv2.imread(img_path['NOISY'], cv2.IMREAD_COLOR)
        if args.rgb_flip:
            img_noisy = np.flip(img_noisy, axis=2)
        config = parse_dir_name(img_path['GT'].split('/')[-2])
        
        img_gt_crops = crop(img_gt, args.patch_size, args.overlap)
        img_noisy_crops = crop(img_noisy, args.patch_size, args.overlap)
        assert len(img_gt_crops) == len(img_noisy_crops)

        output_dir_path  = args.output_base_path
        if args.mode == 'NOISE_GEN':
            if config['scene-instance-number'] in TRAIN_NOISEGEN_INDICES:
                output_dir_path += '/noise_gen/train/'
            elif config['scene-instance-number'] in TEST_INDICES:
                output_dir_path += '/noise_gen/test/'
            else:
                continue
        elif args.mode == 'DENOISER':
            if config['scene-instance-number'] in TRAIN_DENOISER_INDICES:
                output_dir_path += '/denoiser/'
            else:
                continue
        elif args.mode == 'ALL':
            if config['scene-instance-number'] in TRAIN_ALL_INDICES:
                output_dir_path += '/all/'
            else:
                continue
            
        os.makedirs(output_dir_path, exist_ok=True)
        output_file_name = '%04d_%s.hdf5'%(img_idx, img_path['GT'].split('/')[-2])
        hdf_file_path = os.path.join(output_dir_path, output_file_name)
        with h5py.File(hdf_file_path, "w") as f:
            for patch_idx, (gt, noisy) in enumerate(zip(img_gt_crops, img_noisy_crops)):
                f.create_dataset(f'clean/{patch_idx}', data=gt)
                f.create_dataset(f'noisy/{patch_idx}', data=noisy)
            hdf5_config = f.create_group('config')
            for key in config:
                hdf5_config.attrs[key] = config[key]
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # DATA
    parser.add_argument('--SIDD_path', type=str, default="./data/SIDD/SIDD_Medium_Srgb")
    parser.add_argument('--output_base_path', type=str, default="./data/SIDD/HDF5_s96_o48")
    parser.add_argument('--mode', type=str, choices=['NOISE_GEN', 'DENOISER', 'ALL'], default="NOISE_GEN")

    # PATCH
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--overlap', type=int, default=48)
    parser.add_argument('--rgb_flip', type=bool, default=True)

    args = parser.parse_args()
    
    main(args)