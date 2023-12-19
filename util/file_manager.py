import os

import cv2
import numpy as np
import torch

from .util import tensor2np, save_img

class FileManager:
    def __init__(self, session_name, output_path=None):
        if output_path is None:
            self.output_folder = "./output"
        else:
            self.output_folder = output_path
            
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
            print("[WARNING] output folder is not exist, create new one")

        # init session
        self.session_name = session_name
        os.makedirs(os.path.join(self.output_folder, self.session_name), exist_ok=True)

        # mkdir
        for directory in ['checkpoint', 'img']:
            self.make_dir(directory)

    def is_dir_exist(self, dir_name:str) -> bool:
        return os.path.isdir(os.path.join(self.output_folder, self.session_name, dir_name))

    def make_dir(self, dir_name:str) -> str:
        os.makedirs(os.path.join(self.output_folder, self.session_name, dir_name), exist_ok=True) 

    def get_dir(self, dir_name:str) -> str:
        # -> './output/<session_name>/dir_name'
        return os.path.join(self.output_folder, self.session_name, dir_name)

    def save_img_tensor(self, dir_name:str, file_name:str, img:torch.Tensor, ext='png'):
        self.save_img_numpy(dir_name, file_name, tensor2np(img), ext)

    def save_img_numpy(self, dir_name:str, file_name:str, img:np.array, ext='png'):
        if np.shape(img)[2] == 1:
            save_img(self.get_dir(dir_name), '%s.%s'%(file_name, ext), np.squeeze(img, 2))
        else:
            save_img(self.get_dir(dir_name), '%s.%s'%(file_name, ext), img)
    