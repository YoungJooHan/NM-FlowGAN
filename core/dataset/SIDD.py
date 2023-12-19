from .base_dataset import BaseDataset
from . import regist_dataset
import os
import scipy.io
import numpy as np
import h5py
from tqdm import tqdm


@regist_dataset
class SIDD_HDF(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_img_paths(self, dataset_path):
        assert os.path.exists(dataset_path), f"Invalid dataset path: {dataset_path}"
        self.dataset_path = dataset_path
        self.dataset_dir = dataset_path
        # scan all data and fill in self.img_paths
        self.img_paths = []
        self._scan()

    def iterate_hdf5_objects(self, obj, prefix=""):
        """
        Recursively iterate over all objects in HDF5 file.
        Returns tuple of (name, object).
        """
        for name, obj in obj.items():
            if isinstance(obj, h5py.Dataset):
                yield f"{prefix}/{name}", obj
            elif isinstance(obj, h5py.Group):
                yield from self.iterate_hdf5_objects(obj, f"{prefix}/{name}")

    def _scan(self):
        file_paths = self._get_file_paths(self.dataset_dir, ".hdf5")
        for file_path in tqdm(file_paths):
            with h5py.File(file_path, 'r') as hf:
                assert len(hf['clean']) == len(hf['noisy'])
                for name, obj in self.iterate_hdf5_objects(hf['clean']):
                    h, w, _ = obj.shape
                    self.img_paths.append({'clean': f"clean{name}", 'noisy': f"noisy{name}", 'width': w, 'height': h, 'file_path':file_path}) 
            
    def _load_data(self, data_idx):
        with h5py.File(self.img_paths[data_idx]['file_path'], 'r') as hf:
            clean_img = hf[self.img_paths[data_idx]['clean']][...].astype(np.float32)
            noisy_img = hf[self.img_paths[data_idx]['noisy']][...].astype(np.float32)
        
            if len(clean_img.shape) == 2: clean_img = np.expand_dims(clean_img, axis=0)
            if len(noisy_img.shape) == 2: noisy_img = np.expand_dims(noisy_img, axis=0)
            kwargs = dict()
            for key in hf['config'].attrs.keys():
                kwargs[key] = hf['config'].attrs[key]
        
        return {'clean': clean_img, 'real_noisy': noisy_img, 'kwargs':kwargs}
        
        
    def _get_file_paths(self, path, force_extension=".raw"):
        assert os.path.exists(path), f"{path} dosen't exist."
        assert os.path.isdir(path), f"{path} is not directory."
        
        file_paths = list()
        for r, d, f in os.walk(path):
            for file in f:
                if force_extension is not None:
                    if force_extension in file:
                        file_paths.append(os.path.join(r, file))
                else:
                    file_paths.append(os.path.join(r, file))
        file_paths.sort()
        return file_paths
    

@regist_dataset
class SIDD_val(BaseDataset):
    '''
    SIDD validation dataset class 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_dir
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        clean_mat_file_path = os.path.join(dataset_path, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(dataset_path, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        clean_img = self.clean_patches[img_id, patch_id, :].astype(float)
        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        clean_img = self._load_img_from_np(clean_img)
        noisy_img = self._load_img_from_np(noisy_img)

        return {'clean': clean_img, 'real_noisy': noisy_img }

@regist_dataset
class SIDD_benchmark(BaseDataset):
    '''
    SIDD benchmark dataset class
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_dir
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        mat_file_path = os.path.join(dataset_path, 'BenchmarkNoisyBlocksSrgb.mat')

        self.noisy_patches = np.array(scipy.io.loadmat(mat_file_path, appendmat=False)['BenchmarkNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        noisy_img = self._load_img_from_np(noisy_img)

        return {'real_noisy': noisy_img}
