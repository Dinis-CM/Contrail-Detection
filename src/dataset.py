from email.mime import image
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class Dataset(Dataset):
    def __init__(self, file_paths, is_test=False, do_augmentation=False, seed=2006):

        self.file_paths = sorted(file_paths)

        self.index_map = []

        self.is_test = is_test

        self.do_augmentation = do_augmentation

        self.handles = {}

        if self.do_augmentation:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5),
            ], seed=seed)

        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                
                n_samples = f['bands'].shape[0]

                for local_idx in range(n_samples):
                    # Loading as a map to save memory and avoid loading all data at once
                    self.index_map.append((path, local_idx))


    def __len__(self):
        return len(self.index_map)

    def _get_handle(self, path):
        # Avoid reopening the same HDF5 file multiple times in a multi-worker setting
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
        return self.handles[path]

    def convert_to_ash_false_color(self, raw_bands, scale, offset):
        # STANDARD ASH FALSE-COLOR RECIPE (Ash RGB) -> Substitutions for the available bands in the dataset:
        #
        # RED: 12.3 µm - 10.3 µm -> Band 15 (12.3 µm) - Band 14 (11.2 µm)
        #   Range [Min, Max]: [-4 K, 2 K], Gamma = 1.0
        #
        # GREEN: 11.2 µm - 8.4 µm -> Band 14 (11.2 µm) - Band 11 (8.4 µm)
        #   Range [Min, Max]: [-4 K, 5 K], Gamma = 1.0
        #
        # BLUE: 10.3 µm -> Band 14 (11.2 µm)
        #   Range [Min, Max]: [243 K, 303 K], Gamma = 1.0
        

        kelvin = raw_bands.astype(np.float32) * scale + offset

        red = self.normalize_range(kelvin[2] - kelvin[1], -4.0, 2.0)
        green = self.normalize_range(kelvin[1] - kelvin[0], -4.0, 5.0)
        blue = self.normalize_range(kelvin[1], 243.0, 303.0)

        return np.stack([red, green, blue], axis=-1).astype(np.float32)

    def normalize_range(self, data, f_min, f_max):
        return np.clip((data - f_min) / (f_max - f_min), 0.0, 1.0)

    def __getitem__(self, idx):
        file_path, local_idx = self.index_map[idx]

        data = self._get_handle(file_path)

        raw_bands = data['bands'][local_idx]
        scale = data['bands'].attrs['scale']
        offset = data['bands'].attrs['offset']

        image = self.convert_to_ash_false_color(raw_bands, scale, offset)
        
        if not self.is_test:
            
            mask = data['mask'][local_idx].astype(np.float32)
            
            if self.do_augmentation:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # For PyTorch
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
            mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)
            
            return image_tensor, mask_tensor
        else: 
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())       
            img_id = data['id'][local_idx].decode('utf-8')
            return image_tensor, img_id



