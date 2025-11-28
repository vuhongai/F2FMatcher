from configs import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from glob import glob
import random
from torchvision import transforms
import os

class CellposeDataset(Dataset):
    def __init__(self, data_dir=None, list_files=None, size=size_crop_resize):
        if list_files==None:
            self.files = glob(os.path.join(data_dir, '*.npz'))
        else:
            self.files = list_files
        self.size = size
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        flow_x = sample['flow_x']
        flow_y = sample['flow_y']
        roi_mask = sample['roi_mask']

        mag = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)

        mag = np.clip(mag / 10.0, 0, 1)
        angle = (angle + np.pi) / (2 * np.pi)

        input_stack = np.stack([mag, angle, roi_mask.astype(np.float32)], axis=-1)
        input_tensor = self.resize((input_stack * 255).astype(np.uint8))

        flow_x = self.resize(flow_x.astype(np.float32))
        flow_y = self.resize(flow_y.astype(np.float32))
        roi_mask = self.resize(roi_mask.astype(np.float32))

        return input_tensor, flow_x, flow_y, roi_mask

class PairedCellposeDataset(Dataset):
    def __init__(
        self, 
        list_path_crop_ori, 
        list_path_crop_aug, 
        size=size_crop_resize
    ):
        """
        list_path_crop_ori, list_path_crop_aug: path to the .npz crop files, the order and length in 2 lists are the same
        """
        self.files = list_path_crop_ori
        self.files_aug = list_path_crop_aug
        assert len(self.files) == len(self.files_aug)

        self.crop_ids = [f.split('/')[-1].split('.')[0] for f in self.files]
        self.size = size
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_ori = self.load_sample(self.files[idx])
        x_aug = self.load_sample(self.files_aug[idx])
        return x_ori, x_aug

    def load_sample(self, path_npz):
        sample = np.load(path_npz)
        flow_x = sample['flow_x']
        flow_y = sample['flow_y']
        roi_mask = sample['roi_mask']

        mag = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)

        mag = np.clip(mag / 10.0, 0, 1)
        angle = (angle + np.pi) / (2 * np.pi)

        input_stack = np.stack([mag, angle, roi_mask.astype(np.float32)], axis=-1)
        input_tensor = self.resize((input_stack * 255).astype(np.uint8))

        flow_x = self.resize(flow_x.astype(np.float32))
        flow_y = self.resize(flow_y.astype(np.float32))
        roi_mask = self.resize(roi_mask.astype(np.float32))

        return input_tensor, flow_x, flow_y, roi_mask


class DatasetForEmbeddingExtraction(Dataset):
    def __init__(
        self, 
        list_path_npz_files, 
        size=size_crop_resize
    ):
        self.files = list_path_npz_files
        self.size = size
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        input_tensor = self.create_input(path)
        return input_tensor

    def create_input(self, path_npz):
        sample = np.load(path_npz)
        flow_x = sample['flow_x']
        flow_y = sample['flow_y']
        roi_mask = sample['roi_mask']

        mag = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)

        mag = np.clip(mag / 10.0, 0, 1)
        angle = (angle + np.pi) / (2 * np.pi)

        input_stack = np.stack([mag, angle, roi_mask.astype(np.float32)], axis=-1)
        input_tensor = self.resize((input_stack * 255).astype(np.uint8))
        return input_tensor