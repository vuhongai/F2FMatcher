import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob


class CellposeDataset(Dataset):
    def __init__(self, data_dir=None, list_files=None, size=128):
        if list_files is None:
            self.files = glob(os.path.join(data_dir, "*.npz"))
        else:
            self.files = list_files
        self.resize = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        input_tensor = _build_input(sample, self.resize)
        flow_x = self.resize(sample["flow_x"].astype(np.float32))
        flow_y = self.resize(sample["flow_y"].astype(np.float32))
        roi_mask = self.resize(sample["roi_mask"].astype(np.float32))
        return input_tensor, flow_x, flow_y, roi_mask


class PairedCellposeDataset(Dataset):
    def __init__(self, list_path_crop_ori, list_path_crop_aug, size=128):
        assert len(list_path_crop_ori) == len(list_path_crop_aug)
        self.files = list_path_crop_ori
        self.files_aug = list_path_crop_aug
        self.resize = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_ori = self._load(self.files[idx])
        x_aug = self._load(self.files_aug[idx])
        return x_ori, x_aug

    def _load(self, path):
        sample = np.load(path)
        return _build_input(sample, self.resize)


class DatasetForEmbeddingExtraction(Dataset):
    def __init__(self, list_path_npz_files, size=128):
        self.files = list_path_npz_files
        self.resize = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.create_input(self.files[idx])

    def create_input(self, path_npz):
        sample = np.load(path_npz)
        return _build_input(sample, self.resize)


def _build_input(sample, resize):
    flow_x = sample["flow_x"]
    flow_y = sample["flow_y"]
    roi_mask = sample["roi_mask"]

    mag = np.clip(np.sqrt(flow_x ** 2 + flow_y ** 2) / 10.0, 0, 1)
    angle = (np.arctan2(flow_y, flow_x) + np.pi) / (2 * np.pi)
    input_stack = np.stack([mag, angle, roi_mask.astype(np.float32)], axis=-1)
    return resize((input_stack * 255).astype(np.uint8))
