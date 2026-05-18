import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class FiberClassifer(Dataset):
    def __init__(self, path_dataset=None, list_pair_label=None, size=128, dir_embedding="./embed"):
        if list_pair_label is None:
            import pickle
            with open(path_dataset, "rb") as f:
                self.files = pickle.load(f)
        else:
            self.files = list_pair_label
        self.dir_embedding = dir_embedding
        self.resize = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path_1, path_2, label = self.files[idx]
        mu1 = self._load_embedding(path_1)
        mu2 = self._load_embedding(path_2)
        return (mu1, mu2, torch.tensor(label, dtype=torch.float32))

    def _load_embedding(self, path_npz):
        base = os.path.basename(path_npz).replace(".npz", ".npy")
        return np.load(os.path.join(self.dir_embedding, base))
