from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from glob import glob
import random
from torchvision import transforms
import pickle

from configs import *
from FM_models import *
from FM_match import *
from FM_dataset import *

class FiberClassifer(Dataset):
    def __init__(
        self, 
        path_dataset=None, 
        list_pair_label = None,
        size=size_crop_resize,
        dir_embedding = "./embed"
    ):
        if list_pair_label is None:
            with open(path_dataset, "rb") as f:
                self.files = pickle.load(f)
        else:
            self.files = list_pair_label
            
        self.size = size
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.dir_embedding = dir_embedding

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path_1, path_2, label = self.files[idx]
        mu1 = self.get_embedding(path_1)
        mu2 = self.get_embedding(path_2)
        return mu1, mu2, torch.tensor(label, dtype=torch.float32)

    def create_input(self, path_npz):
        """Create tensor input for VAE model"""
        sample = np.load(path_npz)
        flow_x = sample['flow_x']
        flow_y = sample['flow_y']
        roi_mask = sample['roi_mask']

        # transform to magnitude and angle
        mag = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)

        # normalization
        mag = np.clip(mag / 10.0, 0, 1)
        angle = (angle + np.pi) / (2 * np.pi)

        # prepare input tensor
        input_stack = np.stack([mag, angle, roi_mask.astype(np.float32)], axis=-1)
        input_tensor = self.resize((input_stack * 255).astype(np.uint8))
        return input_tensor

    def get_embedding(self, path_npz):
        base = os.path.basename(path_npz).replace(".npz", ".npy")
        p_embed = os.path.join(self.dir_embedding, base)
        mu = np.load(p_embed)
        return mu

# === Classifier Model ===
class PairClassifier(nn.Module):
    def __init__(self, embedding_dim=size_latent):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        x = torch.cat([emb1, emb2], dim=-1)
        return self.fc(x)

def train_classifier(
    train_loader, 
    val_loader, 
    device,
    lr,
    checkpoint_path,
    n_epochs,
    patience,
):
    model = PairClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # early stopping
    best_f1 = float(0)
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for emb1, emb2, label in tqdm(train_loader):
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)
            pred = model(emb1, emb2).squeeze()
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, loss_train={avg_loss:.4f}, loss_val={val_loss:.4f}, acc_val={acc:.4f}, precision_val={prec:.4f}, recall_val={rec:.4f}, F1_val={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   Save checkpoint at epoch {epoch+1} with val_f1={f1:.4f} \n")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for emb1, emb2, label in loader:
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)
            pred = model(emb1, emb2).squeeze()
            loss = criterion(pred, label)
            total_loss += loss.item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, all_preds_bin)
    prec = precision_score(all_labels, all_preds_bin, zero_division=0)
    rec = recall_score(all_labels, all_preds_bin, zero_division=0)
    f1 = f1_score(all_labels, all_preds_bin, zero_division=0)
    return total_loss / len(loader), acc, prec, rec, f1