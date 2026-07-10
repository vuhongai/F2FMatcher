import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from f2fmatcher.vae.models import SharedMultiHeadVAE
from f2fmatcher.vae.dataset import CellposeDataset


def loss_fn(recon_fx, recon_fy, recon_mask, fx, fy, mask,
            mu, logvar, beta_kl=0.001, beta_consistency=1.0):
    recon_loss = (
        F.mse_loss(recon_fx, fx, reduction="sum")
        + F.mse_loss(recon_fy, fy, reduction="sum")
        + F.mse_loss(recon_mask, mask, reduction="sum")
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # latent consistency: for PairedCellposeDataset, mu1 and mu2 would be passed
    # here we keep it simple — single image reconstruction loss
    return recon_loss + beta_kl * kl_loss, recon_loss, kl_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        if len(batch) == 4:
            input_tensor, fx, fy, mask = [b.to(device) for b in batch]
        else:
            input_tensor = batch[0].to(device)
            fx, fy, mask = None, None, None

        recon_fx, recon_fy, recon_mask, mu, logvar = model(input_tensor)
        if fx is not None:
            loss, recon_loss, kl_loss = loss_fn(
                recon_fx, recon_fy, recon_mask, fx, fy, mask, mu, logvar
            )
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(loader, desc="Val"):
        if len(batch) == 4:
            input_tensor, fx, fy, mask = [b.to(device) for b in batch]
        else:
            input_tensor = batch[0].to(device)
            fx, fy, mask = None, None, None

        recon_fx, recon_fy, recon_mask, mu, logvar = model(input_tensor)
        if fx is not None:
            loss, recon_loss, kl_loss = loss_fn(
                recon_fx, recon_fy, recon_mask, fx, fy, mask, mu, logvar
            )
        else:
            loss = torch.tensor(0.0, device=device)

        total_loss += loss.item()
    return total_loss / len(loader)


def TrainFMmodel(model, train_loader, val_loader, device, n_epochs=20, lr=1e-3,
                 checkpoint_dir="./checkpoints", patience=10):
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/vae_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    return model


def plot_model_performance(model, loader, device, save_path=None):
    model.eval()
    images, targets = next(iter(loader))
    input_tensor = images[0:1].to(device)
    with torch.no_grad():
        recon_fx, recon_fy, recon_mask, mu, logvar = model(input_tensor)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(images[0, 0].cpu()); axes[0, 0].set_title("Input mag")
    axes[0, 1].imshow(images[0, 1].cpu()); axes[0, 1].set_title("Input angle")
    axes[0, 2].imshow(images[0, 2].cpu()); axes[0, 2].set_title("Input mask")
    axes[1, 0].imshow(recon_fx[0, 0].cpu()); axes[1, 0].set_title("Recon fx")
    axes[1, 1].imshow(recon_fy[0, 0].cpu()); axes[1, 1].set_title("Recon fy")
    axes[1, 2].imshow(recon_mask[0, 0].cpu()); axes[1, 2].set_title("Recon mask")
    for ax in axes.ravel():
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
