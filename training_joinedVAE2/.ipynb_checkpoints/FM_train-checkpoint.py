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

from FM_dataset import *
from FM_models import *
from contextlib import redirect_stdout


def loss_fn(preds, targets, beta_KL=0.001, beta_consistency=1):
    preds1, preds2 = preds
    p1_fx, p1_fy, p1_mask, p1_mu, p1_logvar = preds1
    p2_fx, p2_fy, p2_mask, p2_mu, p2_logvar = preds2
    
    target1, target2 = targets
    fx1, fy1, rm1 = target1
    fx2, fy2, rm2 = target2

    recon_loss = F.mse_loss(p1_fx, fx1) + F.mse_loss(p2_fx, fx2) + \
                 F.mse_loss(p1_fy, fy1) + F.mse_loss(p2_fy, fy2) + \
                 F.mse_loss(p1_mask, rm1) + F.mse_loss(p2_mask, rm2)

    recon_loss = recon_loss / 6

    kl_div = \
        (-0.5 * torch.sum(1 + p1_logvar - p1_mu.pow(2) - p1_logvar.exp()) / p1_mu.size(0)) + \
        (-0.5 * torch.sum(1 + p2_logvar - p2_mu.pow(2) - p2_logvar.exp()) / p2_mu.size(0))
    kl_div = (kl_div * beta_KL) / 2

    latent_consistency_loss = F.mse_loss(p1_mu, p2_mu)
    latent_consistency_loss = latent_consistency_loss * beta_consistency
    
    loss = recon_loss + kl_div + latent_consistency_loss
    return loss, recon_loss, kl_div, latent_consistency_loss

# ==== Training Loop ====
def train(
    model, dataloader, optimizer, device, 
    log_interval=10,
    beta_KL=0.001, beta_consistency=1, # config loss
    
):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_latent = 0
    
    for step, batch in enumerate(dataloader):
        x_ori, x_aug = [x for x in batch]
        inputs1, fx1, fy1, rm1 = [x.to(device) for x in x_ori]
        inputs2, fx2, fy2, rm2 = [x.to(device) for x in x_aug]
        target1 = (fx1, fy1, rm1)
        target2 = (fx2, fy2, rm2)
        targets = (target1, target2)
        
        optimizer.zero_grad()
        
        preds1 = model(inputs1)
        p1_fx, p1_fy, p1_mask, p1_mu, p1_logvar = preds1
        preds2 = model(inputs2)
        p2_fx, p2_fy, p2_mask, p2_mu, p2_logvar = preds2
        preds = (preds1, preds2)
        
        loss, recon_loss, kl_div, latent_consistency_loss = loss_fn(preds, targets, beta_KL, beta_consistency)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        total_latent += latent_consistency_loss.item()

        if (step + 1) % log_interval == 0:
            print(f"   Step {step+1}/{len(dataloader)}: loss={loss.item():.4f}, recon={recon_loss.item():.4f}, KL={kl_div.item():.4f}, latent={latent_consistency_loss.item():.4f}")

    return total_loss / len(dataloader), total_recon / len(dataloader), total_kl / len(dataloader), total_latent / len(dataloader)

def validate(
    model, dataloader, device,
    beta_KL=0.001, beta_consistency=1, # config loss
):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_latent = 0
    with torch.no_grad():
        for batch in dataloader:
            x_ori, x_aug = [x for x in batch]
            inputs1, fx1, fy1, rm1 = [x.to(device) for x in x_ori]
            inputs2, fx2, fy2, rm2 = [x.to(device) for x in x_aug]
            target1 = (fx1, fy1, rm1)
            target2 = (fx2, fy2, rm2)
            targets = (target1, target2)
            
            preds1 = model(inputs1)
            p1_fx, p1_fy, p1_mask, p1_mu, p1_logvar = preds1
            preds2 = model(inputs2)
            p2_fx, p2_fy, p2_mask, p2_mu, p2_logvar = preds2
            preds = (preds1, preds2)
            
            loss, recon_loss, kl_div, latent_consistency_loss = loss_fn(preds, targets, beta_KL, beta_consistency)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_div.item()
            total_latent += latent_consistency_loss.item()

    return total_loss / len(dataloader), total_recon / len(dataloader), total_kl / len(dataloader), total_latent / len(dataloader)

def TrainFMmodel(
    list_path_crop_ori_train, 
    list_path_crop_aug_train,
    list_path_crop_ori_val, 
    list_path_crop_aug_val,
    list_path_crop_ori_test, 
    list_path_crop_aug_test, 
    batch_size=128, 
    beta_KL=0.001, beta_consistency=1, 
    n_log_train=3,
    n_epoch = 20,
    learning_rate = 1e-3,
    checkpoint_path = f"{checkpoint_dir}/SharedVAE_{size_crop}_{size_crop_resize}.pth",
    log_path="./logs/training_log.txt",
    re_train=False, checkpoint_path_previous=None,
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f_log, redirect_stdout(f_log):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        model = SharedMultiHeadVAE().to(device)
        print(f"Model size: {count_parameters(model)} \n")

        if re_train:
            model.load_state_dict(torch.load(checkpoint_path_previous))
            print(f"Model weight {checkpoint_path_previous.split('/')[-1]} loaded. \n")
        
        train_dataset = PairedCellposeDataset(
            list_path_crop_ori_train, 
            list_path_crop_aug_train, 
            size=size_crop_resize
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = PairedCellposeDataset(
            list_path_crop_ori_val, 
            list_path_crop_aug_val, 
            size=size_crop_resize
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"n_train: {len(train_dataset)}, n_val: {len(val_dataset)}")
            
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_log_interval = int(len(train_dataset) / (batch_size*n_log_train)) + 2

        # Plot the initial prediction
        plot_model_performance(
            list_path_crop_ori_test, 
            list_path_crop_aug_test, 
            model, device, epoch=-1,
            dir_log="./logs"
        )
        
        # training loop
        best_val_loss = float('inf')
        for epoch in range(n_epoch):
            print(f"Epoch {epoch+1}/{n_epoch}:")
            train_loss, train_recon, train_kl, train_latent = train(
                model, train_loader, optimizer, device, 
                train_log_interval,
                beta_KL, beta_consistency
            )
            val_loss, val_recon, val_kl, val_latent = validate(
                model, val_loader, device,
                beta_KL, beta_consistency
            )
            
            print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Latent: {train_latent:.4f}) | "
                f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Latent: {val_latent:.4f})\n")

            # Plot the prediction
            plot_model_performance(
                list_path_crop_ori_test, 
                list_path_crop_aug_test, 
                model, device, epoch,
                dir_log="./logs"
            )

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"   Checkpoint saved at epoch {epoch+1} with loss_val={val_loss:.4f}")

    # Load best model after training
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

def plot_model_performance(
    list_path_crop_ori_test, 
    list_path_crop_aug_test, 
    model, device, epoch,
    dir_log="./logs"
):
    """visualize one image of val_dataset"""

    test_dataset = PairedCellposeDataset(
        list_path_crop_ori_test, 
        list_path_crop_aug_test, 
        size=size_crop_resize
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for batch in test_loader:
        x_ori, x_aug = [x for x in batch]
    
        inputs1, fx1, fy1, rm1 = [x.to(device) for x in x_ori]
        inputs2, fx2, fy2, rm2 = [x.to(device) for x in x_aug]
        
        p1_fx, p1_fy, p1_mask, p1_mu, p1_logvar = model(inputs1)
        p2_fx, p2_fy, p2_mask, p2_mu, p2_logvar = model(inputs2)

    ### plot - original crop
    _, axs = plt.subplots(1, 6, figsize=(8,3))
    
    # ground truth
    axs[0].imshow(fx1[0,0,...].detach().cpu().numpy())
    axs[0].axis('off')

    axs[1].imshow(fy1[0,0,...].detach().cpu().numpy())
    axs[1].axis('off')
    
    axs[2].imshow(rm1[0,0,...].detach().cpu().numpy())
    axs[2].axis('off')

    # prediction
    axs[3].imshow(p1_fx[0,0,...].detach().cpu().numpy())
    axs[3].axis('off')
    
    axs[4].imshow(p1_fy[0,0,...].detach().cpu().numpy())
    axs[4].axis('off')
    
    axs[5].imshow(p1_mask[0,0,...].detach().cpu().numpy())
    axs[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{dir_log}/plot_model_epoch{epoch+1}_original.png", dpi=150, bbox_inches='tight')
    plt.show()

    ### plot - augmented crop
    _, axs = plt.subplots(1, 6, figsize=(8,3))
    
    # ground truth
    axs[0].imshow(fx2[0,0,...].detach().cpu().numpy())
    axs[0].axis('off')

    axs[1].imshow(fy2[0,0,...].detach().cpu().numpy())
    axs[1].axis('off')
    
    axs[2].imshow(rm2[0,0,...].detach().cpu().numpy())
    axs[2].axis('off')

    # prediction
    axs[3].imshow(p2_fx[0,0,...].detach().cpu().numpy())
    axs[3].axis('off')
    
    axs[4].imshow(p2_fy[0,0,...].detach().cpu().numpy())
    axs[4].axis('off')
    
    axs[5].imshow(p2_mask[0,0,...].detach().cpu().numpy())
    axs[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{dir_log}/plot_model_epoch{epoch+1}_augmented.png", dpi=150, bbox_inches='tight')
    plt.show()