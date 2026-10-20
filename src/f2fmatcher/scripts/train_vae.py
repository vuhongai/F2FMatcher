import os
import torch
from tqdm import tqdm

from f2fmatcher.config import load_config
from f2fmatcher.utils.seed import set_seed
from f2fmatcher.segmentation.cellpose_seg import generate_VAE_inputs
from f2fmatcher.vae.dataset import CellposeDataset
from f2fmatcher.vae.models import SharedMultiHeadVAE
from f2fmatcher.vae.train import TrainFMmodel, plot_model_performance
from torch.utils.data import DataLoader, random_split


def train_vae_main(args):
    config = load_config(args.config)
    set_seed(config.get("seed", 1024))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    crop_size = config.get("dataset.crop_size", 256)
    resize = config.get("dataset.resize", 128)
    latent_dim = config.get("vae.latent_dim", 256)
    batch_size = config.get("vae.batch_size", 32)
    lr = config.get("vae.lr", 1e-3)
    n_epochs = config.get("vae.n_epochs", 20)
    patience = config.get("vae.patience", 10)

    ds = CellposeDataset(data_dir=config.get("dataset.dir"), size=resize)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SharedMultiHeadVAE(latent_dim=latent_dim).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

    model = TrainFMmodel(model, train_loader, val_loader, device,
                         n_epochs=n_epochs, lr=lr,
                         checkpoint_dir=args.checkpoint_dir, patience=patience)

    plot_model_performance(model, val_loader, device,
                           save_path=f"{args.checkpoint_dir}/val_reconstruction.png")
    print(f"Training complete. Checkpoints saved to {args.checkpoint_dir}")
