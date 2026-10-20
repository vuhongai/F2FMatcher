import os
import torch
from torch.utils.data import DataLoader, random_split

from f2fmatcher.config import load_config
from f2fmatcher.utils.seed import set_seed
from f2fmatcher.classifier.dataset import FiberClassifer
from f2fmatcher.classifier.train import train_classifier


def train_classifier_main(args):
    config = load_config(args.config)
    set_seed(config.get("seed", 1024))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = config.get("classifier.batch_size", 256)
    lr = config.get("classifier.lr", 1e-4)
    n_epochs = config.get("classifier.n_epochs", 50)
    patience = config.get("classifier.patience", 10)
    dir_emb = config.get("classifier.dir_embedding", "./embed")

    ds = FiberClassifer(path_dataset=config.get("classifier.dataset_path"), dir_embedding=dir_emb)
    n_val = max(1, int(len(ds) * 0.15))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    train_classifier(train_loader, val_loader, device, lr=lr,
                     checkpoint_path=args.checkpoint_path,
                     n_epochs=n_epochs, patience=patience)
    print(f"Training complete. Model saved to {args.checkpoint_path}")
