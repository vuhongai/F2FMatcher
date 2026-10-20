import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from f2fmatcher.classifier.model import PairClassifier


def train_classifier(train_loader, val_loader, device, lr=1e-4,
                     checkpoint_path="./classifier.pth", n_epochs=50, patience=10):
    model = PairClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(n_epochs):
        # train
        model.train()
        total_loss = 0
        for emb1, emb2, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)
            pred = model(emb1, emb2).squeeze()
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # evaluate
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"  train_loss={total_loss / len(train_loader):.4f}  val_loss={val_loss:.4f}  "
              f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    return model


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for emb1, emb2, label in loader:
            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device)
            pred = model(emb1, emb2).squeeze()
            total_loss += criterion(pred, label).item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, preds_bin),
        precision_score(all_labels, preds_bin, zero_division=0),
        recall_score(all_labels, preds_bin, zero_division=0),
        f1_score(all_labels, preds_bin, zero_division=0),
    )
