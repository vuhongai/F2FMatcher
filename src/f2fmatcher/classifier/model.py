import torch
import torch.nn as nn


class PairClassifier(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb1, emb2):
        return self.fc(torch.cat([emb1, emb2], dim=-1))


def compute_classifier_logits(classifier, emb1, emb2, device, batch_size=2048):
    classifier.eval()
    n1, n2 = emb1.shape[0], emb2.shape[0]
    logits = np.zeros((n1, n2), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n1, batch_size):
            ei = torch.from_numpy(emb1[i:i + batch_size]).float().to(device)
            for j in range(0, n2, batch_size):
                ej = torch.from_numpy(emb2[j:j + batch_size]).float().to(device)
                ei_exp = ei.unsqueeze(1).expand(-1, ej.size(0), -1)
                ej_exp = ej.unsqueeze(0).expand(ei.size(0), -1, -1)
                logits[i:i + batch_size, j:j + batch_size] = (
                    classifier(ei_exp.reshape(-1, ei.shape[-1]),
                               ej_exp.reshape(-1, ej.shape[-1]))
                    .reshape(ei.size(0), ej.size(0))
                    .cpu().numpy()
                )
    return logits
