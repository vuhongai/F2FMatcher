import os
import numpy as np
import itertools
import torch
from tqdm import tqdm

from f2fmatcher.matching.spatial import (
    spatial_signature,
    wasserstein_similarity_matrix_parallel,
)


def get_classifier_logits(img1, list_label_1, img2, list_label_2, classifier, dir_embedding, batch_size=2048):
    device = next(classifier.parameters()).device

    emb1 = torch.stack([
        torch.tensor(np.load(os.path.join(dir_embedding, f"{img1}_roi_{l}.npy")), dtype=torch.float32)
        for l in list_label_1
    ]).to(device)

    emb2 = torch.stack([
        torch.tensor(np.load(os.path.join(dir_embedding, f"{img2}_roi_{l}.npy")), dtype=torch.float32)
        for l in list_label_2
    ]).to(device)

    n1, n2 = emb1.shape[0], emb2.shape[0]
    scores = torch.zeros(n1, n2, device=device)

    idx_pairs = list(itertools.product(range(n1), range(n2)))

    with torch.no_grad():
        for i in range(0, len(idx_pairs), batch_size):
            batch = idx_pairs[i:i + batch_size]
            e1 = torch.stack([emb1[i1] for i1, _ in batch])
            e2 = torch.stack([emb2[i2] for _, i2 in batch])
            logits = classifier(e1, e2).squeeze()
            for (i1, i2), logit in zip(batch, logits):
                scores[i1, i2] = logit

    return scores.cpu().numpy()


def compute_cost_matrix(img1, list_label_1, cp_output_1, img2, list_label_2, cp_output_2,
                        classifier, dir_embedding, list_k=(3, 5, 7)):
    scores_path = f"{dir_embedding}/{img1}_vs_{img2}_scores.npy"
    if os.path.exists(scores_path):
        scores = np.load(scores_path)
    else:
        scores = get_classifier_logits(img1, list_label_1, img2, list_label_2, classifier, dir_embedding)

    spatial_dist = np.ones(scores.shape)
    for k in list_k:
        _, dm1 = spatial_signature(cp_output_1, list_label_1, k)
        _, dm2 = spatial_signature(cp_output_2, list_label_2, k)
        spatial_dist *= wasserstein_similarity_matrix_parallel(dm1, dm2)
    spatial_dist **= (1 / len(list_k))

    return scores, spatial_dist
