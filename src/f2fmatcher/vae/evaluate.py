import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


def visualize_latent_space(embeddings, labels=None, method="umap", title="Latent Space"):
    if method == "umap":
        reducer = umap.UMAP(random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", s=5, alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def visualize_latent_space_pca(embeddings, labels=None):
    return visualize_latent_space(embeddings, labels, method="pca")


def mse_loss_np(x, y):
    return np.mean((x - y) ** 2)


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def cosine_distance(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))
