import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_latent_space(latent_vectors, labels=None, roi_ids=None, method='tsne', title='Latent Space'):
    """
    Visualize high-dimensional latent vectors using UMAP or t-SNE.

    Args:
        latent_vectors (np.ndarray): shape [N, latent_dim]
        labels (np.ndarray or list, optional): categorical values for color
        method (str): 'umap' or 'tsne'
        title (str): plot title
    """
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'")

    embedding = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=8, alpha=1)
        plt.legend(*scatter.legend_elements(), title="Labels")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.8)

    if roi_ids is not None:
        for i, roi_id in enumerate(roi_ids):
            plt.text(embedding[i, 0], embedding[i, 1], str(roi_id),
                     fontsize=10, alpha=0.8, ha='center', va='center')

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

def visualize_latent_space_pca(latent_vectors, labels=None, roi_ids=None, title="PCA of Latent Space"):
    """
    Visualize latent vectors using PCA in 2D.

    Args:
        latent_vectors (np.ndarray): shape [N, latent_dim]
        labels (np.ndarray or list, optional): class labels for coloring
        title (str): plot title
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap="flag", s=8, alpha=1)
        plt.legend(*scatter.legend_elements(), title="Labels", loc="best")
    else:
        plt.scatter(components[:, 0], components[:, 1], s=5, alpha=0.8)

    if roi_ids is not None:
        for i, roi_id in enumerate(roi_ids):
            plt.text(components[i, 0], components[i, 1], str(roi_id), 
                     fontsize=10, alpha=1, ha='center', va='center')

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prediction(val_trues, val_preds, idx_roi1, idx_roi2):
    _, axs = plt.subplots(1, 6, figsize=(10,4))
    
    # img1
    axs[0].imshow(val_trues[0][idx_roi1,0,...])
    axs[0].axis('off')
    
    axs[1].imshow(val_trues[1][idx_roi1,0,...])
    axs[1].axis('off')
    
    axs[2].imshow(val_trues[2][idx_roi1,0,...])
    axs[2].axis('off')
    
    # img2
    axs[3].imshow(val_trues[0][idx_roi2,0,...])
    axs[3].axis('off')
    
    axs[4].imshow(val_trues[1][idx_roi2,0,...])
    axs[4].axis('off')
    
    axs[5].imshow(val_trues[2][idx_roi2,0,...])
    axs[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
    _, axs = plt.subplots(1, 6, figsize=(10,4))
    
    # img1
    axs[0].imshow(val_preds[0][idx_roi1,0,...])
    axs[0].axis('off')
    
    axs[1].imshow(val_preds[1][idx_roi1,0,...])
    axs[1].axis('off')
    
    axs[2].imshow(val_preds[2][idx_roi1,0,...])
    axs[2].axis('off')
    
    # img2
    axs[3].imshow(val_preds[0][idx_roi2,0,...])
    axs[3].axis('off')
    
    axs[4].imshow(val_preds[1][idx_roi2,0,...])
    axs[4].axis('off')
    
    axs[5].imshow(val_preds[2][idx_roi2,0,...])
    axs[5].axis('off')
    
    plt.tight_layout()
    plt.show()

def mse_loss_np(preds, targets):
    return np.mean((preds - targets) ** 2)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    cosine_sim = np.dot(a_norm, b_norm)
    return 1 - cosine_sim

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))