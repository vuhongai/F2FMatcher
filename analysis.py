import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pathlib import Path

from .config import STAININGS_ORDER, FEATURE_STATISTICS, COMPARTMENTS, get_group


def _get_feature_columns(df):
    base_cols = {"ref_label", "n_stainings_missing", "frac_missing"}
    drop_cols = set()
    for col in df.columns:
        if col in base_cols:
            drop_cols.add(col)
        if col.endswith("_was_imputed"):
            drop_cols.add(col)
    feat_cols = sorted(set(df.columns) - drop_cols)
    feat_cols = [c for c in feat_cols if not c.startswith("Unnamed")]
    return feat_cols


def run_pca(df, n_components=50):
    feat_cols = _get_feature_columns(df)
    X = df[feat_cols].values
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    return X_pca, pca, scaler, feat_cols, explained


def run_tsne(X_pca, n_components_pca=30, random_state=42):
    n_comp = min(n_components_pca, X_pca.shape[1])
    X_input = X_pca[:, :n_comp]
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    X_tsne = tsne.fit_transform(X_input)
    return X_tsne


def run_umap(X_pca, n_components_pca=30, random_state=42):
    n_comp = min(n_components_pca, X_pca.shape[1])
    X_input = X_pca[:, :n_comp]
    reducer = umap.UMAP(random_state=random_state)
    X_umap = reducer.fit_transform(X_input)
    return X_umap


GROUP_COLORS = {
    "WT": "#4CAF50",
    "mdx": "#F44336",
    "AAV9": "#2196F3",
    "LICA1": "#FF9800",
    "unknown": "#9E9E9E",
}

GROUP_MARKERS = {
    "WT": "o",
    "mdx": "^",
    "AAV9": "s",
    "LICA1": "D",
    "unknown": ".",
}


def plot_embedding(X, labels, title, save_path, method="UMAP"):
    colors = [GROUP_COLORS.get(g, "#999999") for g in labels]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for group in sorted(set(labels)):
        mask = [l == group for l in labels]
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=GROUP_COLORS.get(group, "#999999"),
            marker=GROUP_MARKERS.get(group, "."),
            label=group,
            s=20, alpha=0.7, edgecolors="none",
        )
    ax.set_title(f"{title} - {method}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_dataset(
    df,
    output_dir,
    dataset_label="dataset",
    sample_names=None,
):
    os.makedirs(output_dir, exist_ok=True)
    groups = []
    if sample_names is None:
        sample_names = [None] * len(df)
    for s in sample_names:
        groups.append(get_group(s))

    print(f"Running PCA on {dataset_label}...")
    X_pca, pca_model, scaler, feat_cols, explained = run_pca(df)

    cum_var = np.cumsum(explained)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.6, label="Per component")
    ax.step(range(1, len(explained) + 1), cum_var, where="mid", label="Cumulative")
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="90%")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"{dataset_label} - PCA")
    ax.legend()
    pca_var_path = os.path.join(output_dir, f"{dataset_label}_pca_variance.png")
    fig.savefig(pca_var_path, dpi=150)
    plt.close(fig)

    pca_df = pd.DataFrame(
        X_pca[:, :10],
        columns=[f"PC{i+1}" for i in range(10)],
    )
    pca_df["group"] = groups
    pca_df.to_csv(
        os.path.join(output_dir, f"{dataset_label}_pca_scores.csv"), index=False
    )
    plot_embedding(
        X_pca[:, :2], groups,
        dataset_label,
        os.path.join(output_dir, f"{dataset_label}_pca.png"),
        method="PCA",
    )

    n_pca_for_embed = min(30, X_pca.shape[1])
    print(f"Running t-SNE on {dataset_label}...")
    X_tsne = run_tsne(X_pca, n_components_pca=n_pca_for_embed)
    plot_embedding(
        X_tsne, groups,
        dataset_label,
        os.path.join(output_dir, f"{dataset_label}_tsne.png"),
        method="t-SNE",
    )
    tsne_df = pd.DataFrame({"tSNE1": X_tsne[:, 0], "tSNE2": X_tsne[:, 1], "group": groups})
    tsne_df.to_csv(
        os.path.join(output_dir, f"{dataset_label}_tsne.csv"), index=False
    )

    print(f"Running UMAP on {dataset_label}...")
    X_umap = run_umap(X_pca, n_components_pca=n_pca_for_embed)
    plot_embedding(
        X_umap, groups,
        dataset_label,
        os.path.join(output_dir, f"{dataset_label}_umap.png"),
        method="UMAP",
    )
    umap_df = pd.DataFrame({"UMAP1": X_umap[:, 0], "UMAP2": X_umap[:, 1], "group": groups})
    umap_df.to_csv(
        os.path.join(output_dir, f"{dataset_label}_umap.csv"), index=False
    )

    print(f"Analysis complete for {dataset_label}")
    return {"pca": X_pca, "tsne": X_tsne, "umap": X_umap}
