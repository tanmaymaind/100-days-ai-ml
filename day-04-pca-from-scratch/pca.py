"""Day 4 — Principal Component Analysis (PCA) from scratch (NumPy only).

Idea: a 4-feature dataset is really driven by 2 hidden factors. PCA finds
the directions (principal components) along which the data varies most,
by eigendecomposition of the covariance matrix, and projects the data
down to those directions — keeping most of the variance with fewer numbers.

Run:
    python pca.py
Saves:
    projection.svg  — data projected onto PC1 vs PC2, coloured by true cluster
    scree.svg       — explained-variance bar chart
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_PER_CLUSTER = 80
N_FEATURES = 4


def make_data(seed=SEED):
    """Synthetic 4-D data: 3 clusters, driven by 2 hidden factors.

    Each point is built from two latent factors (f1, f2) mixed into
    4 observed features through a fixed loading matrix, plus noise.
    That makes the features correlated — exactly what PCA exploits.
    """
    rng = np.random.default_rng(seed)
    # Cluster centres in latent (f1, f2) space
    centres = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]])
    # How the 2 latent factors load onto the 4 observed features
    loadings = np.array(
        [
            [1.0, 0.3],
            [0.9, -0.4],
            [0.2, 1.0],
            [-0.5, 0.9],
        ]
    )
    X_list, y_list = [], []
    for k, c in enumerate(centres):
        latent = rng.normal(loc=c, scale=0.7, size=(N_PER_CLUSTER, 2))
        observed = latent @ loadings.T
        observed += rng.normal(loc=0.0, scale=0.35, size=observed.shape)
        X_list.append(observed)
        y_list.append(np.full(N_PER_CLUSTER, k))
    return np.vstack(X_list), np.concatenate(y_list)


def pca(X, n_components=2):
    """PCA by eigendecomposition of the covariance matrix.

    Returns (projected, components, explained_variance_ratio, mean).
    Steps: centre -> covariance -> eigen-decomposition -> sort by
    eigenvalue (descending) -> keep the top eigenvectors.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # symmetric -> real eigenvalues
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components]
    projected = Xc @ components
    explained = eigvals / eigvals.sum()
    return projected, components, explained, mean


def reconstruction_mse(X, mean, components):
    """MSE of rebuilding X from its projection onto `components`."""
    Xc = X - mean
    rebuilt = (Xc @ components) @ components.T + mean
    return float(np.mean((X - rebuilt) ** 2))


def main():
    X, y = make_data()
    print(f"Data: {X.shape[0]} points, {X.shape[1]} features, "
          f"{len(np.unique(y))} true clusters")

    projected, components, explained, mean = pca(X, n_components=2)
    print(f"Explained variance ratio: {np.round(explained, 4)}")
    print(f"Variance kept by PC1+PC2: {explained[:2].sum():.4f}")
    print(f"Variance kept by PC1 alone: {explained[0]:.4f}")

    # Reconstruction error: 2 components vs all 4 (lossless baseline)
    _, components_all, _, _ = pca(X, n_components=4)
    mse_2 = reconstruction_mse(X, mean, components)
    mse_4 = reconstruction_mse(X, mean, components_all)
    print(f"Reconstruction MSE with 2 PCs: {mse_2:.4f}")
    print(f"Reconstruction MSE with 4 PCs: {mse_4:.4f} (should be ~0)")

    # How well do the 2 PCs recover the true clusters? (k-means-free check:
    # nearest-centroid accuracy in the 2-D projection, using true labels
    # only to build the centroids — a quick separability sanity check.)
    centroids = np.array(
        [projected[y == k].mean(axis=0) for k in np.unique(y)]
    )
    dists = np.linalg.norm(
        projected[:, None, :] - centroids[None, :, :], axis=2
    )
    pred = dists.argmin(axis=1)
    acc = float(np.mean(pred == y))
    print(f"Nearest-centroid accuracy in 2-D projection: {acc:.4f}")

    print("Top-2 principal components (rows = original features):")
    for i, row in enumerate(components):
        print(f"  feature {i}: PC1={row[0]:+.3f}  PC2={row[1]:+.3f}")

    # --- Plot 1: PC1 vs PC2 scatter, coloured by true cluster ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#e74c3c", "#2e86c1", "#27ae60"]
    for k in np.unique(y):
        ax.scatter(
            projected[y == k, 0], projected[y == k, 1],
            c=colors[k], s=28, alpha=0.75, edgecolors="white",
            linewidths=0.4, label=f"Cluster {k}",
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title("PCA projection: 4 features → 2 components")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("projection.svg", format="svg")
    plt.close(fig)

    # --- Plot 2: scree (explained variance per component) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(1, N_FEATURES + 1)
    ax.bar(xs, explained, color="#2e86c1", alpha=0.85)
    ax.plot(xs, np.cumsum(explained), "o-", color="#e74c3c",
            label="Cumulative")
    for x, v in zip(xs, explained):
        ax.text(x, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"PC{i}" for i in xs])
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree plot — how much variance each component keeps")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("scree.svg", format="svg")
    plt.close(fig)

    print("Saved projection.svg and scree.svg")


if __name__ == "__main__":
    main()
