"""
Day 2 — K-Means Clustering from Scratch (NumPy only, no scikit-learn).

Idea: K-Means finds k groups in unlabelled data. It alternates two steps
(Lloyd's algorithm) until the centroids stop moving:
  1. Assign every point to its nearest centroid.
  2. Move each centroid to the mean of the points assigned to it.

The thing it minimises is called "inertia": the sum of squared distances
from each point to its own centroid. Lower inertia = tighter clusters.

Data: 3 synthetic blobs (150 points), generated with a fixed seed so the
run is reproducible. No real dataset needed — the point is the algorithm.

Run:  python3 kmeans.py
Output: prints the result + saves clusters.svg (scatter plot).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_PER_BLOB = 50
K = 3
MAX_ITERS = 100
TOL = 1e-4


def make_blobs(seed=SEED):
    """Three well-separated 2D Gaussian blobs, 50 points each."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 4.0], [0.0, 6.0]])
    blobs = [rng.normal(loc=c, scale=0.8, size=(N_PER_BLOB, 2)) for c in centres]
    return np.vstack(blobs)


def kmeans_plusplus_init(X, k, rng):
    """Pick the first centroid at random, then each next one with
    probability proportional to its squared distance to the nearest
    centroid chosen so far. Spreads the starting centroids out."""
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min(
            [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
        )
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(n, p=probs)])
    return np.array(centroids)


def kmeans(X, k, seed=SEED, max_iters=MAX_ITERS, tol=TOL):
    rng = np.random.default_rng(seed)
    centroids = kmeans_plusplus_init(X, k, rng)
    for it in range(1, max_iters + 1):
        # Step 1 — assign: nearest centroid for every point
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Step 2 — update: centroid = mean of its points
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    inertia = float(np.sum((X - centroids[labels]) ** 2))
    return labels, centroids, inertia, it


def elbow(X, k_values, seed=SEED):
    """Inertia for several k — the 'elbow' is where adding more clusters
    stops paying off. Used to sanity-check that k=3 is a sensible pick."""
    return {k: kmeans(X, k, seed=seed)[2] for k in k_values}


def plot(X, labels, centroids, path="clusters.svg"):
    colours = ["#e74c3c", "#2e86c1", "#27ae60"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for j in range(len(centroids)):
        pts = X[labels == j]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=colours[j],
                   label=f"Cluster {j + 1} ({len(pts)} pts)", alpha=0.8)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=220, c="black",
               marker="X", label="Centroids")
    ax.set_title("K-Means Clustering from Scratch (k = 3)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    X = make_blobs()
    labels, centroids, inertia, iters = kmeans(X, K)
    print(f"Points: {X.shape[0]}, k = {K}")
    print(f"Converged in {iters} iterations")
    print(f"Inertia (sum of squared distances): {inertia:.2f}")
    print("Cluster sizes:", [int(np.sum(labels == j)) for j in range(K)])
    print("Centroids:")
    for j, c in enumerate(centroids):
        print(f"  Cluster {j + 1}: ({c[0]:.2f}, {c[1]:.2f})")
    print("Elbow check (inertia vs k):")
    for k, iner in elbow(X, [1, 2, 3, 4, 5, 6]).items():
        print(f"  k = {k}: {iner:.2f}")
    plot(X, labels, centroids)
    print("Saved plot: clusters.svg")


if __name__ == "__main__":
    main()
