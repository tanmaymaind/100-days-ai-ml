"""
Day 11 — Random Forest Classifier from Scratch (NumPy only, no sklearn).

Idea
----
Day 8's decision tree is a high-variance learner: a small change in the
training data can grow a very different tree, and a deep tree memorises
noise. A random forest fixes that with bagging + random feature subsets:

1. Grow many trees (here 25), each on a bootstrap sample — n points drawn
   WITH replacement from the training set, so each tree sees a slightly
   different dataset (~63% unique points; the rest are "out of bag").
2. At every split, each tree may only consider a random subset of the
   features (sqrt of them). This decorrelates the trees: without it every
   tree would split on the same strongest feature first and make the
   same mistakes together.
3. Predict by majority vote across the trees.

Why it works: averaging many noisy, roughly-unbiased, decorrelated
predictors cancels the variance while keeping the bias low. The variance
of the average of B trees with pairwise correlation rho is

    Var = rho * sigma^2 + (1 - rho) * sigma^2 / B

so even with B -> infinity the variance bottoms out at rho * sigma^2 —
that is why step 2 (decorrelation) matters as much as step 1.

A free bonus: out-of-bag (OOB) error. Each training point is OOB for
about a third of the trees, so voting only those trees gives an honest
test-set-like accuracy WITHOUT holding out any data.

Data
----
Synthetic, fixed seed (42): 400 points on two interleaving moons
(noise 0.15) — the same data as Day 9's MLP, so the results are directly
comparable. 80/20 train/test split (320 / 80).

Outputs
-------
- Prints single-tree vs forest train/test accuracy and the OOB accuracy.
- decision_boundary.svg — forest boundary (smooth) vs single-tree
  boundary (jagged) on the test points.
- accuracy_vs_trees.svg — test accuracy as the forest grows 1..100 trees.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- data
def make_moons(n=400, noise=0.15):
    """Two interleaving half-circles (same recipe as Day 9)."""
    n_out = n // 2
    n_in = n - n_out
    t_out = RNG.uniform(0, np.pi, n_out)
    t_in = RNG.uniform(0, np.pi, n_in)
    x_out = np.c_[np.cos(t_out), np.sin(t_out)]
    x_in = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
    X = np.vstack([x_out, x_in])
    y = np.concatenate([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    X += RNG.normal(0, noise, X.shape)
    return X, y


# ---------------------------------------------------------------- tree
def gini(y):
    """Gini impurity of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - float(np.sum(p ** 2))


class Node:
    def __init__(self, *, prediction, feature=None, threshold=None,
                 left=None, right=None, depth=0, n_samples=0):
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.left is None


def best_split(X, y, features, rng):
    """Best (feature, threshold, gain) among `features`, or None.

    Candidate thresholds are capped at 25 evenly spaced midpoints per
    feature, so deep trees on bootstrap samples stay fast.
    """
    parent_gini = gini(y)
    best = None
    n = len(y)
    for feature in features:
        values = np.unique(X[:, feature])
        if len(values) < 2:
            continue
        midpoints = (values[:-1] + values[1:]) / 2.0
        if len(midpoints) > 25:  # thin out evenly, keep the endpoints
            idx = np.linspace(0, len(midpoints) - 1, 25).round().astype(int)
            midpoints = midpoints[np.unique(idx)]
        for t in midpoints:
            mask = X[:, feature] <= t
            if mask.sum() == 0 or mask.sum() == n:
                continue
            weighted = (mask.sum() * gini(y[mask])
                        + (~mask).sum() * gini(y[~mask])) / n
            gain = parent_gini - weighted
            if best is None or gain > best[2]:
                best = (feature, float(t), float(gain))
    return best


def build_tree(X, y, rng, max_depth=6, min_samples_split=5,
               max_features="sqrt", depth=0):
    values, counts = np.unique(y, return_counts=True)
    prediction = int(values[np.argmax(counts)])
    node = Node(prediction=prediction, depth=depth, n_samples=len(y))
    if depth >= max_depth or len(y) < min_samples_split or gini(y) == 0.0:
        return node
    n_feat = X.shape[1]
    if max_features == "sqrt":
        k = max(1, int(np.sqrt(n_feat)))
    elif max_features == "all":
        k = n_feat
    else:
        k = int(max_features)
    features = rng.choice(n_feat, size=min(k, n_feat), replace=False)
    split = best_split(X, y, features, rng)
    if split is None or split[2] <= 0:
        return node
    feature, threshold, _ = split
    mask = X[:, feature] <= threshold
    node.feature, node.threshold = feature, threshold
    node.left = build_tree(X[mask], y[mask], rng, max_depth,
                           min_samples_split, max_features, depth + 1)
    node.right = build_tree(X[~mask], y[~mask], rng, max_depth,
                            min_samples_split, max_features, depth + 1)
    return node


def predict_one(node, x):
    while not node.is_leaf:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


def predict_tree(node, X):
    return np.array([predict_one(node, x) for x in X])


def count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


# ---------------------------------------------------------------- forest
class RandomForest:
    def __init__(self, n_trees=25, max_depth=6, min_samples_split=5,
                 max_features="sqrt", seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.oob_indices = []  # per-tree indices NOT in its bootstrap sample

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.trees, self.oob_indices = [], []
        for _ in range(self.n_trees):
            boot_idx = rng.integers(0, n, size=n)
            oob_idx = np.setdiff1d(np.arange(n), np.unique(boot_idx))
            tree = build_tree(X[boot_idx], y[boot_idx], rng,
                              self.max_depth, self.min_samples_split,
                              self.max_features)
            self.trees.append(tree)
            self.oob_indices.append(oob_idx)
        return self

    def predict(self, X):
        votes = np.array([predict_tree(t, X) for t in self.trees])
        # majority vote per sample (binary classes 0/1)
        return (votes.mean(axis=0) >= 0.5).astype(int)

    def oob_accuracy(self, X, y):
        """Accuracy using, for each point, only the trees it was OOB for."""
        votes = np.zeros(len(y))
        counts = np.zeros(len(y))
        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if len(oob_idx) == 0:
                continue
            votes[oob_idx] += predict_tree(tree, X[oob_idx])
            counts[oob_idx] += 1
        scored = counts > 0
        pred = (votes[scored] / counts[scored] >= 0.5).astype(int)
        return float(np.mean(pred == y[scored])), int(scored.sum())


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------- main
def main():
    X, y = make_moons()
    idx = RNG.permutation(len(y))
    n_test = int(0.2 * len(y))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"train size: {len(y_train)}, test size: {len(y_test)}")

    # --- baseline: a single tree, same depth as the forest's trees ----
    rng = np.random.default_rng(42)
    tree = build_tree(X_train, y_train, rng, max_depth=6,
                      min_samples_split=5, max_features="all")
    tree_tr = accuracy(y_train, predict_tree(tree, X_train))
    tree_te = accuracy(y_test, predict_tree(tree, X_test))
    print(f"single tree (depth 6, {count_nodes(tree)} nodes): "
          f"train {tree_tr:.4f}, test {tree_te:.4f}")

    # --- the forest ----------------------------------------------------
    forest = RandomForest(n_trees=25, max_depth=6, min_samples_split=5,
                          max_features="sqrt", seed=42).fit(X_train, y_train)
    f_tr = accuracy(y_train, forest.predict(X_train))
    f_te = accuracy(y_test, forest.predict(X_test))
    oob_acc, oob_n = forest.oob_accuracy(X_train, y_train)
    print(f"random forest (25 trees, depth 6): train {f_tr:.4f}, "
          f"test {f_te:.4f}")
    print(f"OOB accuracy: {oob_acc:.4f} (scored on {oob_n} of "
          f"{len(y_train)} training points, no hold-out needed)")

    # --- plot 1: forest boundary vs single-tree boundary ---------------
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 160),
                         np.linspace(-1.5, 1.5, 160))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in [
            (axes[0], "tree", "Single tree (depth 6) — jagged"),
            (axes[1], "forest", "Random forest (25 trees) — smooth")]:
        if model == "tree":
            zz = predict_tree(tree, grid).reshape(xx.shape)
        else:
            zz = forest.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                    colors=["#4c72b0", "#dd8452"])
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                   c="#4c72b0", edgecolor="k", s=40, label="class 0 (test)")
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                   c="#dd8452", edgecolor="k", s=40, label="class 1 (test)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig("decision_boundary.svg")
    plt.close(fig)

    # --- plot 2: accuracy vs number of trees ---------------------------
    sizes = [1, 2, 5, 10, 25, 50, 100]
    te_accs = []
    for n in sizes:
        f = RandomForest(n_trees=n, max_depth=6, min_samples_split=5,
                         max_features="sqrt", seed=42).fit(X_train, y_train)
        te_accs.append(accuracy(y_test, f.predict(X_test)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, te_accs, "o-")
    ax.axhline(tree_te, color="gray", ls="--", lw=1,
               label=f"single tree: {tree_te:.4f}")
    ax.axhline(f_te, color="#dd8452", ls=":", lw=1,
               label=f"25-tree forest: {f_te:.4f}")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(sizes)
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Test accuracy vs forest size — gains flatten after ~10 trees")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("accuracy_vs_trees.svg")
    plt.close(fig)

    print("\nforest-size sweep:")
    for n, acc in zip(sizes, te_accs):
        print(f"  {n:3d} trees: test {acc:.4f}")
    print("\nplots saved: decision_boundary.svg, accuracy_vs_trees.svg")


if __name__ == "__main__":
    main()
