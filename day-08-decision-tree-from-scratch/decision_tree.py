"""
Day 8 — Decision Tree Classifier from Scratch (NumPy only, no sklearn).

Idea
----
A decision tree learns a set of yes/no questions ("is feature x <= t?")
that split the data into purer and purer groups, until each leaf is
(mostly) one class. Splits are chosen by minimising Gini impurity:

    Gini(S) = 1 - sum over classes c of p_c^2

where p_c is the fraction of class c in node S. A split is good if the
weighted Gini of its two children is much lower than the parent's.

Data
----
Synthetic, fixed seed (42): 300 points in 2-D, two classes generated
from a simple rule with noise — class 1 if x1 + x2 > 1.2 (with 10%
label flips), otherwise class 0. 80/20 train/test split.

The tree
---------
- Recursive: at each node, try every feature and every midpoint between
  sorted unique values as a candidate threshold, pick the split with the
  lowest weighted Gini.
- Stop when max_depth is reached, a node has fewer than min_samples_split
  samples, or the node is pure.
- Leaf predicts the majority class.

Outputs
-------
- Prints train/test accuracy and the learned tree structure.
- decision_boundary.svg — test points coloured by true class, background
  shaded by the tree's prediction.
- accuracy_vs_depth.svg — train/test accuracy as max_depth grows 1..8.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- data
def make_data(n=300):
    x = RNG.uniform(0, 2, size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 1.2).astype(int)
    # 10% label noise so the boundary is not perfectly learnable
    flip = RNG.random(n) < 0.10
    y[flip] = 1 - y[flip]
    return x, y


# ---------------------------------------------------------------- tree
def gini(y):
    """Gini impurity of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - float(np.sum(p ** 2))


def best_split(X, y):
    """Return (feature, threshold, gain) of the best split, or None."""
    parent_gini = gini(y)
    best = None
    n = len(y)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / n
            gain = parent_gini - weighted
            if best is None or gain > best[2]:
                best = (feature, float(t), float(gain))
    return best


class Node:
    def __init__(self, *, prediction, feature=None, threshold=None,
                 left=None, right=None, depth=0, n_samples=0, gini=0.0):
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.n_samples = n_samples
        self.gini = gini

    @property
    def is_leaf(self):
        return self.left is None


def build_tree(X, y, max_depth=4, min_samples_split=10, depth=0):
    values, counts = np.unique(y, return_counts=True)
    prediction = int(values[np.argmax(counts)])
    node = Node(prediction=prediction, depth=depth,
                n_samples=len(y), gini=gini(y))
    if depth >= max_depth or len(y) < min_samples_split or node.gini == 0.0:
        return node
    split = best_split(X, y)
    if split is None or split[2] <= 0:
        return node
    feature, threshold, _ = split
    mask = X[:, feature] <= threshold
    node.feature, node.threshold = feature, threshold
    node.left = build_tree(X[mask], y[mask], max_depth, min_samples_split, depth + 1)
    node.right = build_tree(X[~mask], y[~mask], max_depth, min_samples_split, depth + 1)
    return node


def predict_one(node, x):
    while not node.is_leaf:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


def predict(node, X):
    return np.array([predict_one(node, x) for x in X])


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def print_tree(node, indent=""):
    if node.is_leaf:
        print(f"{indent}leaf: predict {node.prediction} "
              f"(n={node.n_samples}, gini={node.gini:.3f})")
    else:
        print(f"{indent}if x{node.feature} <= {node.threshold:.3f} "
              f"(n={node.n_samples}, gini={node.gini:.3f}):")
        print_tree(node.left, indent + "  ")
        print(f"{indent}else:")
        print_tree(node.right, indent + "  ")


def count_nodes(node):
    if node.is_leaf:
        return 1, 1  # total, leaves
    lt, ll = count_nodes(node.left)
    rt, rl = count_nodes(node.right)
    return 1 + lt + rt, ll + rl


# ---------------------------------------------------------------- main
def main():
    X, y = make_data()
    idx = RNG.permutation(len(y))
    n_test = int(0.2 * len(y))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    tree = build_tree(X_train, y_train, max_depth=4, min_samples_split=10)
    total, leaves = count_nodes(tree)
    train_acc = accuracy(y_train, predict(tree, X_train))
    test_acc = accuracy(y_test, predict(tree, X_test))
    print(f"train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"tree nodes: {total} ({leaves} leaves), max depth 4")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy:  {test_acc:.4f}")
    print("\nLearned tree:")
    print_tree(tree)

    # --- plot 1: decision boundary -------------------------------------
    xx, yy = np.meshgrid(np.linspace(0, 2, 200), np.linspace(0, 2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predict(tree, grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                colors=["#4c72b0", "#dd8452"])
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               c="#4c72b0", edgecolor="k", s=45, label="class 0 (test)")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               c="#dd8452", edgecolor="k", s=45, label="class 1 (test)")
    ax.plot([0, 1.2], [1.2, 0], "k--", lw=1.2, label="true boundary x1+x2=1.2")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision Tree (depth 4) — learned boundary vs true boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig("decision_boundary.svg")
    plt.close(fig)

    # --- plot 2: accuracy vs depth -------------------------------------
    depths = range(1, 9)
    tr_accs, te_accs = [], []
    for d in depths:
        t = build_tree(X_train, y_train, max_depth=d, min_samples_split=10)
        tr_accs.append(accuracy(y_train, predict(t, X_train)))
        te_accs.append(accuracy(y_test, predict(t, X_test)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(depths), tr_accs, "o-", label="train accuracy")
    ax.plot(list(depths), te_accs, "s-", label="test accuracy")
    ax.set_xlabel("max depth")
    ax.set_ylabel("accuracy")
    ax.set_xticks(list(depths))
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Accuracy vs tree depth — test accuracy peaks, then overfits")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("accuracy_vs_depth.svg")
    plt.close(fig)

    print("\ndepth sweep:")
    for d, tr, te in zip(depths, tr_accs, te_accs):
        print(f"  depth {d}: train {tr:.4f}, test {te:.4f}")
    print("\nplots saved: decision_boundary.svg, accuracy_vs_depth.svg")


if __name__ == "__main__":
    main()
