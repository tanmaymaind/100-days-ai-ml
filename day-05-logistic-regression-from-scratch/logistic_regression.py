"""
Day 5 — Logistic Regression from Scratch (NumPy only, no sklearn)

The idea
--------
Linear regression predicts a continuous number, but many questions are
yes/no: will this student pass? Is this email spam? Is this tumour
malignant? Logistic regression answers yes/no questions by predicting a
*probability* between 0 and 1, then thresholding it at 0.5.

The maths in brief
-------------------
For an input x = (x1, x2) we first compute a linear score

    z = w1*x1 + w2*x2 + b

and squash it through the sigmoid function

    p = sigmoid(z) = 1 / (1 + exp(-z))

so p is always in (0, 1) and reads as P(y = 1 | x).

We fit w and b by minimising the binary cross-entropy (log loss) over
n training examples:

    L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )

Its gradients have the same neat form as linear regression:

    dL/dw = (1/n) * X^T (p - y),   dL/db = (1/n) * sum(p - y)

so plain gradient descent works: w -= lr * dL/dw.

The data
---------
Two synthetic features, fixed seed (42) so the run is reproducible:
imagine a student with "hours studied" and "hours slept". Two Gaussian
blobs — one for students who passed, one for students who failed — with
a little overlap so the problem is not trivially separable. 200 examples,
80/20 train/test split.

What this script does
---------------------
1. Generates the data and splits it into train/test.
2. Trains logistic regression with gradient descent (500 epochs).
3. Reports train/test accuracy and the confusion matrix.
4. Saves two plots: the learned decision boundary (decision_boundary.svg)
   and the log-loss curve (loss_curve.svg).

Run:  python3 logistic_regression.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
N_SAMPLES = 200
TEST_FRACTION = 0.20
LEARNING_RATE = 0.1
EPOCHS = 500


def sigmoid(z):
    """Squash any real number into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def generate_data(seed=RNG_SEED, n=N_SAMPLES):
    """Two Gaussian blobs: pass (1) vs fail (0), in (hours studied, hours slept)."""
    rng = np.random.default_rng(seed)
    # Class 1 (passed): studied more, slept a bit more
    passed = rng.normal(loc=[7.0, 7.5], scale=[1.6, 1.2], size=(n // 2, 2))
    # Class 0 (failed): studied less, slept less
    failed = rng.normal(loc=[3.5, 6.0], scale=[1.6, 1.2], size=(n // 2, 2))
    X = np.vstack([passed, failed])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    # Shuffle so the classes are mixed
    idx = rng.permutation(n)
    return X[idx], y[idx]


def train_test_split(X, y, test_fraction=TEST_FRACTION, seed=RNG_SEED):
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def log_loss(y_true, y_pred):
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def train(X_train, y_train, lr=LEARNING_RATE, epochs=EPOCHS):
    """Gradient descent on log loss. Returns weights, bias, loss history."""
    n, d = X_train.shape
    rng = np.random.default_rng(RNG_SEED)
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    losses = []
    for _ in range(epochs):
        p = sigmoid(X_train @ w + b)
        losses.append(log_loss(y_train, p))
        error = p - y_train
        w -= lr * (X_train.T @ error) / n
        b -= lr * error.mean()
    return w, b, losses


def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)


def confusion_matrix(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def plot_decision_boundary(X, y, w, b, path):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = sigmoid(grid @ w + b).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, zz, levels=20, cmap="RdYlGn", alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", edgecolor="k",
               label="Failed (0)", s=40)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="green", edgecolor="k",
               label="Passed (1)", s=40)
    ax.set_xlabel("Hours studied")
    ax.set_ylabel("Hours slept")
    ax.set_title("Logistic regression — decision boundary (black line = p = 0.5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_loss_curve(losses, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color="#2563eb")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title("Training loss over 500 epochs of gradient descent")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Feature standardisation (fit on train only, applied to both)
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    w, b, losses = train(X_train_s, y_train)
    train_acc = (predict(X_train_s, w, b) == y_train).mean()
    test_acc = (predict(X_test_s, w, b) == y_test).mean()
    tp, tn, fp, fn = confusion_matrix(y_test, predict(X_test_s, w, b))

    print(f"Learned weights: w1 = {w[0]:.4f}, w2 = {w[1]:.4f}, b = {b:.4f}")
    print(f"Final train log loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
    print(f"Train accuracy: {train_acc * 100:.1f}%  ({len(y_train)} examples)")
    print(f"Test accuracy:  {test_acc * 100:.1f}%  ({len(y_test)} examples)")
    print(f"Test confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Test precision: {tp / (tp + fp):.3f}, recall: {tp / (tp + fn):.3f}")

    # Plot the boundary in the original (unstandardised) feature space
    w_orig = w / sigma
    b_orig = b - np.sum(w * mu / sigma)
    plot_decision_boundary(X, y, w_orig, b_orig, "decision_boundary.svg")
    plot_loss_curve(losses, "loss_curve.svg")
    print("Saved decision_boundary.svg and loss_curve.svg")


if __name__ == "__main__":
    main()
