"""
Day 9 — Multi-Layer Perceptron (MLP) from Scratch
=================================================

A tiny feed-forward neural network built with NumPy only: one hidden layer,
trained with full-batch gradient descent and backpropagation, on the
two-moons dataset (synthetic, fixed seed).

Architecture:  2 inputs -> 8 tanh hidden units -> 1 sigmoid output
Loss:          binary cross-entropy
Optimizer:     plain gradient descent, lr = 0.5, 2000 epochs

Run:  python3 mlp.py
Plots: decision_boundary.svg, loss_curve.svg
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)


def make_moons(n_samples=400, noise=0.15):
    """Two interleaving half-circles, the classic non-linear toy problem."""
    n_half = n_samples // 2
    # Upper moon: angles 0..pi
    t0 = np.linspace(0, np.pi, n_half)
    x0 = np.column_stack([np.cos(t0), np.sin(t0)])
    # Lower moon: angles pi..2pi, shifted right and down
    t1 = np.linspace(np.pi, 2 * np.pi, n_half)
    x1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])
    X = np.vstack([x0, x1]) + RNG.normal(0, noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    # Shuffle so train/test split is fair
    idx = RNG.permutation(n_samples)
    return X[idx], y[idx]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier-style init: keep activations in a sane range at start
        self.W1 = rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1          # (n, hidden)
        self.a1 = tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2    # (n, 1)
        self.a2 = sigmoid(self.z2)
        return self.a2

    @staticmethod
    def loss(y, y_hat):
        eps = 1e-12
        y = np.asarray(y).ravel()
        y_hat = np.clip(y_hat.ravel(), eps, 1 - eps)
        return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def backward(self, X, y, y_hat, lr):
        n = X.shape[0]
        y = y.reshape(-1, 1)
        # Output layer: dL/dz2 = (y_hat - y) / n  (BCE + sigmoid cancel out)
        dz2 = (y_hat - y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        # Hidden layer: tanh' = 1 - tanh^2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        # Gradient descent step
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def fit(self, X, y, epochs=2000, lr=0.5, log_every=200):
        history = []
        for epoch in range(epochs + 1):
            y_hat = self.forward(X)
            history.append(self.loss(y, y_hat))
            self.backward(X, y, y_hat, lr)
            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch:5d}  loss {history[-1]:.4f}")
        return history

    def predict(self, X):
        return (self.forward(X).ravel() > 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


def plot_decision_boundary(model, X, y, path):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = model.forward(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11),
                cmap="RdBu", alpha=0.8)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#d73027", s=18,
               edgecolors="white", linewidths=0.4, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#2166ac", s=18,
               edgecolors="white", linewidths=0.4, label="class 1")
    ax.set_title("MLP decision boundary (2-8-1, tanh hidden, sigmoid output)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#2166ac")
    ax.set_title("Binary cross-entropy loss vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    X, y = make_moons(n_samples=400, noise=0.15)
    X_train, y_train = X[:320], y[:320]
    X_test, y_test = X[320:], y[320:]

    model = MLP(n_in=2, n_hidden=8, n_out=1)
    history = model.fit(X_train, y_train, epochs=2000, lr=0.5)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy:  {test_acc:.4f}")
    print(f"final loss:     {history[-1]:.4f}")

    plot_decision_boundary(model, X, y, "decision_boundary.svg")
    plot_loss_curve(history, "loss_curve.svg")

    # Hidden-size sweep — the "how wide is wide enough?" lesson.
    print("\nhidden units | train acc | test acc")
    sweep = []
    for h in [1, 2, 4, 8, 16, 32]:
        m = MLP(2, h, 1)
        m.fit(X_train, y_train, epochs=2000, lr=0.5, log_every=None)
        tr, te = m.accuracy(X_train, y_train), m.accuracy(X_test, y_test)
        sweep.append((h, tr, te))
        print(f"{h:12d} | {tr:.4f}    | {te:.4f}")

    with open("results.txt", "w") as f:
        f.write(f"final_loss={history[-1]:.4f}\n")
        f.write(f"train_acc={train_acc:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
        for h, tr, te in sweep:
            f.write(f"hidden={h} train={tr:.4f} test={te:.4f}\n")


if __name__ == "__main__":
    main()
