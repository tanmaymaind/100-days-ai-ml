"""Day 1 — Linear Regression from scratch (NumPy, gradient descent).

Fits y = m*x + b to synthetic data. No scikit-learn.
Run:  python linear_regression.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


def make_data(n=100, seed=42):
    """Synthetic data: y = 2.5x + 1.0 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = 2.5 * x + 1.0 + rng.normal(0, 1.0, n)
    return x, y


def train(x, y, lr=0.01, epochs=1000):
    """Gradient descent on mean squared error. Returns m, b, loss history."""
    m, b = 0.0, 0.0
    n = len(x)
    losses = []
    for _ in range(epochs):
        y_hat = m * x + b
        error = y_hat - y
        dm = (2 / n) * np.sum(error * x)
        db = (2 / n) * np.sum(error)
        m -= lr * dm
        b -= lr * db
        losses.append(np.mean(error**2))
    return m, b, losses


def metrics(x, y, m, b):
    y_hat = m * x + b
    mse = np.mean((y_hat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - np.sum((y - y_hat) ** 2) / ss_tot
    return mse, r2


def plot(x, y, m, b, losses, path="fit.svg"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(x, y, s=18, alpha=0.7, label="data")
    xs = np.linspace(0, 10, 100)
    ax1.plot(xs, m * xs + b, color="red", label=f"fit: y = {m:.2f}x + {b:.2f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Linear regression fit")
    ax1.legend()
    ax2.plot(losses, color="green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE loss")
    ax2.set_title("Loss falls with gradient descent")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    print(f"saved plot -> {path}")


if __name__ == "__main__":
    x, y = make_data()
    m, b, losses = train(x, y)
    mse, r2 = metrics(x, y, m, b)
    print(f"learned: m = {m:.4f} (true 2.5), b = {b:.4f} (true 1.0)")
    print(f"MSE = {mse:.4f}, R2 = {r2:.4f}")
    plot(x, y, m, b, losses)
