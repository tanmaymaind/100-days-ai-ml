"""Plot accuracy vs Laplace smoothing alpha for the Day 3 classifier."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from naive_bayes import SPAM, HAM, build_vocab, vectorize, train, predict

rng = np.random.default_rng(42)
texts = SPAM + HAM
labels = np.array([1] * len(SPAM) + [0] * len(HAM))
perm = rng.permutation(len(texts))
train_idx, test_idx = perm[:30], perm[30:]

train_texts = [texts[i] for i in train_idx]
test_texts = [texts[i] for i in test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

word_to_idx = build_vocab(train_texts)
X_train = vectorize(train_texts, word_to_idx)
X_test = vectorize(test_texts, word_to_idx)

alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
accs = []
for a in alphas:
    classes, lp, ll = train(X_train, y_train, alpha=a)
    accs.append((predict(X_test, classes, lp, ll) == y_test).mean())

plt.figure(figsize=(7, 4.2))
plt.plot(alphas, accs, marker="o", color="#2563eb", linewidth=2)
plt.xscale("log")
plt.ylim(0.5, 1.05)
plt.xlabel("Laplace smoothing alpha (log scale)")
plt.ylabel("Test accuracy")
plt.title("Naive Bayes spam classifier: accuracy vs smoothing")
plt.grid(True, alpha=0.3)
for x, y in zip(alphas, accs):
    plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("accuracy_vs_alpha.svg", format="svg")
print("saved accuracy_vs_alpha.svg:", list(zip(alphas, [round(a, 2) for a in accs])))
