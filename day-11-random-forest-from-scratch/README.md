# Day 11 — Random Forest Classifier from Scratch

Many decision trees, each trained on a different bootstrap sample and
each allowed to see only a random subset of features at every split —
then a majority vote. That is the whole trick, and it turns Day 8's
jumpy, overfitting single tree into a smooth, stable classifier.

## The idea

A single deep tree has low bias but high variance: change a few training
points and you get a different tree. A random forest attacks the variance
two ways:

1. **Bagging (bootstrap aggregating).** Each tree trains on n points
   drawn *with replacement* from the n training points. On average a
   bootstrap sample contains only ~63% unique points — the rest are
   duplicates, and the ~37% left out are "out of bag" (OOB) for that tree.
2. **Random feature subsets.** At every split, a tree may only consider
   `sqrt(n_features)` randomly chosen features. Without this, every tree
   would split on the strongest feature first and all trees would make
   the same mistakes together — averaging them would help far less.

Prediction is a majority vote across the trees.

## The maths — why averaging helps, and its limit

If each tree has variance σ² and the trees have pairwise correlation ρ,
the variance of the average of B trees is:

```
Var(average) = ρ·σ² + (1 − ρ)·σ² / B
```

As B grows, the second term vanishes — but the first term, ρ·σ², never
does. Bagging alone (ρ stays high) hits a floor; the random feature
subsets exist to push ρ down. That is the whole point of a *random*
forest versus plain bagging.

**Out-of-bag error, free of charge.** Each training point is OOB for
about a third of the trees, so voting only those trees gives an honest
accuracy estimate without holding out any data — no separate validation
split needed.

## The data

Synthetic, fixed seed (42): 400 points on two interleaving moons
(noise 0.15) — the same data as Day 9's MLP, so the results are directly
comparable. 80/20 train/test split (320 / 80).

## How to run

```bash
python3 random_forest.py
```

Requires NumPy and matplotlib only. Prints the single-tree baseline,
the forest results and a forest-size sweep; saves
`decision_boundary.svg` and `accuracy_vs_trees.svg`.

## Measured results

Single tree (depth 6) vs forest (25 trees, depth 6, `max_features=sqrt`):

| Metric | Single tree | Random forest |
|---|---:|---:|
| Train accuracy | 0.9906 | 0.9938 |
| Test accuracy | 0.9750 | **1.0000** |
| OOB accuracy | — | **0.9719** (all 320 training points scored) |

The forest gets every test point right where the single tree misses 2 of
80 — and the OOB estimate (0.9719) lands close to the true test accuracy
without ever touching the test set. The boundary plot tells the same
story visually: the single tree's boundary is jagged, the forest's is
smooth (see `decision_boundary.svg`).

### Forest-size sweep — how many trees are enough?

| Trees | Test acc |
|---:|---:|
| 1 | 0.9750 |
| 2 | 0.9250 |
| 5 | 0.9750 |
| 10 | 0.9750 |
| 25 | **1.0000** |
| 50 | **1.0000** |
| 100 | **1.0000** |

Gains flatten after ~25 trees — exactly what the variance formula
predicts: once `(1 − ρ)·σ² / B` is small, more trees buy almost nothing.
The 2-tree dip is real, not a bug: with an even number of trees a vote
can tie, and ties break toward class 1.

### How it stacks up on the same moons data

| Model | Test accuracy |
|---|---:|
| Day 5 logistic regression (straight line) | could not bend around the moons |
| Day 9 MLP (2-8-1, backprop) | 0.9000 |
| Day 11 random forest (25 trees) | **1.0000** |

The forest wins here — not because trees are "better" than networks in
general, but because axis-aligned splits plus voting happen to fit this
small, noisy 2-D problem very well, and the MLP was trained with plain
full-batch gradient descent.

## What I learned

- Bagging reduces variance; random feature subsets reduce the
  correlation that limits bagging. Both halves are needed — the variance
  formula shows that with ρ high, averaging alone bottoms out at ρ·σ²,
  so a forest that always saw all features would gain less from voting.
- OOB error is the cheapest honest evaluation in ML: a built-in
  cross-validation that costs zero extra data and zero extra training.
- More trees never hurt (test accuracy is monotone-ish after the wobble
  at 2 trees), they just stop helping — so pick the smallest forest
  past the plateau, here ~25 trees.
- A 1.0000 test accuracy on 80 points is not a miracle: with noise 0.15
  the moons barely overlap, and 80 points leave little room for a miss.
  On a bigger test set the forest would sit just under 1.0 — the OOB
  figure (0.9719, measured on 320 points) is the more honest number.
