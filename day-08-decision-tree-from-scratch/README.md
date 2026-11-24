# Day 8 — Decision Tree Classifier from Scratch

A decision tree, built with NumPy only — no sklearn. It learns a set of
yes/no questions ("is x1 <= 0.57?") that split the data into purer groups
until each leaf is (mostly) one class.

## The idea

At every node the tree tries **every feature × every midpoint threshold**
and picks the split that reduces impurity the most. Impurity is measured
with the **Gini index**:

```
Gini(S) = 1 − Σ p_c²          (p_c = fraction of class c in node S)
```

- Gini = 0 → node is pure (all one class).
- Gini = 0.5 → a perfectly mixed binary node.

A split's quality is the **gain** = parent Gini − weighted average Gini
of the two children. Splitting stops when the max depth is reached, a
node has fewer than `min_samples_split` samples, or the node is pure.

## The data

Synthetic, fixed seed (42): 300 points in 2-D, class 1 when `x1 + x2 > 1.2`,
with 10% of labels flipped as noise. 80/20 train/test split (240 / 60).

## How to run

```bash
python3 decision_tree.py
```

Requires NumPy and matplotlib only.

## Measured results

Depth-4 tree (23 nodes, 12 leaves):

| Metric | Value |
|---|---|
| Train accuracy | **0.8958** |
| Test accuracy | **0.8667** |

The learned tree's first split is `x0 <= 0.571` — a reasonable
approximation of the true diagonal boundary `x1 + x2 = 1.2`, built from
axis-aligned cuts only (see `decision_boundary.svg`).

### Depth sweep — the overfitting lesson

| Max depth | Train acc | Test acc |
|---|---|---|
| 1 | 0.7792 | 0.7500 |
| 2 | 0.8625 | **0.8833** |
| 3 | 0.8833 | **0.8833** |
| 4 | 0.8958 | 0.8667 |
| 5 | 0.9042 | 0.8500 |
| 6 | 0.9208 | 0.8500 |
| 7 | 0.9208 | 0.8000 |
| 8 | 0.9250 | 0.8000 |

Train accuracy rises monotonically with depth, but test accuracy peaks
at depth 2–3 and then **falls** — the deep tree memorises the 10% label
noise instead of the real boundary. This is exactly why trees are
pruned, or combined into random forests (a later day).

## What I learned

- A tree can only make axis-aligned cuts, so a diagonal boundary needs
  many small splits to approximate — that is why the boundary plot looks
  like a staircase.
- Gini gain is a simple, fast splitting rule, but greedy splits can
  still overfit: depth is the main regulariser here.
- The noise in the data (flipped labels) puts a ceiling on accuracy —
  even a perfect tree cannot beat ~90% on this data.
