# Day 2 — K-Means Clustering from Scratch

**Topic:** Unsupervised learning · **Libraries:** NumPy + matplotlib only (no scikit-learn)

## The idea

K-Means finds groups in data that has no labels. You tell it how many
clusters (`k`) to look for, and it alternates two steps until the centroids
stop moving — this is **Lloyd's algorithm**:

1. **Assign** — every point joins its nearest centroid.
2. **Update** — each centroid moves to the mean of its points.

What it minimises is called **inertia**: the sum of squared distances from
each point to its own centroid. Lower inertia = tighter clusters.

Starting centroids matter: a bad random start can trap the algorithm in a
poor grouping. This version uses **k-means++ initialisation** — the first
centroid is random, then each next one is picked with probability
proportional to its squared distance from the centroids already chosen, so
the starting points are spread out.

## The maths, briefly

- Distance: Euclidean, `d(x, c) = ||x − c||`
- Update rule: `c_j = mean of all points assigned to cluster j`
- Objective (inertia): `Σ ||x_i − c_{label(i)}||²`
- Stops when the total centroid shift drops below `1e-4`, or after 100 iterations.

## The data

150 points in 3 synthetic 2D blobs (50 points each, Gaussian, σ = 0.8),
centred near (0, 0), (5, 4) and (0, 6). Fixed seed (`42`), so the run is
fully reproducible.

## How to run

```bash
python3 kmeans.py
```

Prints the result and saves `clusters.svg`.

## Results (measured from the actual run)

| Metric | Value |
|--------|-------|
| Points / k | 150 / 3 |
| Converged in | **3 iterations** |
| Inertia | **165.40** |
| Cluster sizes | 50, 50, 50 — exactly the true blob sizes |
| Centroids | (−0.03, −0.05), (5.01, 3.98), (−0.02, 5.92) |

The recovered centroids sit almost exactly on the true blob centres —
k-means++ plus well-separated blobs makes this an easy case, which is the
point: the algorithm works as the maths says it should.

**Elbow check** (inertia vs k, same data):

| k | Inertia |
|---|---------|
| 1 | 1937.41 |
| 2 | 892.24 |
| 3 | **165.40** |
| 4 | 145.67 |
| 5 | 129.03 |
| 6 | 110.00 |

The sharp drop from k = 2 to k = 3, then the flattening, is the classic
"elbow" — it confirms k = 3 is the right number of clusters here.

## What I learned

- K-Means is really just two lines of logic repeated: assign, then average.
- The elbow method is a practical way to pick k when you don't know it.
- Good initialisation (k-means++) matters more than many iterations — this
  run converged in just 3 steps.

## Try it

Change `K = 3` to `K = 2` or `K = 5` in `kmeans.py` and re-run — watch the
inertia change and the elbow table shift.
