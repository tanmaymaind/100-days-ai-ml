"""
Day 6 — Hypothesis Testing from Scratch: A/B Testing (NumPy only)

The idea
--------
A shop changes its checkout button and wants to know: did the change
actually help, or is the difference just noise? Hypothesis testing turns
that question into a number — the p-value — the probability of seeing a
difference this big *if the change did nothing at all*.

The maths in brief
-------------------
Null hypothesis H0: the two groups have the same mean (the change did
nothing). Alternative H1: the means differ.

Observed effect:
    diff = mean(B) - mean(A)

1. Welch's t-statistic (no assumption of equal variances):

       t = diff / sqrt(sA^2/nA + sB^2/nB)

   With n = 200 per group the t-distribution is practically normal, so
   the two-sided p-value comes from the standard normal CDF, which we
   compute from math.erf — no scipy, no tables:

       Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))

2. Permutation test — the from-scratch way to get a p-value with no
   distribution at all: shuffle the group labels 10,000 times, recompute
   the difference each time, and count how often |shuffled diff| is at
   least as big as the observed one. That fraction *is* the p-value.

3. Effect size (Cohen's d) — because "significant" is not the same as
   "big":

       d = diff / pooled_std

4. 95% confidence interval for the true difference:

       diff +/- 1.96 * SE,   SE = sqrt(sA^2/nA + sB^2/nB)

The data
--------
Synthetic, fixed seed (42) so the run is reproducible: time (seconds)
200 users take to check out. Group A (old button) ~ N(45, 10),
Group B (new button) ~ N(41, 10) — the new button really is 4 s faster,
so we expect a small p-value. A second, A/A check splits group A in half
at random: there the true effect is zero, so the p-value should be large
— the test must not cry wolf when nothing changed.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
N_PER_GROUP = 200
N_PERMUTATIONS = 10_000
ALPHA = 0.05

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
group_a = rng.normal(loc=45.0, scale=10.0, size=N_PER_GROUP)  # old button
group_b = rng.normal(loc=41.0, scale=10.0, size=N_PER_GROUP)  # new button

# ---------------------------------------------------------------- helpers
def normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def welch_t(a: np.ndarray, b: np.ndarray):
    """Return (t_stat, standard_error) for two independent samples."""
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (b.mean() - a.mean()) / se, se


def permutation_pvalue(a: np.ndarray, b: np.ndarray,
                       n_perm: int, rng: np.random.Generator):
    """Two-sided permutation p-value for the difference of means."""
    observed = abs(b.mean() - a.mean())
    pooled = np.concatenate([a, b])
    count = 0
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[len(a):].mean() - pooled[:len(a)].mean()
        perm_diffs[i] = diff
        if abs(diff) >= observed:
            count += 1
    return count / n_perm, perm_diffs


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-standard-deviation effect size."""
    pooled_var = ((len(a) - 1) * a.var(ddof=1)
                  + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return (b.mean() - a.mean()) / math.sqrt(pooled_var)


# ---------------------------------------------------------------- A/B test
mean_a, mean_b = group_a.mean(), group_b.mean()
diff = mean_b - mean_a
t_stat, se = welch_t(group_a, group_b)
p_z = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
p_perm, perm_diffs = permutation_pvalue(group_a, group_b,
                                        N_PERMUTATIONS, rng)
d = cohens_d(group_a, group_b)
ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

print(f"Group A (old button): n={len(group_a)}, "
      f"mean={mean_a:.2f}s, std={group_a.std(ddof=1):.2f}s")
print(f"Group B (new button): n={len(group_b)}, "
      f"mean={mean_b:.2f}s, std={group_b.std(ddof=1):.2f}s")
print(f"Observed difference (B - A): {diff:.2f}s")
print(f"Welch's t = {t_stat:.3f}, z-test p-value = {p_z:.5f}")
print(f"Permutation p-value ({N_PERMUTATIONS:,} shuffles): {p_perm:.5f}")
print(f"Cohen's d = {d:.3f}")
print(f"95% CI for the true difference: [{ci_low:.2f}, {ci_high:.2f}]s")
print(f"Significant at alpha={ALPHA}? "
      f"{'YES' if p_perm < ALPHA else 'NO'} "
      f"(both p-values agree: {'YES' if (p_z < ALPHA) == (p_perm < ALPHA) else 'NO'})")

# ---------------------------------------------------------------- A/A check
# Split group A in half at random: true effect is zero, so a good test
# should usually NOT reject here.
rng_aa = np.random.default_rng(SEED + 1)
half = rng_aa.permutation(group_a)
a1, a2 = half[:100], half[100:]
t_aa, _ = welch_t(a1, a2)
p_aa = 2.0 * (1.0 - normal_cdf(abs(t_aa)))
print(f"\nA/A sanity check (same group, split in half): "
      f"diff={a2.mean() - a1.mean():.2f}s, p-value={p_aa:.3f} "
      f"-> {'correctly NOT significant' if p_aa >= ALPHA else 'FALSE ALARM!'}")

# ---------------------------------------------------------------- plots
plt.rcParams.update({"font.size": 11})

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(group_a, bins=25, alpha=0.6, label=f"A (old): mean {mean_a:.1f}s")
ax.hist(group_b, bins=25, alpha=0.6, label=f"B (new): mean {mean_b:.1f}s")
ax.axvline(mean_a, color="C0", linestyle="--")
ax.axvline(mean_b, color="C1", linestyle="--")
ax.set_xlabel("Checkout time (seconds)")
ax.set_ylabel("Users")
ax.set_title("Checkout time: old button (A) vs new button (B)")
ax.legend()
fig.tight_layout()
fig.savefig("group_distributions.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(perm_diffs, bins=50, color="grey", alpha=0.7,
        label="Null distribution (10,000 shuffles)")
ax.axvline(diff, color="crimson", linewidth=2,
           label=f"Observed difference: {diff:.2f}s")
ax.axvline(-diff, color="crimson", linewidth=2, linestyle="--")
ax.set_xlabel("Difference of means, B - A (seconds)")
ax.set_ylabel("Permutation count")
ax.set_title(f"Permutation test: p = {p_perm:.4f} "
             f"(fraction of shuffles beyond the red lines)")
ax.legend()
fig.tight_layout()
fig.savefig("permutation_test.svg")
plt.close(fig)

print("\nPlots saved: group_distributions.svg, permutation_test.svg")
