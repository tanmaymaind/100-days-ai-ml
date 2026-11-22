# Day 6 — Hypothesis Testing from Scratch: A/B Testing

## The idea

A shop changes its checkout button and wants to know: did the change
actually help, or is the difference just noise? Hypothesis testing turns
that question into a number — the **p-value**: the probability of seeing a
difference this big *if the change did nothing at all*.

## The maths in brief

- **Null hypothesis H₀:** the two groups have the same mean (the change
  did nothing). **Alternative H₁:** the means differ.
- **Welch's t-statistic** (no assumption of equal variances):

  `t = (mean(B) − mean(A)) / √(sA²/nA + sB²/nB)`

  With n = 200 per group the t-distribution is practically normal, so the
  two-sided p-value comes from the standard normal CDF, computed from
  `math.erf` — no scipy, no tables.
- **Permutation test** — the from-scratch way to get a p-value with no
  distribution at all: shuffle the group labels 10,000 times, recompute
  the difference each time, and count how often |shuffled diff| ≥
  |observed diff|. That fraction *is* the p-value.
- **Cohen's d** (effect size) — because "significant" is not the same as
  "big": `d = diff / pooled_std`.
- **95% confidence interval** for the true difference: `diff ± 1.96 · SE`.

## The data

Synthetic, fixed seed (42): checkout time in seconds for 200 users per
group. Group A (old button) ~ N(45, 10), Group B (new button) ~ N(41, 10)
— the new button really is 4 s faster.

## How to run

```bash
python3 ab_test.py
```

NumPy + matplotlib only. Produces `group_distributions.svg` and
`permutation_test.svg`.

## Results (measured, seed 42)

| Metric | Value |
|---|---|
| Group A mean (old button) | 44.70 s (std 8.82) |
| Group B mean (new button) | 41.20 s (std 10.20) |
| Observed difference (B − A) | −3.49 s |
| Welch's t | −3.666 |
| z-test p-value | 0.00025 |
| Permutation p-value (10,000 shuffles) | 0.00030 |
| Cohen's d | −0.367 (small-to-medium effect) |
| 95% CI for the true difference | [−5.36, −1.63] s |
| Significant at α = 0.05? | **Yes** — both methods agree |

The p-value says: if the new button did nothing, a difference of 3.49 s
would happen only about 3 times in 10,000 experiments. And the confidence
interval doesn't include zero, so the direction is clear too — the new
button is genuinely faster.

## Sanity check: the A/A test

Splitting group A in half at random (true effect = 0) gives a difference
of −0.21 s and p = 0.868 — correctly **not** significant. A test that
cried "significant" on identical groups would be useless; this one
doesn't.

## What I learned

- A p-value is a statement about the data under H₀, **not** the
  probability that H₀ is true.
- Two completely different methods (a normal-approximation z-test and a
  brute-force permutation test) landing on p ≈ 0.0003 is a good sign both
  are right.
- Significance ≠ importance: Cohen's d = 0.37 is a real but modest
  effect — worth shipping, not worth celebrating as a revolution.
- Always run an A/A test before trusting an A/B test: it catches broken
  randomisation and buggy metrics before they cost real decisions.
