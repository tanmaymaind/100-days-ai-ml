# Day 10 — Exploratory Data Analysis: Student Performance

Before fitting any model, look at the data. This is a first EDA pass
over 200 synthetic students — study hours, sleep hours, attendance,
previous exam score, and final exam score — answering four questions:
what does each variable look like, which variables move together, are
there outliers, and how do study habits differ across grade bands?

## The idea

EDA is the step most beginners skip and every practitioner does first.
A model trained on data you have never looked at is a model you cannot
trust: a skewed feature, a hidden outlier, or two variables that are
really the same thing will all show up later as a "mysterious" bad
result. So Day 10 has no model at all — just histograms, a
correlation matrix, an outlier check, and a box plot.

## The maths in brief

- **Pearson correlation** `r` measures linear association, from −1
  to +1. For two columns `x` and `y`,
  `r = cov(x, y) / (σx · σy)`. Roughly: |r| < 0.1 is nothing, 0.3 is
  moderate, 0.5+ is strong. It only sees *straight-line* relationships
  — a perfect U-shape scores near 0.
- **IQR outlier rule (Tukey):** anything below `Q1 − 1.5·IQR` or
  above `Q3 + 1.5·IQR` is flagged. The 1.5 is a convention, not a law
  of nature — it is a prompt to *look*, not a proof something is wrong.
- **Box plot:** the box spans Q1–Q3 with a line at the median, the
  whiskers reach the IQR fences, and points beyond them are drawn
  individually — the same outliers the rule above flags.

## The data

Synthetic, fixed seed (42), n = 200. The final score was generated as

```
final = 0.55·prev_score + 3.0·study_hours + 0.10·attendance + noise(σ=8)
```

clipped to 0–100. Sleep hours were deliberately **not** in the
formula — that is the point. If EDA is honest, sleep's correlation
with the final score should come out near zero, and any small value
it does show is pure chance.

## How to run

```bash
python3 eda_student_performance.py
```

Requires NumPy and matplotlib only. Prints the summary table, the
correlations, the outlier count and the grade-band breakdown, and
saves `distributions.svg`, `correlation_heatmap.svg`,
`scatter_study_vs_score.svg` and `boxplot_by_grade.svg`.

## Measured results

### Summary statistics (n = 200)

| Variable | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| study_hours | 4.94 | 4.90 | 1.76 | 0.74 | 10.83 |
| sleep_hours | 7.02 | 7.07 | 1.22 | 4.00 | 10.00 |
| attendance | 84.05 | 84.22 | 9.65 | 55.35 | 100.00 |
| prev_score | 64.56 | 65.14 | 12.21 | 34.22 | 95.59 |
| final_score | 58.50 | 58.49 | 12.13 | 24.92 | 95.04 |

Mean ≈ median for every column: nothing is badly skewed, and the
histograms in `distributions.svg` confirm it — all five look roughly
bell-shaped, as they should, since they were drawn from normals
(clipped at the edges).

### Correlations with the final score

| Variable | Pearson r |
|---|---:|
| prev_score | **+0.57** |
| study_hours | **+0.46** |
| attendance | +0.11 |
| sleep_hours | −0.17 |

Previous score is the strongest predictor (+0.57), study hours next
(+0.46) — exactly the two biggest terms in the generating formula.
Attendance is weak (+0.11): its coefficient was small (0.10) *and* its
spread is narrow (most students attend 75–95%), so it has little room
to move the score.

Sleep shows r = −0.17 even though it is **not in the formula at all**.
That is the lesson: with 200 points, a correlation of about ±0.14 is
already at the edge of "statistically significant", so a −0.17 can
appear by pure chance. A small r is not evidence of a real effect —
and it is definitely not evidence that sleeping less helps you pass.

The full matrix (see `correlation_heatmap.svg`) also shows the four
input variables are nearly uncorrelated with each other (|r| ≤ 0.09)
— they were generated independently, so there is no multicollinearity
to worry about before modelling.

### Outliers

The IQR fences for the final score are [27.94, 90.47]: **3 of 200**
students fall outside them. All three are real generated values, not
data errors — the rule flags them for a look, and a look says they are
just the tails of a normal distribution. Nothing to delete.

### Grade bands

| Grade | Range | Students | Mean study hours/day |
|---|---|---:|---:|
| A | ≥ 85 | 5 | 6.43 |
| B | 70–85 | 26 | 5.80 |
| C | 50–70 | 123 | 5.13 |
| D | < 50 | 46 | 3.77 |

The pattern is monotone: each higher grade band studied more on
average, from 3.77 hours/day (D) to 6.43 (A). The box plot
(`boxplot_by_grade.svg`) shows the same story with the spread — the
medians climb cleanly, though the boxes overlap, so study hours alone
do not determine the grade.

The scatter plot (`scatter_study_vs_score.svg`) with its trend line
makes the same point in one picture: `final ≈ 3.15·study_hours + 42.92`
— about +3 marks per extra hour of study a day (the same slope Day 1's
regression would have found).

## What I learned

- EDA first, model second: five minutes of histograms and a
  correlation matrix already told me which variables matter
  (prev_score, study_hours), which barely do (attendance), and which
  are noise (sleep) — before a single model was trained.
- Correlation ≠ causation, in both directions: sleep's −0.17 is a
  chance fluctuation in data where sleep has *no* effect by
  construction, and study hours' +0.46 is partly real (it is in the
  formula) and partly confounded with previous score.
- An outlier flag is a question, not an answer. The 3 flagged
  students are legitimate tails; deleting them would have been
  deleting real data.
- Mean ≈ median is a quick skew check worth doing on every column
  before trusting any model that assumes normality.
