"""
Day 10 — Exploratory Data Analysis: Student Performance
========================================================

A first EDA pass over a small synthetic dataset of 200 students:
study hours, sleep hours, attendance, previous exam score, and final
exam score. No modelling yet — just look at the data first.

Questions answered here:
  1. What does each variable look like (centre, spread, shape)?
  2. Which variables move together (Pearson correlation)?
  3. Are there outliers, and how many (IQR rule)?
  4. How do study habits differ across grade bands?

Run:
    python3 eda_student_performance.py

Requires NumPy and matplotlib only. Saves four SVG plots:
  distributions.svg, correlation_heatmap.svg,
  scatter_study_vs_score.svg, boxplot_by_grade.svg
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N = 200


def make_data(n=N, seed=42):
    """Synthetic student data, fixed seed so results are reproducible."""
    rng = np.random.default_rng(seed)
    study_hours = np.clip(rng.normal(5.0, 2.0, n), 0, 12)
    sleep_hours = np.clip(rng.normal(7.0, 1.2, n), 4, 10)
    attendance = np.clip(rng.normal(85.0, 10.0, n), 40, 100)
    prev_score = np.clip(rng.normal(65.0, 12.0, n), 20, 100)
    # Final score: mostly previous score + study hours + a little
    # attendance; sleep is deliberately NOT in the formula, so its
    # correlation with the final score should come out near zero.
    noise = rng.normal(0.0, 8.0, n)
    final_score = np.clip(
        0.55 * prev_score + 3.0 * study_hours + 0.10 * attendance + noise,
        0,
        100,
    )
    return {
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "prev_score": prev_score,
        "final_score": final_score,
    }


def summary_stats(data):
    """Mean, median, std, min, max for every column."""
    rows = []
    for name, col in data.items():
        rows.append(
            (
                name,
                col.mean(),
                np.median(col),
                col.std(ddof=1),
                col.min(),
                col.max(),
            )
        )
    return rows


def correlation_matrix(data):
    """Pearson correlation matrix over the five columns."""
    names = list(data.keys())
    matrix = np.corrcoef(np.vstack([data[n] for n in names]))
    return names, matrix


def iqr_outliers(col):
    """Tukey IQR rule: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (col < lower) | (col > upper)
    return mask.sum(), lower, upper


def grade_band(score):
    """A >= 85, B 70-85, C 50-70, D < 50."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def plot_distributions(data):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (name, col) in zip(axes, data.items()):
        ax.hist(col, bins=15, edgecolor="black", alpha=0.8)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    axes[-1].axis("off")
    fig.suptitle("Distributions of the five variables (n = 200)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("distributions.svg", format="svg")
    plt.close(fig)


def plot_correlation_heatmap(names, matrix):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(matrix[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation matrix (Pearson r)")
    fig.tight_layout()
    fig.savefig("correlation_heatmap.svg", format="svg")
    plt.close(fig)


def plot_scatter(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    bands = np.array([grade_band(s) for s in data["final_score"]])
    colors = {"A": "green", "B": "steelblue", "C": "orange", "D": "red"}
    for band in ["A", "B", "C", "D"]:
        mask = bands == band
        ax.scatter(
            data["study_hours"][mask],
            data["final_score"][mask],
            c=colors[band],
            label=f"Grade {band} (n={mask.sum()})",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )
    # Least-squares trend line (same maths as Day 1).
    m, b = np.polyfit(data["study_hours"], data["final_score"], 1)
    xs = np.array([0, 12])
    ax.plot(xs, m * xs + b, "k--", label=f"trend: y = {m:.2f}x + {b:.2f}")
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Final exam score")
    ax.set_title("More study hours -> higher final score")
    ax.legend()
    fig.tight_layout()
    fig.savefig("scatter_study_vs_score.svg", format="svg")
    plt.close(fig)
    return m, b


def plot_boxplot_by_grade(data):
    bands = np.array([grade_band(s) for s in data["final_score"]])
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = [data["study_hours"][bands == band] for band in ["A", "B", "C", "D"]]
    ax.boxplot(groups, labels=["A", "B", "C", "D"])
    ax.set_xlabel("Grade band (final score)")
    ax.set_ylabel("Study hours per day")
    ax.set_title("Study hours by grade band — the median climbs with the grade")
    fig.tight_layout()
    fig.savefig("boxplot_by_grade.svg", format="svg")
    plt.close(fig)


def main():
    data = make_data()
    print(f"Students: {N}\n")

    print("Summary statistics")
    print(f"{'variable':<13}{'mean':>8}{'median':>8}{'std':>8}{'min':>8}{'max':>8}")
    for name, mean, med, std, lo, hi in summary_stats(data):
        print(f"{name:<13}{mean:8.2f}{med:8.2f}{std:8.2f}{lo:8.2f}{hi:8.2f}")

    names, corr = correlation_matrix(data)
    print("\nPearson correlation with final_score:")
    final_idx = names.index("final_score")
    for i, name in enumerate(names):
        if name != "final_score":
            print(f"  {name:<13} r = {corr[final_idx, i]:+.3f}")

    print("\nIQR outliers in final_score:")
    count, lower, upper = iqr_outliers(data["final_score"])
    print(f"  fences: [{lower:.2f}, {upper:.2f}]  outliers: {count} of {N}")

    bands = np.array([grade_band(s) for s in data["final_score"]])
    print("\nGrade bands (final score):")
    for band in ["A", "B", "C", "D"]:
        mask = bands == band
        mean_study = data["study_hours"][mask].mean() if mask.any() else float("nan")
        print(
            f"  Grade {band}: {mask.sum():3d} students, "
            f"mean study hours = {mean_study:.2f}"
        )

    plot_distributions(data)
    plot_correlation_heatmap(names, corr)
    m, b = plot_scatter(data)
    plot_boxplot_by_grade(data)
    print(f"\nTrend line (study hours -> final score): slope = {m:.2f}, "
          f"intercept = {b:.2f}")
    print("Plots saved: distributions.svg, correlation_heatmap.svg, "
          "scatter_study_vs_score.svg, boxplot_by_grade.svg")


if __name__ == "__main__":
    main()
