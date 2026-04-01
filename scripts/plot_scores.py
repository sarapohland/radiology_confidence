"""
Plot the distribution of CRIMSON judge scores.

Produces:
  - plots/judge/score_distribution.png  — histogram + KDE with percentile markers
  - plots/judge/threshold_sensitivity.png — fraction passing at each threshold
  - plots/judge/error_breakdown.png       — mean per-study counts by error type
  - results/judge_summary.csv             — threshold table saved to disk

Usage:
    python scripts/plot_scores.py \
        --judge_scores_path outputs/judge_scores.jsonl \
        --plots_dir plots/judge/ \
        --output_path results/judge_summary.csv
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import load_jsonl

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Plot 1: Score distribution
# ---------------------------------------------------------------------------

def plot_score_distribution(scores: list[float], save_path: Path) -> None:
    percentiles = [10, 25, 50, 75, 90]
    pct_values = np.percentile(scores, percentiles)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(scores, bins=30, kde=True, ax=ax, color="steelblue", alpha=0.6)

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(percentiles)))
    for pct, val, color in zip(percentiles, pct_values, colors):
        ax.axvline(val, color=color, linestyle="--", linewidth=1.2,
                   label=f"P{pct} = {val:.3f}")

    ax.set_xlabel("CRIMSON score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of CRIMSON scores")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Threshold sensitivity
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(scores: list[float], save_path: Path) -> list[dict]:
    thresholds = np.linspace(-1.0, 1.0, 201)
    fractions = [np.mean(np.array(scores) >= t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, fractions, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Fraction of studies passing")
    ax.set_title("Fraction passing at each CRIMSON threshold")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Build summary rows for a set of candidate thresholds
    candidate_thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.6, 0.75, 0.9]
    rows = []
    for t in candidate_thresholds:
        frac = float(np.mean(np.array(scores) >= t))
        rows.append({"threshold": t, "fraction_passing": round(frac, 4),
                     "n_passing": int(np.sum(np.array(scores) >= t)),
                     "n_total": len(scores)})
    return rows


# ---------------------------------------------------------------------------
# Plot 3: Error type breakdown
# ---------------------------------------------------------------------------

ERROR_FIELDS = [
    "false_findings",
    "missing_findings",
    "attribute_errors",
    "location_errors",
    "severity_errors",
    "descriptor_errors",
    "measurement_errors",
    "certainty_errors",
    "unspecific_errors",
    "overinterpretation_errors",
    "temporal_errors",
]


def plot_error_breakdown(records: list[dict], save_path: Path) -> None:
    # Collect per-study error counts from question_scores
    counts: dict[str, list[float]] = {f: [] for f in ERROR_FIELDS}
    for rec in records:
        qs = rec.get("question_scores") or {}
        for field in ERROR_FIELDS:
            counts[field].append(float(qs.get(field, 0)))

    means = {f: float(np.mean(v)) for f, v in counts.items() if v}
    labels = [f.replace("_", " ") for f in means]
    values = list(means.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color="steelblue", alpha=0.75)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean count per study")
    ax.set_title("Mean per-study error counts by type")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_threshold_table(rows: list[dict]) -> None:
    header = f"{'Threshold':>10}  {'Passing':>8}  {'/ Total':>8}  {'Fraction':>10}"
    print("\n" + "=" * len(header))
    print("THRESHOLD SENSITIVITY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['threshold']:>10.2f}  {r['n_passing']:>8}  {r['n_total']:>8}  {r['fraction_passing']:>10.4f}")
    print("=" * len(header))


def save_summary_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "fraction_passing", "n_passing", "n_total"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot CRIMSON score distribution.")
    p.add_argument("--judge_scores_path", type=Path, default=Path("outputs/judge_scores.jsonl"))
    p.add_argument("--plots_dir", type=Path, default=Path("plots/judge/"))
    p.add_argument("--output_path", type=Path, default=Path("results/judge_summary.csv"))
    args = p.parse_args()

    records = load_jsonl(args.judge_scores_path)
    print(f"Loaded {len(records)} records from {args.judge_scores_path}")

    scores = [r["score"] for r in records if r.get("score") is not None]
    n_null = len(records) - len(scores)
    if n_null:
        print(f"  {n_null} records have null score (excluded from plots)")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Mean: {np.mean(scores):.4f}  Median: {np.median(scores):.4f}")

    plot_score_distribution(scores, args.plots_dir / "score_distribution.png")

    threshold_rows = plot_threshold_sensitivity(scores, args.plots_dir / "threshold_sensitivity.png")

    plot_error_breakdown(records, args.plots_dir / "error_breakdown.png")

    print_threshold_table(threshold_rows)
    save_summary_csv(threshold_rows, args.output_path)


if __name__ == "__main__":
    main()
