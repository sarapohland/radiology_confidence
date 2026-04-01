"""
Evaluate the predictive power of confidence metrics of a given type (e.g., logit,
self-consistency, stability).

Outputs
-------
  - Distribution plots (KDE) for each metric for correct vs incorrect subsets
  - Precision-recall curves for each metric (positive = incorrect)
  - ROC curves for each metric (positive = incorrect)
  - Summary CSV with AUPRC, Precision @ 90/95/99% Recall, AUROC, 
        Sensitivity @ 90/95/99% Specificity

The score type is auto-detected from the input filename stem, e.g.:
  logit_scores.jsonl    -> type=logit   -> plots/logit/,   results/*/logit_summary.csv

Usage
-----
    python scripts/analyze.py \\
        --input_path outputs/val/logit_scores.jsonl \\
        --plots_dir plots/
"""

import argparse
import csv
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    load_jsonl,
    _METADATA_FIELDS,
    detect_score_type,
    metric_label,
    get_metric_fields,
    compute_pr_metrics,
    compute_roc_metrics,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Directionality
# ---------------------------------------------------------------------------

_HIGHER_IS_WORSE = {
    "max_ent", "mean_ent", "var_ent", "p90_ent", "p95_ent", "topk_ent",
    "order_ent", "semantic_ent", "domain_ent",
    "var_lp",  # higher variance = more erratic confidence = less reliable
    "prob_incorrect"
}


def should_negate(field: str) -> bool:
    """
    Return True if the field should be negated before scoring incorrect samples.

    Higher-is-worse metrics (entropy variants + var_lp) already score incorrect
    samples high, so no negation is needed. All other log-prob metrics are
    higher-is-better (more confident), so they must be negated.
    """
    return field not in _HIGHER_IS_WORSE


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_distribution(
    perfect_vals: list,
    partial_vals: list,
    incorrect_vals: list,
    field: str,
    save_path: str,
):
    """KDE distribution plot for perfect, partial, and incorrect subsets."""
    label = metric_label(field)
    fig, ax = plt.subplots(figsize=(9, 5))

    def _clean(vals):
        return [v for v in vals if v is not None and np.isfinite(v)]

    p = _clean(perfect_vals)
    pa = _clean(partial_vals)
    i = _clean(incorrect_vals)

    def _stats(vals, name):
        if vals:
            return f"{name} (μ={np.mean(vals):.3f}, σ²={np.var(vals):.4f}, n={len(vals)})"
        return f"{name} (no data)"

    p_label  = _stats(p,  "Perfect (score=1.0)")
    pa_label = _stats(pa, "Partial (0≤score<1)")
    i_label  = _stats(i,  "Incorrect (score<0)")

    if len(p) >= 2:
        sns.kdeplot(p, ax=ax, label=p_label, color="steelblue", fill=True, alpha=0.4)
    elif p:
        ax.axvline(p[0], color="steelblue", label=p_label)

    if len(pa) >= 2:
        sns.kdeplot(pa, ax=ax, label=pa_label, color="goldenrod", fill=True, alpha=0.4)
    elif pa:
        ax.axvline(pa[0], color="goldenrod", label=pa_label)

    if len(i) >= 2:
        sns.kdeplot(i, ax=ax, label=i_label, color="tomato", fill=True, alpha=0.4)
    elif i:
        ax.axvline(i[0], color="tomato", label=i_label)

    ax.set_title(f"{label} Distribution")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(pr: dict, field: str, save_path: str):
    """Precision-recall curve plot."""
    label = metric_label(field)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr["recalls"], pr["precisions"], color="steelblue", linewidth=2,
            label=f"AUPRC = {pr['auprc']:.3f}")
    ax.axhline(pr["p_at_90r"], color="orange", linestyle="--", linewidth=1,
               label=f"P@90R = {pr['p_at_90r']:.3f}")
    ax.axhline(pr["p_at_95r"], color="tomato", linestyle="--", linewidth=1,
               label=f"P@95R = {pr['p_at_95r']:.3f}")
    ax.axhline(pr["p_at_99r"], color="purple", linestyle="--", linewidth=1,
               label=f"P@99R = {pr['p_at_99r']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall for {label}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(roc: dict, field: str, save_path: str):
    """ROC curve plot with operating points at 90/95/99% specificity."""
    label = metric_label(field)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(roc["fprs"], roc["tprs"], color="steelblue", linewidth=2,
            label=f"AUROC = {roc['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Random")

    for fpr_thresh, sens_val, color, spec_pct in [
        (0.10, roc["sens_at_90spec"], "orange",  "90%"),
        (0.05, roc["sens_at_95spec"], "tomato",  "95%"),
        (0.01, roc["sens_at_99spec"], "purple",  "99%"),
    ]:
        ax.axvline(fpr_thresh, color=color, linestyle="--", linewidth=1,
                   label=f"Sens@{spec_pct}Spec = {sens_val:.3f}")

    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve for {label}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def print_summary_table(rows: list, score_type: str):
    """Print PR and ROC summary tables to stdout."""
    if not rows:
        return

    def _print_table(title, headers, keys, fmt_fn):
        col_widths = [
            max(len(h), max(len(fmt_fn(r, k)) for r in rows))
            for h, k in zip(headers, keys)
        ]
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        print(f"\n{title}")
        print(sep)
        print(header_row)
        print(sep)
        for row in rows:
            vals = [fmt_fn(row, k) for k in keys]
            print("| " + " | ".join(v.ljust(w) for v, w in zip(vals, col_widths)) + " |")
        print(sep)

    def fmt(row, key):
        v = row.get(key, "")
        if key == "metric":
            return metric_label(v)
        if key in ("n_perfect", "n_partial", "n_incorrect", "n_total"):
            return str(v)
        return f"{v:.4f}"

    pr_headers = ["Metric", "n_perfect", "n_partial", "n_incorrect", "AUPRC", "P@90R", "P@95R", "P@99R"]
    pr_keys    = ["metric", "n_perfect", "n_partial", "n_incorrect", "auprc", "p_at_90r", "p_at_95r", "p_at_99r"]
    _print_table(f"{score_type.upper()} — PRECISION-RECALL", pr_headers, pr_keys, fmt)

    roc_headers = ["Metric", "AUROC", "Sens@90Spec", "Sens@95Spec", "Sens@99Spec"]
    roc_keys    = ["metric", "auroc", "sens_at_90spec", "sens_at_95spec", "sens_at_99spec"]
    _print_table(f"{score_type.upper()} — ROC", roc_headers, roc_keys, fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot distributions and PR curves for a confidence score file."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to a scored JSONL file (e.g. outputs/val/logit_scores.jsonl).",
    )
    parser.add_argument(
        "--plots_dir", default="plots/",
        help="Root directory for plots. Score-type subdirectory is created automatically.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input_path)
    print(f"Loaded {len(records)} records from {args.input_path}")

    score_type = detect_score_type(args.input_path)
    print(f"Score type: {score_type}")

    type_plots_dir = os.path.join(args.plots_dir, score_type)
    dist_dir = os.path.join(type_plots_dir, "distributions")
    pr_dir = os.path.join(type_plots_dir, "pr_curves")
    roc_dir = os.path.join(type_plots_dir, "roc_curves")

    results_dir = Path("results")
    summary_path = results_dir / f"{score_type}_summary.csv"

    # Split into three distributions by score
    def _score(r):
        return r.get("score")

    perfect   = [r for r in records if _score(r) == 1.0]
    partial   = [r for r in records if _score(r) is not None and 0.0 <= _score(r) < 1.0]
    incorrect = [r for r in records if _score(r) is not None and _score(r) < 0.0]
    unlabeled = [r for r in records if _score(r) is None]
    print(
        f"  Perfect (score=1.0): {len(perfect)}, "
        f"Partial (0≤score<1): {len(partial)}, "
        f"Incorrect (score<0): {len(incorrect)}, "
        f"Unlabeled: {len(unlabeled)}"
    )

    metric_fields = get_metric_fields(records)
    if not metric_fields:
        print("No numeric metric fields found — nothing to plot.")
        return

    print(f"  Metrics: {metric_fields}")

    # PR is binary: incorrect (score < 0) = positive, all others = negative
    can_compute_pr = len(incorrect) >= 1 and (len(perfect) + len(partial)) >= 1

    summary_rows = []

    for field in metric_fields:
        p_vals  = [r[field] for r in perfect   if r.get(field) is not None]
        pa_vals = [r[field] for r in partial   if r.get(field) is not None]
        i_vals  = [r[field] for r in incorrect if r.get(field) is not None]

        dist_path = os.path.join(dist_dir, f"{field}.png")
        plot_distribution(p_vals, pa_vals, i_vals, field, dist_path)

        if can_compute_pr:
            labeled = [
                (r[field], r["score"])
                for r in records
                if r.get("score") is not None and r.get(field) is not None
            ]
            # positive = incorrect (score < 0), negative = partial + perfect
            bin_labels = [1 if s < 0.0 else 0 for _, s in labeled]
            if len(labeled) >= 2 and len(set(bin_labels)) == 2:
                metric_scores = [s for s, _ in labeled]
                negate = should_negate(field)

                pr = compute_pr_metrics(metric_scores, bin_labels, negate)
                pr_path = os.path.join(pr_dir, f"{field}.png")
                plot_pr_curve(pr, field, pr_path)

                roc = compute_roc_metrics(metric_scores, bin_labels, negate)
                roc_path = os.path.join(roc_dir, f"{field}.png")
                plot_roc_curve(roc, field, roc_path)

                summary_rows.append({
                    "metric": field,
                    "n_total": len(labeled),
                    "n_perfect": sum(1 for _, s in labeled if s == 1.0),
                    "n_partial": sum(1 for _, s in labeled if 0.0 <= s < 1.0),
                    "n_incorrect": sum(1 for _, s in labeled if s < 0.0),
                    "auprc": round(pr["auprc"], 4),
                    "p_at_90r": round(pr["p_at_90r"], 4),
                    "p_at_95r": round(pr["p_at_95r"], 4),
                    "p_at_99r": round(pr["p_at_99r"], 4),
                    "auroc": round(roc["auroc"], 4),
                    "sens_at_90spec": round(roc["sens_at_90spec"], 4),
                    "sens_at_95spec": round(roc["sens_at_95spec"], 4),
                    "sens_at_99spec": round(roc["sens_at_99spec"], 4),
                })
            else:
                print(f"  Skipping PR/ROC for {field}: need both correct and incorrect classes.")
        else:
            print("  Skipping PR/ROC curves: need at least one incorrect and one correct record.")

    # Write summary CSV
    if summary_rows:
        os.makedirs(str(results_dir), exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary saved to {summary_path}")
    else:
        print("No summary rows to write.")

    print(f"Plots saved under {type_plots_dir}/ (distributions/, pr_curves/, roc_curves/)")

    print_summary_table(summary_rows, score_type)


if __name__ == "__main__":
    main()
