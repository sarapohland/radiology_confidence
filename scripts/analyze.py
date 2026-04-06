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


def plot_correlation_matrix(records: list, metric_fields: list,
                            score_type: str, save_path: str):
    """
    Pairwise Pearson correlation heatmap between all metric fields.

    Columns are ordered by hierarchical clustering so that related metrics
    appear together. The CRIMSON score is appended as a reference column
    so the reader can see which metrics correlate most with quality.
    """
    import pandas as pd

    all_fields = metric_fields + ["score"]
    rows = []
    for r in records:
        if r.get("score") is None:
            continue
        vals = {f: r.get(f) for f in all_fields}
        if any(v is None or (isinstance(v, float) and not np.isfinite(v))
               for v in vals.values()):
            continue
        rows.append(vals)

    if len(rows) < 5:
        print(f"  Not enough labeled records for correlation matrix ({len(rows)})")
        return

    df = pd.DataFrame(rows, columns=all_fields)
    col_labels = [metric_label(f) for f in metric_fields] + ["CRIMSON Score"]
    df.columns = col_labels

    corr_ordered = df.corr(method="pearson")

    n = len(col_labels)
    fig_size = max(8, n * 0.65)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr_ordered,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        annot=n <= 14,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )

    ax.set_title(f"{score_type.capitalize()} Metric Pairwise Correlations", pad=14)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    # Highlight the CRIMSON Score row/col with a border
    last = len(col_labels) - 1
    ax.add_patch(plt.Rectangle(
        (last, 0), 1, len(col_labels),
        fill=False, edgecolor="black", lw=2, clip_on=False
    ))
    ax.add_patch(plt.Rectangle(
        (0, last), len(col_labels), 1,
        fill=False, edgecolor="black", lw=2, clip_on=False
    ))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Correlation matrix saved to {save_path}")


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
# Selective accuracy (abstention / selective prediction)
# ---------------------------------------------------------------------------

def compute_selective_accuracy(records: list) -> dict:
    """
    Compute accuracy and coverage as a function of P(incorrect) threshold.

    For each threshold t, only records with prob_incorrect <= t are retained.
    Accuracy = fraction of retained labeled records with score >= 0 (correct).
    Coverage = fraction of all labeled records retained.

    Thresholds are swept over all unique prob_incorrect values in [0, 1],
    plus endpoints 0.0 and 1.0.

    Returns a dict with parallel lists:
        thresholds  : float list, ascending
        accuracies  : float list, accuracy on retained records at each threshold
        coverages   : float list, fraction of labeled records retained
        acc_changes : float list, accuracy - baseline_accuracy at each threshold
        n_retained  : int list, number of retained labeled records
    """
    labeled = [
        r for r in records
        if r.get("score") is not None and r.get("prob_incorrect") is not None
    ]
    if not labeled:
        return {}

    baseline_accuracy = sum(1 for r in labeled if r["score"] >= 0.0) / len(labeled)
    n_total = len(labeled)

    thresholds = sorted(set([0.0, 1.0] + [r["prob_incorrect"] for r in labeled]))

    results = {"thresholds": [], "accuracies": [], "coverages": [],
               "acc_changes": [], "n_retained": []}

    for t in thresholds:
        retained = [r for r in labeled if r["prob_incorrect"] <= t]
        if not retained:
            results["thresholds"].append(t)
            results["accuracies"].append(float("nan"))
            results["coverages"].append(0.0)
            results["acc_changes"].append(float("nan"))
            results["n_retained"].append(0)
            continue
        acc = sum(1 for r in retained if r["score"] >= 0.0) / len(retained)
        results["thresholds"].append(t)
        results["accuracies"].append(acc)
        results["coverages"].append(len(retained) / n_total)
        results["acc_changes"].append(acc - baseline_accuracy)
        results["n_retained"].append(len(retained))

    return {"baseline_accuracy": baseline_accuracy, "n_total": n_total, **results}


def _selective_accuracy_valid(sel: dict):
    """Return (thresholds, accuracies, acc_changes, coverages) with nan rows removed."""
    valid = [
        (t, a, ac, c)
        for t, a, ac, c in zip(
            sel["thresholds"], sel["accuracies"], sel["acc_changes"], sel["coverages"]
        )
        if a == a  # nan != nan
    ]
    if not valid:
        return None
    return zip(*valid)


def plot_selective_accuracy(sel: dict, score_type: str, save_path: str):
    """
    Plot raw accuracy and coverage as a function of P(incorrect) threshold.

    Primary y-axis : accuracy on retained records, with a dashed baseline reference.
    Secondary y-axis: coverage (fraction of labeled records retained).
    """
    unpacked = _selective_accuracy_valid(sel)
    if unpacked is None:
        return
    thresholds_v, accs_v, _, coverages_v = unpacked
    thresholds_v, accs_v, coverages_v = list(thresholds_v), list(accs_v), list(coverages_v)

    baseline    = sel["baseline_accuracy"]
    color_acc   = "steelblue"
    color_cov   = "tomato"

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(thresholds_v, accs_v, color=color_acc, linewidth=2,
             label="Accuracy (retained records)")
    ax1.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                label=f"Baseline accuracy ({baseline:.3f})")
    ax1.set_xlabel("P(incorrect) threshold")
    ax1.set_ylabel("Accuracy", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(max(0, min(accs_v) - 0.05), 1.05)

    ax2 = ax1.twinx()
    ax2.plot(thresholds_v, coverages_v, color=color_cov, linewidth=2,
             linestyle="--", label="Coverage (fraction retained)")
    ax2.set_ylabel("Coverage", color=color_cov)
    ax2.tick_params(axis="y", labelcolor=color_cov)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title(f"Selective Accuracy — {score_type} (prob_incorrect threshold)")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_selective_accuracy_change(sel: dict, score_type: str, save_path: str):
    """
    Plot accuracy change from baseline and coverage as a function of P(incorrect) threshold.

    Primary y-axis : accuracy change (accuracy on retained data − baseline accuracy).
    Secondary y-axis: coverage (fraction of labeled records retained).
    """
    unpacked = _selective_accuracy_valid(sel)
    if unpacked is None:
        return
    thresholds_v, _, acc_changes_v, coverages_v = unpacked
    thresholds_v, acc_changes_v, coverages_v = (
        list(thresholds_v), list(acc_changes_v), list(coverages_v)
    )

    baseline  = sel["baseline_accuracy"]
    color_acc = "steelblue"
    color_cov = "tomato"

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(thresholds_v, acc_changes_v, color=color_acc, linewidth=2,
             label=f"Accuracy change (baseline = {baseline:.3f})")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("P(incorrect) threshold")
    ax1.set_ylabel("Accuracy change from baseline", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xlim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(thresholds_v, coverages_v, color=color_cov, linewidth=2,
             linestyle="--", label="Coverage (fraction retained)")
    ax2.set_ylabel("Coverage", color=color_cov)
    ax2.tick_params(axis="y", labelcolor=color_cov)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title(f"Selective Accuracy Change — {score_type} (prob_incorrect threshold)")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_accuracy_targeted_rows(sel: dict,
                                   accuracy_breakpoints: list = None) -> list:
    """
    For each target accuracy level, find the highest coverage achievable
    while maintaining accuracy >= target.

    Returns a list of dicts with the same schema as print_selective_accuracy_table
    rows, but keyed by target_accuracy instead of target_coverage.
    """
    if accuracy_breakpoints is None:
        accuracy_breakpoints = [0.90, 0.95, 0.975, 0.99]

    thresholds  = sel["thresholds"]
    accuracies  = sel["accuracies"]
    acc_changes = sel["acc_changes"]
    coverages   = sel["coverages"]
    n_retained  = sel["n_retained"]

    rows = []
    for target_acc in accuracy_breakpoints:
        # Find the highest coverage where accuracy >= target (most permissive threshold)
        candidates = [
            (i, cov) for i, (cov, acc) in enumerate(zip(coverages, accuracies))
            if acc == acc and acc >= target_acc  # acc == acc filters nan
        ]
        if not candidates:
            continue
        idx = max(candidates, key=lambda x: x[1])[0]
        rows.append({
            "target_accuracy": target_acc,
            "threshold": round(thresholds[idx], 4),
            "actual_coverage": round(coverages[idx], 4),
            "n_retained": n_retained[idx],
            "accuracy": round(accuracies[idx], 4),
            "acc_change": round(acc_changes[idx], 4),
        })
    return rows


def print_selective_accuracy_table(sel: dict, score_type: str,
                                   coverage_breakpoints: list = None):
    """
    Print a table of accuracy change and coverage at key coverage breakpoints.
    """
    if coverage_breakpoints is None:
        coverage_breakpoints = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    thresholds  = sel["thresholds"]
    accuracies  = sel["accuracies"]
    acc_changes = sel["acc_changes"]
    coverages   = sel["coverages"]
    n_retained  = sel["n_retained"]
    baseline    = sel["baseline_accuracy"]
    n_total     = sel["n_total"]

    # For each breakpoint, find the row closest to that coverage level
    rows = []
    for target_cov in coverage_breakpoints:
        # Find threshold that gives coverage closest to (but not exceeding) target
        candidates = [
            (i, cov) for i, cov in enumerate(coverages)
            if cov <= target_cov + 1e-9 and accuracies[i] == accuracies[i]
        ]
        if not candidates:
            continue
        idx = max(candidates, key=lambda x: x[1])[0]
        rows.append({
            "target_coverage": target_cov,
            "threshold": round(thresholds[idx], 4),
            "actual_coverage": round(coverages[idx], 4),
            "n_retained": n_retained[idx],
            "accuracy": round(accuracies[idx], 4),
            "acc_change": round(acc_changes[idx], 4),
        })

    if not rows:
        return rows

    headers = ["Target Cov", "Threshold", "Actual Cov", "N Retained",
               "Accuracy", "Δ Accuracy"]
    keys    = ["target_coverage", "threshold", "actual_coverage",
               "n_retained", "accuracy", "acc_change"]

    def fmt(row, key):
        v = row[key]
        if key == "n_retained":
            return str(v)
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    col_widths = [max(len(h), max(len(fmt(r, k)) for r in rows))
                  for h, k in zip(headers, keys)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hrow = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    print(f"\n{score_type.upper()} — SELECTIVE ACCURACY  "
          f"(baseline accuracy = {baseline:.4f}, n = {n_total})")
    print(sep)
    print(hrow)
    print(sep)
    for row in rows:
        vals = [fmt(row, k) for k in keys]
        print("| " + " | ".join(v.ljust(w) for v, w in zip(vals, col_widths)) + " |")
    print(sep)

    return rows


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

    # Selective accuracy analysis (only when prob_incorrect is present)
    has_prob_incorrect = any(r.get("prob_incorrect") is not None for r in records)
    if has_prob_incorrect:
        sel = compute_selective_accuracy(records)
        if sel:
            sel_plot_path = os.path.join(type_plots_dir, "selective_accuracy.png")
            plot_selective_accuracy(sel, score_type, sel_plot_path)
            print(f"Selective accuracy plot saved to {sel_plot_path}")

            sel_change_plot_path = os.path.join(type_plots_dir, "selective_accuracy_change.png")
            plot_selective_accuracy_change(sel, score_type, sel_change_plot_path)
            print(f"Selective accuracy change plot saved to {sel_change_plot_path}")

            sel_csv_path = results_dir / f"{score_type}_selective_accuracy.csv"
            sel_rows = print_selective_accuracy_table(sel, score_type)
            acc_rows = compute_accuracy_targeted_rows(sel)
            if sel_rows or acc_rows:
                os.makedirs(str(results_dir), exist_ok=True)
                # Unified schema: target_coverage, target_accuracy, threshold,
                # actual_coverage, n_retained, accuracy, acc_change
                fieldnames = ["target_coverage", "target_accuracy", "threshold",
                              "actual_coverage", "n_retained", "accuracy", "acc_change"]
                cov_unified = [
                    {"target_coverage": r["target_coverage"], "target_accuracy": "",
                     **{k: r[k] for k in ("threshold", "actual_coverage",
                                          "n_retained", "accuracy", "acc_change")}}
                    for r in sel_rows
                ]
                acc_unified = [
                    {"target_coverage": "", "target_accuracy": r["target_accuracy"],
                     **{k: r[k] for k in ("threshold", "actual_coverage",
                                          "n_retained", "accuracy", "acc_change")}}
                    for r in acc_rows
                ]
                with open(sel_csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(cov_unified + acc_unified)
                print(f"Selective accuracy table saved to {sel_csv_path}")

    print_summary_table(summary_rows, score_type)

    # Correlation matrix
    corr_path = os.path.join(type_plots_dir, "correlation_matrix.png")
    plot_correlation_matrix(records, metric_fields, score_type, corr_path)


if __name__ == "__main__":
    main()
