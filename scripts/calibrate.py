"""
Calibrate confidence scores from any scored JSONL using a multivariate 
logistic regression over all numeric metric fields.

Fits a single L2-regularized logistic regression that maps the full feature
vector of a method's confidence metrics to P(incorrect). Fitting and
evaluation use stratified k-fold cross-validation. The final model is 
fitted on all labeled data and saved for inference-time use.

This is intended as a simple baseline calibration step for each individual
method (logit, consistency, stability).

Outputs
-------
  - Calibrated JSONL: input records extended with a single `prob_incorrect`
    field (probabilities for unseen records)
  - PR and ROC summary printed to stdout (same metrics as analyze.py)
  - Fitted model saved to calibrators_path if provided

Usage
-----
    python scripts/calibrate.py \\
        --input_path outputs/logit_scores.jsonl \\
        --output_path outputs/logit_scores_calibrated.jsonl \\
        [--n_folds 5] \\
        [--calibrators_path results/logit_calibrator.pkl]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    load_jsonl,
    _METADATA_FIELDS,
    detect_score_type,
    get_metric_fields,
    compute_pr_metrics,
    compute_roc_metrics,
)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_feature_matrix(records: list, fields: list) -> tuple:
    """
    Build a feature matrix from the given metric fields.

    Records missing any field are dropped. Returns (X, ids) where X has shape
    (n_valid, n_fields) and ids is the corresponding list of record ids.
    """
    rows, ids = [], []
    for r in records:
        vals = [r.get(f) for f in fields]
        if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in vals):
            continue
        rows.append(vals)
        ids.append(r["id"])
    return np.array(rows, dtype=float), ids


def logistic_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """
    Fit a multivariate logistic regression with stratified k-fold CV.

    Returns
    -------
    probs_oof : np.ndarray
        Out-of-fold P(incorrect) estimates. Use for ECE/Brier/AUPRC/AUROC
        evaluation to avoid overfitting bias.
    clf_full : LogisticRegression
        Model fitted on all data for deployment and unlabeled records.
    """
    probs_oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        probs_oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    clf_full = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf_full.fit(X, y)

    return probs_oof, clf_full


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(score_type: str, n_labeled: int, n_incorrect: int,
                  n_features: int, pr: dict, roc: dict):
    label = f"{score_type} (calibrated, {n_features} features)"

    def _table(title, headers, keys, values):
        col_widths = [max(len(h), len(str(values[k]))) for h, k in zip(headers, keys)]
        sep  = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        hrow = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        vrow = "| " + " | ".join(
            (f"{values[k]:.4f}" if isinstance(values[k], float) else str(values[k])).ljust(w)
            for k, w in zip(keys, col_widths)
        ) + " |"
        print(f"\n{title}")
        print(sep); print(hrow); print(sep); print(vrow); print(sep)

    print(f"\n{score_type.upper()} CALIBRATION RESULTS  (cross-validated)")
    print(f"  Features: {n_features},  Labeled: {n_labeled},  "
          f"Incorrect: {n_incorrect},  Correct: {n_labeled - n_incorrect}")

    pr_values  = {"metric": label, **pr}
    roc_values = {"metric": label, **roc}

    _table(f"{score_type.upper()} — PRECISION-RECALL",
           ["Metric", "AUPRC", "P@90R", "P@95R", "P@99R"],
           ["metric", "auprc", "p_at_90r", "p_at_95r", "p_at_99r"],
           pr_values)

    _table(f"{score_type.upper()} — ROC",
           ["Metric", "AUROC", "Sens@90Spec", "Sens@95Spec", "Sens@99Spec"],
           ["metric", "auroc", "sens_at_90spec", "sens_at_95spec", "sens_at_99spec"],
           roc_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a method's confidence scores via multivariate logistic regression."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Scored JSONL file (e.g. outputs/logit_scores.jsonl). "
             "Must contain a numeric 'score' field and at least one metric field.",
    )
    parser.add_argument(
        "--output_path", default=None,
        help="Path to write the JSONL extended with a prob_incorrect field. "
             "Defaults to --input_path (in-place update).",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--calibrators_path", type=str, default=None,
        help="Optional path to save the fitted calibrator as a pickle file "
             "(e.g. results/logit_calibrator.pkl). "
             "Load with pickle.load() to apply at inference time; "
             "the pickle contains {'fields': [...], 'model': LogisticRegression}.",
    )
    args = parser.parse_args()

    output_path = args.output_path or args.input_path

    records = load_jsonl(args.input_path)
    print(f"Loaded {len(records)} records from {args.input_path}")

    score_type = detect_score_type(args.input_path)
    print(f"Score type: {score_type}")

    metric_fields = get_metric_fields(records)
    if not metric_fields:
        print("No numeric metric fields found — nothing to calibrate.")
        return
    print(f"  Features ({len(metric_fields)}): {metric_fields}")

    # Split labeled / unlabeled; build binary labels
    labeled   = [r for r in records if r.get("score") is not None]
    unlabeled = [r for r in records if r.get("score") is None]
    labels_map = {r["id"]: (1 if r["score"] < 0.0 else 0) for r in labeled}

    n_incorrect = sum(v for v in labels_map.values())
    n_correct   = len(labels_map) - n_incorrect
    print(f"  Labeled: {len(labeled)}  (incorrect={n_incorrect}, correct={n_correct})"
          f",  Unlabeled: {len(unlabeled)}")

    # Build feature matrix for labeled records (drop any with missing values)
    X_labeled, ids_labeled = build_feature_matrix(labeled, metric_fields)
    y = np.array([labels_map[rid] for rid in ids_labeled], dtype=int)

    n_dropped = len(labeled) - len(ids_labeled)
    if n_dropped:
        print(f"  Dropped {n_dropped} labeled records with missing metric values.")

    if len(set(y)) < 2:
        print("Only one class present in labeled data — cannot calibrate.")
        return
    if len(y) < args.n_folds * 2:
        print(f"Too few labeled samples ({len(y)}) for {args.n_folds}-fold CV.")
        return

    # Fit cross-validated logistic regression
    probs_oof, clf_full = logistic_cv(X_labeled, y, n_folds=args.n_folds)

    # Evaluate on OOF probabilities using the same metrics as analyze.py
    pr  = compute_pr_metrics(probs_oof, y)
    roc = compute_roc_metrics(probs_oof, y)

    # Build id -> prob_incorrect mapping
    oof_probs_map = dict(zip(ids_labeled, probs_oof))

    # Full-model probs for unlabeled records
    unlabeled_probs_map = {}
    X_unlabeled, ids_unlabeled = build_feature_matrix(unlabeled, metric_fields)
    if len(X_unlabeled) > 0:
        p_unlabeled = clf_full.predict_proba(X_unlabeled)[:, 1]
        unlabeled_probs_map = dict(zip(ids_unlabeled, p_unlabeled))

    # Write updated JSONL (adds prob_incorrect field to existing records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            out = dict(r)
            rid = r["id"]
            if rid in oof_probs_map:
                out["prob_incorrect"] = round(float(oof_probs_map[rid]), 6)
            elif rid in unlabeled_probs_map:
                out["prob_incorrect"] = round(float(unlabeled_probs_map[rid]), 6)
            f.write(json.dumps(out) + "\n")
    print(f"Written to {output_path}")

    # Save calibrator
    if args.calibrators_path:
        os.makedirs(os.path.dirname(args.calibrators_path) or ".", exist_ok=True)
        with open(args.calibrators_path, "wb") as f:
            pickle.dump({"fields": metric_fields, "model": clf_full}, f)
        print(f"Calibrator saved to {args.calibrators_path}")

    print_results(score_type, len(y), int(y.sum()), len(metric_fields), pr, roc)


if __name__ == "__main__":
    main()
