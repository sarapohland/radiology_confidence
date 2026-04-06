"""
Calibrate a combined set of features drawn from multiple signal sources
(logit, consistency, stability) using multivariate logistic regression.

Each source JSONL is joined by study ID. Only studies present in all provided
files and with no missing feature values are included. Fitting and evaluation
use stratified k-fold cross-validation. The final model is fitted on all
labeled data and saved for inference-time use.

By default, features are selected automatically via greedy forward selection:
starting from the single best feature (by CV AUPRC), each step adds the
candidate that produces the largest AUPRC gain. Selection stops when no
remaining feature improves AUPRC by more than --min_gain (default 0.001) or
when --max_features is reached. Pass --features to fix the feature set and
skip selection.

Outputs
-------
  - Calibrated JSONL: one record per study with selected features and a
    `prob_incorrect` field
  - Correlation heatmap across all candidate features (plots/multi/)
  - Feature selection results CSV (results/multi_feature_selection.csv)
  - Feature importance table (standardized logistic regression coefficients)
  - PR and ROC summary printed to stdout (same metrics as analyze.py)
  - Fitted model saved to calibrators_path if provided

Usage
-----
    python scripts/multi_calibrate.py \\
        --logit_path      outputs/logit_scores.jsonl \\
        --consistency_path outputs/consistency_scores.jsonl \\
        --stability_path  outputs/stability_scores.jsonl \\
        --output_path     outputs/multi_scores.jsonl \\
        [--features botk_lp domain_lp topk_ent ...]  # fix features, skip selection
        [--max_features 10] \\
        [--min_gain 0.001] \\
        [--n_folds 5] \\
        [--plots_dir plots/] \\
        [--calibrators_path results/multi_calibrator.pkl]
"""

import argparse
import csv
import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from calibrate import build_feature_matrix, logistic_cv, print_results
from utils import load_jsonl_by_id, compute_pr_metrics, compute_roc_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

_METADATA_FIELDS = {
    "id", "score", "question_scores", "processing_time",
    "response", "tokens", "log_probs", "entropies", "prob_incorrect", "passed",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def merge_sources(sources: list[dict]) -> list[dict]:
    """
    Join multiple {id -> record} dicts by study ID.

    Returns a flat list of merged records containing fields from all sources.
    Only studies present in every source are included. The 'score' field is
    taken from the first source that provides it.
    """
    ids = set(sources[0].keys())
    for src in sources[1:]:
        ids &= set(src.keys())

    merged = []
    for sid in sorted(ids):
        record = {}
        for src in sources:
            for k, v in src[sid].items():
                if k == "score" and "score" in record:
                    continue
                record[k] = v
        merged.append(record)
    return merged


def get_candidate_features(records: list) -> list:
    """Return all numeric non-metadata fields present in every record."""
    if not records:
        return []
    present = set()
    for r in records:
        for k, v in r.items():
            if k not in _METADATA_FIELDS and isinstance(v, (int, float)):
                present.add(k)
    return sorted(present)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlation_matrix(X: np.ndarray, features: list) -> np.ndarray:
    """Pearson correlation matrix over labeled records with no missing values."""
    return np.corrcoef(X.T)


def plot_correlation_heatmap(corr: np.ndarray, features: list, save_path: str):
    """Save a Pearson correlation heatmap for all candidate features."""
    n = len(features)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": max(6, 9 - n // 5)},
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        xticklabels=features, yticklabels=features,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (Pearson)")
    ax.tick_params(axis="x", rotation=45, labelsize=max(6, 9 - n // 6))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(6, 9 - n // 6))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Correlation heatmap saved to {save_path}")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _auprc_for_features(X_all: np.ndarray, y: np.ndarray,
                         indices: list, n_folds: int) -> float:
    """CV AUPRC for the subset of columns given by indices."""
    X_sub = X_all[:, indices]
    probs_oof, _ = logistic_cv(X_sub, y, n_folds=n_folds)
    return compute_pr_metrics(probs_oof, y)["auprc"]


def forward_feature_selection(
    X_all: np.ndarray,
    y: np.ndarray,
    features: list,
    n_folds: int = 5,
    max_features: int = None,
    min_gain: float = 0.001,
) -> tuple:
    """
    Greedy forward feature selection maximising CV AUPRC.

    At each step, evaluates every candidate feature individually (added to the
    current set) and picks the one with the highest AUPRC gain. Stops when no
    remaining feature improves AUPRC by more than min_gain or max_features is
    reached.

    Returns
    -------
    selected_features : list[str]   features in selection order
    selection_log     : list[dict]  one entry per step with keys:
                            step, feature_added, auprc, auprc_gain,
                            selected_so_far
    individual_auprcs : dict[str, float]  single-feature AUPRC for each feature
    """
    if max_features is None:
        max_features = len(features)

    feat_idx = {f: i for i, f in enumerate(features)}

    # Individual (single-feature) AUPRCs
    print("\nComputing individual feature AUPRCs...")
    individual_auprcs = {}
    for f in features:
        auprc = _auprc_for_features(X_all, y, [feat_idx[f]], n_folds)
        individual_auprcs[f] = auprc
        print(f"  {f}: {auprc:.4f}")

    selected_indices = []
    selected_features = []
    remaining = list(features)
    current_auprc = 0.0
    selection_log = []

    print("\nForward feature selection:")
    for step in range(1, max_features + 1):
        best_feat, best_auprc, best_gain = None, 0.0, -np.inf
        for candidate in remaining:
            trial_indices = selected_indices + [feat_idx[candidate]]
            auprc = _auprc_for_features(X_all, y, trial_indices, n_folds)
            gain = auprc - current_auprc
            if auprc > best_auprc:
                best_feat, best_auprc, best_gain = candidate, auprc, gain

        if best_feat is None or best_gain < min_gain:
            print(f"  Step {step}: no feature improves AUPRC by >{min_gain:.4f} — stopping.")
            break

        selected_features.append(best_feat)
        selected_indices.append(feat_idx[best_feat])
        remaining.remove(best_feat)
        current_auprc = best_auprc

        log_entry = {
            "step": step,
            "feature_added": best_feat,
            "auprc": round(best_auprc, 4),
            "auprc_gain": round(best_gain, 4),
            "selected_so_far": ", ".join(selected_features),
        }
        selection_log.append(log_entry)
        print(f"  Step {step}: +{best_feat}  AUPRC={best_auprc:.4f} "
              f"(gain={best_gain:+.4f})")

    return selected_features, selection_log, individual_auprcs


def save_selection_results(
    selection_log: list,
    individual_auprcs: dict,
    save_path: str,
):
    """Write the forward selection log and individual AUPRCs to a CSV."""
    # Build a combined table: individual AUPRC for every feature (step 0),
    # then one row per selection step.
    rows = []

    # Individual AUPRCs sorted descending
    for f, auprc in sorted(individual_auprcs.items(), key=lambda x: -x[1]):
        rows.append({
            "step": 0,
            "feature_added": f,
            "individual_auprc": round(auprc, 4),
            "combined_auprc": "",
            "auprc_gain": "",
            "selected_so_far": f,
        })

    for entry in selection_log:
        rows.append({
            "step": entry["step"],
            "feature_added": entry["feature_added"],
            "individual_auprc": round(individual_auprcs[entry["feature_added"]], 4),
            "combined_auprc": entry["auprc"],
            "auprc_gain": entry["auprc_gain"],
            "selected_so_far": entry["selected_so_far"],
        })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Feature selection results saved to {save_path}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(clf, features: list, X: np.ndarray):
    """
    Print standardized logistic regression coefficients as a feature importance
    proxy. Coefficients are scaled by the standard deviation of each feature so
    that magnitudes are comparable across features with different units/ranges.
    A positive coefficient means higher feature value → higher P(incorrect).
    """
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features
    scaled_coefs = clf.coef_[0] * stds

    print("\nFEATURE IMPORTANCE (standardized logistic regression coefficients)")
    print("  Positive = higher value → more likely incorrect")
    pairs = sorted(zip(features, scaled_coefs), key=lambda x: -abs(x[1]))
    col_w = max(len(f) for f in features)
    for feat, coef in pairs:
        bar = "█" * int(abs(coef) / max(abs(scaled_coefs)) * 20 + 0.5)
        direction = "+" if coef >= 0 else "-"
        print(f"  {feat:<{col_w}}  {direction}{abs(coef):.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-signal confidence calibration with forward feature selection."
    )
    parser.add_argument("--logit_path", required=True)
    parser.add_argument("--consistency_path", required=True)
    parser.add_argument("--stability_path", required=True)
    parser.add_argument(
        "--output_path", default="outputs/multi_scores.jsonl",
        help="Path to write the calibrated JSONL (default: outputs/multi_scores.jsonl).",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Fix the feature set and skip forward selection. "
             "If omitted, features are chosen automatically.",
    )
    parser.add_argument(
        "--max_features", type=int, default=None,
        help="Maximum number of features to select (default: no limit).",
    )
    parser.add_argument(
        "--min_gain", type=float, default=0.001,
        help="Minimum AUPRC gain required to add a feature (default: 0.001).",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--plots_dir", default="plots/",
        help="Root directory for plots (default: plots/).",
    )
    parser.add_argument(
        "--calibrators_path", type=str, default=None,
        help="Optional path to save the fitted calibrator as a pickle file.",
    )
    args = parser.parse_args()

    # ---- Load and merge ----
    logit       = load_jsonl_by_id(args.logit_path)
    consistency = load_jsonl_by_id(args.consistency_path)
    stability   = load_jsonl_by_id(args.stability_path)

    print(f"Loaded {len(logit)} logit, {len(consistency)} consistency, "
          f"{len(stability)} stability records")

    records = merge_sources([logit, consistency, stability])
    print(f"Studies present in all three sources: {len(records)}")

    candidate_features = get_candidate_features(records)
    print(f"Candidate features ({len(candidate_features)}): {candidate_features}")

    # ---- Labels ----
    labeled   = [r for r in records if r.get("score") is not None]
    unlabeled = [r for r in records if r.get("score") is None]
    labels_map = {r["id"]: (1 if r["score"] < 0.0 else 0) for r in labeled}

    n_incorrect = sum(v for v in labels_map.values())
    n_correct   = len(labels_map) - n_incorrect
    print(f"  Labeled: {len(labeled)}  (incorrect={n_incorrect}, correct={n_correct})"
          f",  Unlabeled: {len(unlabeled)}")

    if len(set(labels_map.values())) < 2:
        print("Only one class present — cannot calibrate.")
        return

    # ---- Build full feature matrix for all candidates ----
    X_all, ids_labeled = build_feature_matrix(labeled, candidate_features)
    y = np.array([labels_map[rid] for rid in ids_labeled], dtype=int)

    n_dropped = len(labeled) - len(ids_labeled)
    if n_dropped:
        print(f"  Dropped {n_dropped} labeled records with missing feature values.")
    if len(y) < args.n_folds * 2:
        print(f"Too few labeled samples ({len(y)}) for {args.n_folds}-fold CV.")
        return

    # ---- Correlation heatmap ----
    corr = compute_correlation_matrix(X_all, candidate_features)
    corr_path = os.path.join(args.plots_dir, "multi", "feature_correlation.png")
    plot_correlation_heatmap(corr, candidate_features, corr_path)

    # ---- Feature selection or fixed feature set ----
    selection_csv = Path("results") / "multi_feature_selection.csv"

    if args.features is not None:
        missing = [f for f in args.features if f not in candidate_features]
        if missing:
            print(f"ERROR: requested features not found: {missing}")
            sys.exit(1)
        selected_features = args.features
        print(f"\nUsing fixed feature set ({len(selected_features)}): {selected_features}")
        individual_auprcs = {}
        selection_log = []
    else:
        selected_features, selection_log, individual_auprcs = forward_feature_selection(
            X_all, y,
            features=candidate_features,
            n_folds=args.n_folds,
            max_features=args.max_features,
            min_gain=args.min_gain,
        )
        if not selected_features:
            print("No features selected — cannot calibrate.")
            return
        save_selection_results(selection_log, individual_auprcs, str(selection_csv))
        print(f"\nSelected features ({len(selected_features)}): {selected_features}")

    # ---- Fit final model on selected features ----
    feat_idx = {f: i for i, f in enumerate(candidate_features)}
    selected_indices = [feat_idx[f] for f in selected_features]
    X_selected = X_all[:, selected_indices]

    probs_oof, clf_full = logistic_cv(X_selected, y, n_folds=args.n_folds)
    pr  = compute_pr_metrics(probs_oof, y)
    roc = compute_roc_metrics(probs_oof, y)

    oof_probs_map = dict(zip(ids_labeled, probs_oof))

    # Full-model probs for unlabeled records
    unlabeled_probs_map = {}
    X_unlabeled, ids_unlabeled = build_feature_matrix(unlabeled, selected_features)
    if len(X_unlabeled) > 0:
        p_unlabeled = clf_full.predict_proba(X_unlabeled)[:, 1]
        unlabeled_probs_map = dict(zip(ids_unlabeled, p_unlabeled))

    # ---- Write output JSONL ----
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in records:
            rid = r["id"]
            out = {"id": rid, "score": r.get("score")}
            for feat in selected_features:
                out[feat] = r.get(feat)
            if rid in oof_probs_map:
                out["prob_incorrect"] = round(float(oof_probs_map[rid]), 6)
            elif rid in unlabeled_probs_map:
                out["prob_incorrect"] = round(float(unlabeled_probs_map[rid]), 6)
            f.write(json.dumps(out) + "\n")
    print(f"Written to {args.output_path}")

    # ---- Save calibrator ----
    if args.calibrators_path:
        os.makedirs(os.path.dirname(args.calibrators_path) or ".", exist_ok=True)
        with open(args.calibrators_path, "wb") as f:
            pickle.dump({"fields": selected_features, "model": clf_full}, f)
        print(f"Calibrator saved to {args.calibrators_path}")

    # ---- Reporting ----
    print_feature_importance(clf_full, selected_features, X_selected)
    print_results("multi", len(y), int(y.sum()), len(selected_features), pr, roc)


if __name__ == "__main__":
    main()
