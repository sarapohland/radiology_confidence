"""
Compute logit-based confidence metrics from per-token log probabilities 
and entropies.

Each record in the output JSONL contains the study id, judge correctness 
label, and one scalar per requested metric. The --metrics flag selects 
which metrics to compute; by default all metrics are computed.

Metrics
-------
Log-probability metrics (higher = more confident):
  min_lp        minimum token log probability
  mean_lp       mean token log probability
  var_lp        variance of token log probabilities
  p5_lp         5th-percentile token log probability
  p10_lp        10th-percentile token log probability
  botk_lp       mean of the k lowest token log probabilities
  order_lp      mean log probability weighted by token position (earlier = higher weight)
  semantic_lp   mean log probability weighted by semantic significance (content words over filler)
  domain_lp     mean log probability weighted by RadLex domain importance

Entropy metrics (lower = more confident):
  max_ent       maximum token entropy
  mean_ent      mean token entropy
  var_ent       variance of token entropies
  p90_ent       90th-percentile token entropy
  p95_ent       95th-percentile token entropy
  topk_ent      mean of the k highest token entropies
  order_ent     mean entropy weighted by token position
  semantic_ent  mean entropy weighted by semantic significance
  domain_ent    mean entropy weighted by RadLex domain importance

Usage
-----
    python scripts/logits.py \\
        --predictions_path outputs/val/predictions.jsonl \\
        --judge_scores_path outputs/val/judge_scores.jsonl \\
        --output_path outputs/val/logit_scores.jsonl \\
        [--lexicon_path lexicon/RadLex.owl] \\
        [--no-findings_only] \\
        [--k 5] \\
        [--metrics min_lp mean_lp ...]

Helper Functions
----------------
    load_radlex_lexicon(path) -> frozenset[str]
    findings_slice(tokens) -> (int, int)
    order_weights(n) -> list[float]
    semantic_weights(tokens, importance_scores=None) -> list[float]
    domain_lexical_weights(tokens, radlex_words=None) -> list[float]
    compute_record_metrics(record, selected, radlex_words=None,
                           findings_only=False, k=5) -> dict
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from text_utils import is_filler_token
from utils import findings_slice


# ---------------------------------------------------------------------------
# Metrics for utilizing token logits
# ---------------------------------------------------------------------------

ALL_METRICS = [
    "min_lp", "mean_lp", "var_lp", "p5_lp", "p10_lp", "botk_lp",
    "order_lp", "semantic_lp", "domain_lp",
    "max_ent", "mean_ent", "var_ent", "p90_ent", "p95_ent", "topk_ent",
    "order_ent", "semantic_ent", "domain_ent",
]


# ---------------------------------------------------------------------------
# RadLex lexicon loading
# ---------------------------------------------------------------------------

def load_radlex_lexicon(path: str) -> frozenset:
    """
    Parse RadLex OWL and return a frozenset of lowercased single-word medical terms.

    Extracts all English rdfs:label values, splits multi-word phrases into
    individual words, and filters out filler words and very short tokens.
    The resulting set can be passed to domain_lexical_weights to weight tokens
    that appear in the RadLex ontology higher than general English words.

    Parameters
    ----------
    path : str
        Path to RadLex.owl.

    Returns
    -------
    frozenset[str]
        Lowercased single-word terms that appear in RadLex labels.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    labels = re.findall(r'<rdfs:label xml:lang="en">([^<]+)</rdfs:label>', content)

    words = set()
    for label in labels:
        label = html.unescape(label)
        for word in re.split(r'[\s\-/]+', label.lower()):
            word = re.sub(r'[^a-z]', '', word)  # keep only lowercase letters
            if 4 <= len(word) <= 20 and not is_filler_token(word):
                words.add(word)

    return frozenset(words)


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

def order_weights(n: int) -> list:
    """
    Linearly decreasing weights by token position (earlier = more weight).
    Rationale: first tokens in a report set the diagnostic context.
    """
    if n == 0:
        return []
    weights = [n - i for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def semantic_weights(
    tokens: list,
    importance_scores: Optional[list] = None,
) -> list:
    """
    Weight tokens by their semantic significance.

    If importance scores based on report knowledge are available, use 
    them. Otherwise, assign 2x weight to non-filler tokens.
    """
    n = len(tokens)
    if n == 0:
        return []
    
    # Importance scores available
    elif importance_scores is not None and len(importance_scores) == n:
        arr = np.array(importance_scores, dtype=float)
        total = arr.sum()
        if total > 0:
            return (arr / total).tolist()
        
    # Importance scores not available
    else:
        weights = [1.0 if not is_filler_token(t) else 0.5 for t in tokens]
        total = sum(weights)
        return [w / total for w in weights]


def domain_lexical_weights(
    tokens: list,
    radlex_words: Optional[frozenset] = None,
) -> list:
    """
    Weight tokens by domain importance using the RadLex ontology.

    Parameters
    ----------
    tokens : list[str]
        Token strings from model generation.
    radlex_words : frozenset[str], optional
        Set of lowercased RadLex single-word terms from load_radlex_lexicon().
        If None, falls back to a small hardcoded set of common radiology terms.

    Returns
    -------
    list[float]
        Normalized per-token weights where RadLex terms receive weight 1.0,
        non-RadLex content words receive 0.5, and filler/punctuation receives 0.0.
    """
    FALLBACK = {
        "opacity", "consolidation", "effusion", "pneumothorax", "pneumonia",
        "atelectasis", "edema", "cardiomegaly", "nodule", "mass", "fracture",
        "infiltrate", "pleural", "enlarged", "bilateral", "unilateral",
        "normal", "clear", "no", "without",
    }
    lexicon = radlex_words if radlex_words is not None else FALLBACK

    weights = []
    for token in tokens:
        clean = token.strip().lower().strip(".,;:()'\"")
        if is_filler_token(token):
            weights.append(0.0)
        elif clean in lexicon:
            weights.append(1.0)
        else:
            weights.append(0.5)

    total = sum(weights)
    if total == 0:
        n = len(tokens)
        return [1.0 / n] * n if n > 0 else []
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# Per-record metric computation
# ---------------------------------------------------------------------------

def compute_record_metrics(
    record: dict,
    selected: list,
    radlex_words: Optional[frozenset] = None,
    findings_only: bool = True,
    k: int = 5,
) -> dict:
    """
    Compute the selected logit-based metrics for a single record.

    Parameters
    ----------
    record : dict
        A record from predictions.jsonl with 'tokens', 'log_probs', 'entropies'.
    selected : list[str]
        Subset of ALL_METRICS to compute.
    radlex_words : frozenset[str], optional
        RadLex word set from load_radlex_lexicon(). Required for domain_lp / domain_ent.
    findings_only : bool
        If True (default), restrict all metrics to the FINDINGS section tokens only,
        excluding the section header and everything from IMPRESSION onward.
    k : int
        Number of tokens used by botk_lp and topk_ent (default 5).

    Returns
    -------
    dict mapping metric name to scalar float (or None if data missing).
    """
    tokens = record.get("tokens", [])
    log_probs = record.get("log_probs", [])
    entropies = record.get("entropies", [])

    if findings_only and tokens:
        s, e = findings_slice(tokens)
        tokens = tokens[s:e]
        log_probs = log_probs[s:e] if log_probs else []
        entropies = entropies[s:e] if entropies else []

    n = len(tokens)
    result = {}

    lp_metrics = [
        "min_lp", "mean_lp", "var_lp", "p5_lp", "p10_lp", "botk_lp",
        "order_lp", "semantic_lp", "domain_lp",
    ]
    ent_metrics = [
        "max_ent", "mean_ent", "var_ent", "p90_ent", "p95_ent", "topk_ent",
        "order_ent", "semantic_ent", "domain_ent",
    ]
    need_lp = any(m in selected for m in lp_metrics)
    need_ent = any(m in selected for m in ent_metrics)

    if need_lp and log_probs:
        arr_lp = np.array(log_probs, dtype=float)
        if "min_lp" in selected:
            result["min_lp"] = float(arr_lp.min())
        if "mean_lp" in selected:
            result["mean_lp"] = float(arr_lp.mean())
        if "var_lp" in selected:
            result["var_lp"] = float(arr_lp.var())
        if "p5_lp" in selected:
            result["p5_lp"] = float(np.percentile(arr_lp, 5))
        if "p10_lp" in selected:
            result["p10_lp"] = float(np.percentile(arr_lp, 10))
        if "botk_lp" in selected:
            kk = min(k, len(arr_lp))
            result["botk_lp"] = float(np.sort(arr_lp)[:kk].mean()) if kk > 0 else None
        if "order_lp" in selected:
            w = np.array(order_weights(n))
            result["order_lp"] = float(np.dot(w, arr_lp)) if n > 0 else None
        if "semantic_lp" in selected:
            w = np.array(semantic_weights(tokens))
            result["semantic_lp"] = float(np.dot(w, arr_lp)) if n > 0 else None
        if "domain_lp" in selected:
            w = np.array(domain_lexical_weights(tokens, radlex_words))
            result["domain_lp"] = float(np.dot(w, arr_lp)) if n > 0 else None
    else:
        for m in lp_metrics:
            if m in selected:
                result[m] = None

    if need_ent and entropies:
        arr_ent = np.array(entropies, dtype=float)
        if "max_ent" in selected:
            result["max_ent"] = float(arr_ent.max())
        if "mean_ent" in selected:
            result["mean_ent"] = float(arr_ent.mean())
        if "var_ent" in selected:
            result["var_ent"] = float(arr_ent.var())
        if "p90_ent" in selected:
            result["p90_ent"] = float(np.percentile(arr_ent, 90))
        if "p95_ent" in selected:
            result["p95_ent"] = float(np.percentile(arr_ent, 95))
        if "topk_ent" in selected:
            kk = min(k, len(arr_ent))
            result["topk_ent"] = float(np.sort(arr_ent)[-kk:].mean()) if kk > 0 else None
        if "order_ent" in selected:
            w = np.array(order_weights(n))
            result["order_ent"] = float(np.dot(w, arr_ent)) if n > 0 else None
        if "semantic_ent" in selected:
            w = np.array(semantic_weights(tokens))
            result["semantic_ent"] = float(np.dot(w, arr_ent)) if n > 0 else None
        if "domain_ent" in selected:
            w = np.array(domain_lexical_weights(tokens, radlex_words))
            result["domain_ent"] = float(np.dot(w, arr_ent)) if n > 0 else None
    else:
        for m in ent_metrics:
            if m in selected:
                result[m] = None

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute logit-based confidence metrics from inference outputs."
    )
    parser.add_argument(
        "--predictions_path", required=True,
        help="Path to predictions JSONL from infer.py (must contain tokens, log_probs, entropies).",
    )
    parser.add_argument(
        "--judge_scores_path", required=True,
        help="Path to judge scores JSONL from evaluate.py (must contain id, score).",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Path to write logit_scores JSONL.",
    )
    parser.add_argument(
        "--lexicon_path", type=str, default="lexicon/RadLex.owl",
        help=(
            "Path to RadLex.owl for domain lexical weighting. "
            "Required to compute domain_lp and domain_ent. "
            "Falls back to a small hardcoded radiology term list if not provided."
        ),
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS,
        choices=ALL_METRICS,
        metavar="METRIC",
        help=(
            "Metrics to compute. Default: all. "
            "Choices: " + " ".join(ALL_METRICS)
        ),
    )
    parser.add_argument(
        "--findings_only", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Restrict all metrics to the FINDINGS section tokens only, "
            "excluding the section header and everything from IMPRESSION onward. "
            "Default: True (use --no-findings_only to score all tokens)."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of extreme tokens used by botk_lp and topk_ent (default: 5).",
    )
    args = parser.parse_args()

    # Load RadLex lexicon if provided
    radlex_words = None
    if args.lexicon_path:
        print(f"Loading RadLex lexicon from {args.lexicon_path} ...")
        radlex_words = load_radlex_lexicon(args.lexicon_path)
        print(f"  {len(radlex_words):,} unique RadLex terms loaded.")

    # Load judge scores indexed by id
    judge = {}
    with open(args.judge_scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                judge[str(rec["id"])] = rec

    selected = args.metrics

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    n_written = 0
    with open(args.predictions_path) as in_f, open(args.output_path, "w") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = str(record["id"])
            j = judge.get(sid, {})

            out = {
                "id": sid,
                "score": j.get("score"),
            }
            out.update(compute_record_metrics(
                record, selected, radlex_words,
                findings_only=args.findings_only, k=args.k,
            ))
            out_f.write(json.dumps(out) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {args.output_path}")


if __name__ == "__main__":
    main()
