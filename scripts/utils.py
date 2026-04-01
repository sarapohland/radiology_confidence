"""
Shared utilities for the confidence estimation pipeline.

Functions
---------
Report parsing
    extract_field(report_text, field) -> str
    extract_findings(text) -> str
    extract_findings_text(text) -> str
    findings_slice(tokens) -> (int, int)
    load_ground_truth(data_dir, study_id) -> (str, str)

I/O
    load_study_ids(ids_path) -> list
    load_jsonl(path) -> list[dict]
    load_jsonl_by_id(path) -> dict

Study data
    discover_series(data_dir, study_id) -> list[Path]

Image helpers
    mp4_frames_to_pil(video_path) -> list[PIL.Image]
    pil_to_data_uri(img) -> str
    frames_to_data_uris(video_paths, perturb_fn=None) -> list[str]

ROUGE
    rouge1_f1(reference, hypothesis) -> float
    rougeL_f1(reference, hypothesis) -> float
    mean_pairwise_rouge(texts) -> (float, float)

Metric metadata (shared by analyze.py and calibrate.py)
    _METADATA_FIELDS
    detect_score_type(input_path) -> str
    metric_label(field) -> str
    get_metric_fields(records) -> list[str]

PR / ROC metrics
    compute_pr_metrics(scores, labels, negate=False) -> dict
    compute_roc_metrics(scores, labels, negate=False) -> dict
"""

import base64
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve


# ---------------------------------------------------------------------------
# Report parsing helpers
# ---------------------------------------------------------------------------

_ANATOMICAL_SECTIONS = re.compile(
    r"\n[ \t]*(LUNGS|MEDIASTINUM|HEART|PLEURA|BONES|PULMONARY|VASCULAR|SOFT TISSUE|DIAPHRAGM)\s*:",
    re.IGNORECASE,
)


def extract_field(report_text: str, field: str) -> str:
    """Extract the text of a named field from a structured report.txt.

    Handles both mixed-case titled format ('Findings:') and all-caps format
    ('FINDINGS:'), as well as 'HISTORY:' as an alias for 'Indication'.
    """
    aliases = [re.escape(field)]
    if field.lower() == "indication":
        aliases.append("HISTORY")
    candidates = "|".join(aliases)

    pattern = re.compile(
        rf"(?:{candidates})\s*:\s*(.*?)(?=\n[ \t]*[A-Za-z][A-Za-z0-9 \t]*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(report_text)
    if not m:
        return ""
    return " ".join(m.group(1).split())


def extract_findings(text: str) -> str:
    """Extract findings from a ground-truth report.txt.

    Priority:
    1. 'Findings:' / 'FINDINGS:' standard header.
    2. Anatomical-section format (LUNGS:/MEDIASTINUM:/etc.) — concatenate all
       such sections as the findings body.
    """
    findings = extract_field(text, "Findings")
    if findings:
        return findings

    first_match = _ANATOMICAL_SECTIONS.search(text)
    if first_match:
        body = text[first_match.start():]
        return " ".join(body.split())

    return ""


def extract_findings_text(text: str) -> str:
    """Return the FINDINGS body from a model-generated response.

    Extracts text between 'FINDINGS:' (exclusive) and 'IMPRESSION:' (exclusive).
    Falls back to the full text if neither marker is present.
    """
    fm = re.search(r"FINDINGS\s*:\s*", text, re.IGNORECASE)
    im = re.search(r"IMPRESSION\s*:", text, re.IGNORECASE)
    start = fm.end() if fm else 0
    end = im.start() if im else len(text)
    extracted = text[start:end].strip()
    return extracted if extracted else text


def findings_slice(tokens: list) -> tuple:
    """Return (start, end) token indices covering the FINDINGS body only.

    Excludes the "FINDINGS:" header tokens and everything from "IMPRESSION:"
    onward. Falls back to (0, len(tokens)) if section markers are absent.
    """
    full = "".join(tokens)
    fm = re.search(r"FINDINGS\s*:\s*", full, re.IGNORECASE)
    im = re.search(r"IMPRESSION\s*:", full, re.IGNORECASE)
    start_char = fm.end() if fm else 0
    end_char = im.start() if im else len(full)

    cumlen = [0]
    for t in tokens:
        cumlen.append(cumlen[-1] + len(t))

    start_tok = next(
        (i for i in range(len(tokens)) if cumlen[i + 1] > start_char), 0
    )
    end_tok = next(
        (i for i in range(len(tokens)) if cumlen[i] >= end_char), len(tokens)
    )
    return start_tok, end_tok


def load_ground_truth(data_dir: Path, study_id: str) -> tuple:
    """Return (findings, indication) from the study's report.txt."""
    report_path = data_dir / study_id / "report.txt"
    if not report_path.exists():
        return "", ""
    text = report_path.read_text(encoding="utf-8")
    return extract_findings(text), extract_field(text, "Indication")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_study_ids(ids_path: Path) -> list:
    with open(ids_path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return records as a list."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_jsonl_by_id(path: Path) -> dict:
    """Load a JSONL file and index records by their 'id' field."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[str(r["id"])] = r
    return records


# ---------------------------------------------------------------------------
# Study data helpers
# ---------------------------------------------------------------------------

def discover_series(data_dir: Path, study_id: str) -> list:
    """Return sorted list of volume.mp4 paths under data_dir/study_id/."""
    study_path = data_dir / study_id
    if not study_path.is_dir():
        return []
    return sorted(study_path.rglob("volume.mp4"))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def mp4_frames_to_pil(video_path: Path) -> list:
    """Decode an MP4 and return each frame as a PIL Image."""
    import av
    container = av.open(str(video_path))
    frames = [frame.to_image() for frame in container.decode(video=0)]
    container.close()
    return frames


def pil_to_data_uri(img: Image.Image) -> str:
    """Encode a PIL image as a PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def frames_to_data_uris(video_paths: list, perturb_fn=None) -> list:
    """Decode all MP4s in video_paths and return frames as data URIs.

    Parameters
    ----------
    video_paths : list[Path]
        Paths to volume.mp4 files.
    perturb_fn : callable, optional
        Applied to each PIL frame before encoding (e.g. add noise or blur).
    """
    uris = []
    for vp in video_paths:
        for img in mp4_frames_to_pil(vp):
            if perturb_fn is not None:
                img = perturb_fn(img)
            uris.append(pil_to_data_uri(img))
    return uris


# ---------------------------------------------------------------------------
# ROUGE (no external dependencies beyond stdlib)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _lcs_length(a: list, b: list) -> int:
    """Length of longest common subsequence via O(n) space DP."""
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge1_f1(reference: str, hypothesis: str) -> float:
    """Unigram F1 between reference and hypothesis."""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    overlap = sum(min(ref_counts[t], hyp_counts[t]) for t in hyp_counts)
    p = overlap / len(hyp)
    r = overlap / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def rougeL_f1(reference: str, hypothesis: str) -> float:
    """Longest-common-subsequence F1 between reference and hypothesis."""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    p = lcs / len(hyp)
    r = lcs / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mean_pairwise_rouge(texts: list) -> tuple:
    """Mean pairwise ROUGE-1 and ROUGE-L F1 over all unique pairs.

    Returns (mean_rouge1, mean_rougeL). Returns (0.0, 0.0) if fewer than 2 texts.
    """
    r1_vals, rL_vals = [], []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            r1_vals.append(rouge1_f1(texts[i], texts[j]))
            rL_vals.append(rougeL_f1(texts[i], texts[j]))
    if not r1_vals:
        return 0.0, 0.0
    return float(np.mean(r1_vals)), float(np.mean(rL_vals))


# ---------------------------------------------------------------------------
# Metric metadata (shared by analyze.py and calibrate.py)
# ---------------------------------------------------------------------------

_METADATA_FIELDS = {"id", "score", "question_scores", "processing_time"}

_KNOWN_TYPES = ["logit", "grounding", "consistency", "stability", "probe"]


def detect_score_type(input_path) -> str:
    stem = Path(input_path).stem  # e.g. "logit_scores"
    for t in _KNOWN_TYPES:
        if stem.startswith(t):
            return t
    return stem.split("_")[0]  # best-effort fallback


_METRIC_LABELS = {
    "min_lp":       "Minimum Log Probability",
    "mean_lp":      "Mean Log Probability",
    "var_lp":       "Log Probability Variance",
    "p5_lp":        "5th-Percentile Log Probability",
    "p10_lp":       "10th-Percentile Log Probability",
    "botk_lp":      "Bottom-k Mean Log Probability",
    "order_lp":     "Order-Weighted Log Probability",
    "semantic_lp":  "Semantic-Weighted Log Probability",
    "domain_lp":    "Domain-Weighted Log Probability",
    "max_ent":      "Maximum Entropy",
    "mean_ent":     "Mean Entropy",
    "var_ent":      "Entropy Variance",
    "p90_ent":      "90th-Percentile Entropy",
    "p95_ent":      "95th-Percentile Entropy",
    "topk_ent":     "Top-k Mean Entropy",
    "order_ent":    "Order-Weighted Entropy",
    "semantic_ent": "Semantic-Weighted Entropy",
    "domain_ent":   "Domain-Weighted Entropy",
}

_METRIC_ORDER = [
    "min_lp", "mean_lp", "var_lp", "p5_lp", "p10_lp", "botk_lp",
    "order_lp", "semantic_lp", "domain_lp",
    "max_ent", "mean_ent", "var_ent", "p90_ent", "p95_ent", "topk_ent",
    "order_ent", "semantic_ent", "domain_ent",
]


def metric_label(field: str) -> str:
    """Return a human-readable label for a metric field name."""
    return _METRIC_LABELS.get(field, field.replace("_", " ").title())


def get_metric_fields(records: list) -> list:
    """Return metric field names in canonical order (unknown fields appended alphabetically)."""
    if not records:
        return []
    present = set()
    for rec in records:
        for k, v in rec.items():
            if k not in _METADATA_FIELDS and isinstance(v, (int, float)) and v is not None:
                present.add(k)
    ordered = [m for m in _METRIC_ORDER if m in present]
    ordered += sorted(present - set(_METRIC_ORDER))
    return ordered


# ---------------------------------------------------------------------------
# PR and ROC metrics
# ---------------------------------------------------------------------------

def compute_pr_metrics(scores, labels, negate: bool = False) -> dict:
    """Compute PR curve metrics. Positive class = incorrect (label=1).

    Parameters
    ----------
    scores : array-like      raw metric values (or calibrated probabilities)
    labels : array-like      1 = incorrect, 0 = correct
    negate : bool            if True, flip scores before computing (for
                             higher-is-better metrics in analyze.py)

    Returns
    -------
    dict with precisions, recalls, auprc, p_at_90r, p_at_95r, p_at_99r
    """
    arr = np.array(scores, dtype=float)
    if negate:
        arr = -arr

    precision, recall, _ = precision_recall_curve(labels, arr)
    auprc = float(auc(recall, precision))

    def p_at_r(target: float) -> float:
        mask = recall >= target
        return float(precision[mask].max()) if mask.any() else 0.0

    return {
        "precisions": precision.tolist(),
        "recalls":    recall.tolist(),
        "auprc":      auprc,
        "p_at_90r":   p_at_r(0.90),
        "p_at_95r":   p_at_r(0.95),
        "p_at_99r":   p_at_r(0.99),
    }


def compute_roc_metrics(scores, labels, negate: bool = False) -> dict:
    """Compute ROC curve metrics. Positive class = incorrect (label=1).

    Parameters
    ----------
    scores : array-like      raw metric values (or calibrated probabilities)
    labels : array-like      1 = incorrect, 0 = correct
    negate : bool            if True, flip scores before computing

    Returns
    -------
    dict with fprs, tprs, auroc, sens_at_90spec, sens_at_95spec, sens_at_99spec
    """
    arr = np.array(scores, dtype=float)
    if negate:
        arr = -arr

    fpr, tpr, _ = roc_curve(labels, arr)
    auroc = float(auc(fpr, tpr))

    def sens_at_spec(target_spec: float) -> float:
        mask = fpr <= (1.0 - target_spec)
        return float(tpr[mask].max()) if mask.any() else 0.0

    return {
        "fprs":            fpr.tolist(),
        "tprs":            tpr.tolist(),
        "auroc":           auroc,
        "sens_at_90spec":  sens_at_spec(0.90),
        "sens_at_95spec":  sens_at_spec(0.95),
        "sens_at_99spec":  sens_at_spec(0.99),
    }
