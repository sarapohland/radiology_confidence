"""
Score MedGemma-generated reports against ground-truth findings using CRIMSON.
Reads predictions.jsonl produced by infer.py, loads the ground-truth Findings
and Indication from each study's report.txt, and runs CRIMSON batch evaluation.

Usage:
    python scripts/evaluate.py \
        --predictions_path outputs/predictions.jsonl \
        --data_dir data/xray_samples/studies/ \
        --ids_path data/xray_samples/selected_study_ids.json \
        --output_path outputs/judge_scores.jsonl \
        [--crimson_api hf|vllm] \
        [--batch_size 4]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CRIMSON_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CRIMSON_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from CRIMSON.generate_score import CRIMSONScore
from utils import load_ground_truth


# ---------------------------------------------------------------------------
# CRIMSON result → output record
# ---------------------------------------------------------------------------

def crimson_to_record(study_id: str, result: dict) -> dict:
    """Flatten a CRIMSON result dict into the judge_scores.jsonl schema."""
    ec = result.get("error_counts", {})
    return {
        "id": study_id,
        "score": result["crimson_score"],
        "question_scores": {
            "false_findings": ec.get("false_findings", 0),
            "missing_findings": ec.get("missing_findings", 0),
            "attribute_errors": ec.get("attribute_errors", 0),
            "location_errors": ec.get("location_errors", 0),
            "severity_errors": ec.get("severity_errors", 0),
            "descriptor_errors": ec.get("descriptor_errors", 0),
            "measurement_errors": ec.get("measurement_errors", 0),
            "certainty_errors": ec.get("certainty_errors", 0),
            "unspecific_errors": ec.get("unspecific_errors", 0),
            "overinterpretation_errors": ec.get("overinterpretation_errors", 0),
            "temporal_errors": ec.get("temporal_errors", 0),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score generated reports with CRIMSON.")
    parser.add_argument("--predictions_path", type=Path, default=Path("outputs/predictions.jsonl"),
                        help="JSONL produced by infer.py.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/xray_samples/studies"),
                        help="Path to studies directory (contains <study_id>/ subdirs).")
    parser.add_argument("--ids_path", type=Path, default=Path("data/xray_samples/selected_study_ids.json"),
                        help="JSON file containing the study IDs to evaluate.")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/judge_scores.jsonl"),
                        help="Output JSONL path.")
    parser.add_argument("--crimson_api", choices=["hf", "vllm"], default="hf",
                        help="CRIMSON backend: 'hf' (HuggingFace local) or 'vllm'.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Forward-pass batch size for HuggingFace inference.")
    args = parser.parse_args()

    # Load predictions indexed by study id
    predictions: dict[str, str] = {}
    with open(args.predictions_path) as f:
        for line in f:
            rec = json.loads(line)
            predictions[rec["id"]] = rec.get("response", "")
    print(f"Loaded {len(predictions)} predictions from {args.predictions_path}")

    # Restrict to requested study IDs, in order
    with open(args.ids_path) as f:
        study_ids = json.load(f)
    study_ids = [sid for sid in study_ids if sid in predictions]
    print(f"Evaluating {len(study_ids)} studies")

    # Resume: skip studies already written to the output file
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done: set[str] = set()
    if args.output_path.exists():
        with open(args.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_done.add(json.loads(line)["id"])
        if already_done:
            print(f"Resuming: {len(already_done)} studies already scored, skipping")

    # Load ground truth
    work: list[tuple[str, str, str, dict | None]] = []  # (sid, ref, pred, context)
    n_missing_gt = 0
    for sid in study_ids:
        if sid in already_done:
            continue
        findings, indication = load_ground_truth(args.data_dir, sid)
        if not findings:
            print(f"  [warn] No Findings in report.txt for {sid}, skipping")
            n_missing_gt += 1
            continue
        context = {"indication": indication} if indication else None
        work.append((sid, findings, predictions[sid], context))

    if n_missing_gt:
        print(f"  Skipped {n_missing_gt} studies with missing ground truth")

    if not work:
        print("Nothing to score.")
        return

    # Initialise CRIMSON scorer
    print(f"\nLoading CRIMSON (api={args.crimson_api}) ...")
    scorer = CRIMSONScore(api=args.crimson_api)

    # Evaluate one study at a time and write results immediately
    print(f"Scoring {len(work)} studies ...")
    t0 = time.time()
    n_ok = n_fail = 0
    scores = []

    with open(args.output_path, "a") as out_f:
        for sid, ref, pred, context in tqdm(work, unit="study"):
            try:
                result = scorer.evaluate(ref, pred, patient_context=context)
                record = crimson_to_record(sid, result)
                scores.append(result["crimson_score"])
                n_ok += 1
            except Exception as exc:
                print(f"  [warn] {sid} failed: {exc}")
                record = {"id": sid, "score": None, "question_scores": None}
                n_fail += 1
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {n_ok} succeeded, {n_fail} failed")
    if scores:
        print(f"Mean CRIMSON score: {sum(scores) / len(scores):.4f}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
